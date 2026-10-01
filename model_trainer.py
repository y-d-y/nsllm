import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from typing import Iterator
import math
import time

import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader


from model import NSConfig, NsModel, CUR_DIR



# 1. Dataset
class FineWebStreamingDataset(IterableDataset):

    def __init__(
        self,
        parquet_path: str,
        tokenizer_path: str,
        context_length: int = 2048,
        shuffle: bool = True,
        shuffle_buffer_size: int = 10_000,
        seed: int = 42,
        skip_sequences: int = 0,
    ):
        super().__init__()

        self.parquet_path = parquet_path
        self.tokenizer_path = tokenizer_path
        self.context_length = context_length
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.skip_sequences = skip_sequences


    def _create_dataset(self):

        dataset = load_dataset("parquet", data_files=self.parquet_path, split="train", streaming=True)

        if self.shuffle:
            dataset = dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)

        return dataset

    def __iter__(self) -> Iterator[dict]:

        worker_info = get_worker_info()

        if worker_info is None:
            # 单进程
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        dataset = self._create_dataset()

        if num_workers > 1:
            dataset = dataset.shard(num_shards=num_workers, index=worker_id)


        tokenizer = Tokenizer.from_file(self.tokenizer_path)
        eos_token_id = tokenizer.token_to_id("<eos>")

        if eos_token_id is None:
            raise ValueError(f"<eos> token not found in tokenizer: {self.tokenizer_path}") 

        buffer = []

        sequence_length = self.context_length + 1

        sequence_count = 0

        for row in dataset:
            text = row.get("text")
            if not text:
                continue

            token_ids = tokenizer.encode(text).ids

            if not token_ids:
                continue

            # 档之间加入 EOS, 防止不同 document 无边界地连接
            #
            # 例如：document A <eos> document B <eos>

            token_ids.append(eos_token_id)

            buffer.extend(token_ids)

            while len(buffer) >= sequence_length:

                tokens = buffer[: sequence_length]

                del buffer[: sequence_length]

                input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
                labels = torch.tensor(tokens[1:], dtype=torch.long)

                # 当前 sequence 是否属于需要跳过的验证集
                if sequence_count < self.skip_sequences:
                    sequence_count += 1
                    continue

                yield (input_ids, labels)


# 2. Learning Rate Schedule
def get_lr(step, max_lr, min_lr, warmup_steps, total_steps):

    # Warmup
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    # Cosine Decay
    progress = (step - warmup_steps) / (total_steps - warmup_steps)

    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    lr = min_lr + (max_lr - min_lr) * cosine
    return lr



# 3. 定期保存训练曲线到PNG
def plot_and_save(step, epoch, epochs, train_step_history, train_loss_history, eval_step_history, eval_loss_history, save_path, smooth_window):


    fig, ax = plt.subplots(figsize=(12, 5))

    # 训练 Loss
    ax.plot(train_step_history, train_loss_history, alpha=0.3, color="tab:blue", label="raw")

    # 平滑处理
    if len(train_loss_history) >= smooth_window:
        weights = torch.ones(smooth_window) / smooth_window
        smooth_loss = torch.conv1d(
            torch.tensor(train_loss_history).view(1, 1, -1),
            weights.view(1, 1, -1),
        ).flatten().numpy()
        smooth_steps = train_step_history[smooth_window - 1:]

        ax.plot(smooth_steps, smooth_loss, linewidth=2, label="smooth")

    # 验证Loss
    if len(eval_loss_history) > 0:
        ax.plot(eval_step_history, eval_loss_history, linewidth=2, marker="o", markersize=3, color="tab:orange", label="val")
        
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"Training Loss | Epoch {epoch + 1}/{epochs} | Step {step}"
    )
    ax.grid(True)
    ax.legend()

    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)   # 防止内存泄漏


# 10. 构建验证集缓存
def build_val_cache(dataset, max_sequences=10000):
    cache = []
    for i, (input_ids, labels) in enumerate(dataset):
        if i >= max_sequences:
            break
        cache.append((input_ids, labels))
    return cache


# 11. eval
def evaluate(model, val_dataset, device, batch_size=12):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i in range(0, len(val_dataset), batch_size):
            batch = val_dataset[i: i + batch_size]
            input_ids = torch.stack([x[0] for x in batch]).to(device)
            labels = torch.stack([x[1] for x in batch]).to(device)
            logits = model(input_ids)

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / max(count, 1)


# 12. train
def train():


    val_dataset = FineWebStreamingDataset(
        parquet_path=(
            f"{CUR_DIR}/data/pretrain/general_en/fineweb-6b/fineweb-6b.parquet"
        ),
        tokenizer_path=(
            "{CUR_DIR}/model/tokenizer.json"
        ),
        context_length=2048,
        shuffle=True,
        shuffle_buffer_size=10_000,
        skip_sequences=0,
    )

    # 先取前N条作为验证集并缓存
    eval_dataset = build_val_cache(val_dataset, max_sequences=10000)
    print(f"Validation cache size: {len(eval_dataset)} sequences")

    train_dataset = FineWebStreamingDataset(
        parquet_path=(
            f"{CUR_DIR}/data/pretrain/general_en/fineweb-6b/fineweb-6b.parquet"
        ),
        tokenizer_path=(
            "{CUR_DIR}/model/tokenizer.json"
        ),
        context_length=2048,
        shuffle=True,
        shuffle_buffer_size=10_000,
        seed=42,                    # 必须和 val 一样
        skip_sequences=10_000,      # 关键
    )

    # 再取后续数据集作为训练集
    train_loader = DataLoader(
        train_dataset,
        batch_size=12,
        num_workers=0,
        pin_memory=True
    )


    epochs = 1
    min_lr = 3e-5
    max_lr = 3e-4
    total_steps = 244_000
    warmup = 10_000

    device = "cuda:0"


    step = 0

    train_loss_history = []
    train_step_history = []
    log_interval = 10
    smooth_window = 10

    eval_loss_history = []
    eval_step_history = []
    eval_interval = 5000

    # 保存图片
    save_path = f"{CUR_DIR}/loss_curve_v2.png"
    save_interval = 100

    cfg = NSConfig()
    model = NsModel(cfg)

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)
    

    start_time = time.time()

    for epoch in range(epochs):

        print(f"Epoch: {epoch + 1}/{epochs} started")

        for input_ids, labels in train_loader:

            input_ids = input_ids.to(device=device)
            labels = labels.to(device=device)

            lr = get_lr(step, max_lr, min_lr, warmup, total_steps=total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            optimizer.zero_grad()

            logits = model(input_ids)

            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)

            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

            step += 1

            loss_value = loss.item()

            train_loss_history.append(loss_value)
            train_step_history.append(step)

            if step % log_interval == 0:
                
                cost_time = time.time() - start_time

                print(
                    f"Epoch: {epoch + 1}/{epochs}, "
                    f"Step: {step}, "
                    f"Loss: {loss_value:.6f}, "
                    f"LR: {lr:.8f}, "
                    f"Cost: {cost_time/3600:.2f}h"
                )

            if step % eval_interval == 0:
                t1 = time.time()
                eval_loss = evaluate(model, eval_dataset, device)
                eval_loss_history.append(eval_loss)
                eval_step_history.append(step)

                print(
                    f"Step: {step}, "
                    f"Train Loss: {loss_value:.6f}, "
                    f"Eval Loss: {eval_loss:.6f}, "
                    f"Eval Cost: {time.time() - t1:.2f}s"
                )

            if step % save_interval == 0:
                plot_and_save(step, epoch, epochs, train_step_history, train_loss_history, eval_step_history, eval_loss_history, save_path, smooth_window)

            if step == 100000:
                torch.save(model.state_dict(), f"{CUR_DIR}/model/ns_model_v2_epoch_{epoch + 1}_step_10w.bin")        

        print(f"Epoch: {epoch + 1}/{epochs} finished")

        torch.save(model.state_dict(), f"{CUR_DIR}/model/ns_model_v2_epoch_{epoch + 1}.bin")


if __name__ == "__main__":
    train()

    


