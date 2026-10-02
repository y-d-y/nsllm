import json
import math
import time
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer
from datasets import load_dataset

from model import NsModel, NSConfig, CUR_DIR
from model_pre_trainer import plot_and_save


def prepare_sft_model(model_path: str, device: str):
    """ SFT 阶段新增im_start、im_end 对话模板 special token """

    tokenizer = Tokenizer.from_file(f"{CUR_DIR}/model/tokenizer_fixed.json")
    tokenizer.add_special_tokens(["<|im_start|>", "<|im_end|>"])
    tokenizer.save(f"{CUR_DIR}/model/tokenizer_sft.json")


    cfg = NSConfig()
    model = NsModel(cfg)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    cfg.vocab_size = cfg.vocab_size + 2

    # 1. 扩展embeddings
    old_emb = model.emb
    old_vocab, dim = old_emb.weight.shape

    new_emb = nn.Embedding(old_vocab + 2, dim)

    # Embedding、lm_out两个权重矩阵的行（第0维）都对应"词表维度"
    new_emb.weight.data[:old_vocab] = old_emb.weight.data 
    # 方式1：使用其他token特征的平均值
    # new_emb.weight.data[old_vocab:] = old_emb.weight.data.mean(dim=0)
    # 方式2：special token 是结构性 token，推荐随机初始化
    nn.init.normal_(
        new_emb.weight[old_vocab:],
        mean=0,
        std=0.02
    )

    model.emb = new_emb.to(device)

    # 2. 扩展lm_head
    old_lm_out = model.lm_out
    new_lm_out = nn.Linear(dim, old_vocab + 2, bias=False)

    new_lm_out.weight.data[:old_vocab, :] = old_lm_out.weight.data
    # new_lm_out.weight.data[old_vocab:, :] = old_lm_out.weight.data.mean(dim=0)
    nn.init.normal_(
        new_lm_out.weight[old_vocab:],
        mean=0,
        std=0.02
    )

    model.lm_out = new_lm_out.to(device)

    print(model.emb.weight.shape)
    print(model.lm_out.weight.shape)

    torch.save(model.state_dict(), f"{CUR_DIR}/model/sft_model.pt")

    with open(f"{CUR_DIR}/model/config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    

class SFTDataset(Dataset):

    def __init__(self, dataset: Dataset, tokenizer, max_length=2048):
        
        self.dataset = dataset
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.system_prompt = "You are a helpful assistant."

        self.buffer = []
        
        self.stats = {
            "total": 0,
            "invalid": 0,
            "user_too_long": 0,
            "too_long": 0,
            "kept": 0,
        }

        self._load()

    def _load(self):
        for idx in range(len(self.dataset)):

            self.stats["total"] += 1

            conversations = self.dataset[idx]["conversations"]
            if len(conversations) != 2:
                self.stats["invalid"] += 1
                continue

            from_human = conversations[0]["value"]
            from_gpt = conversations[1]["value"]

            user_text = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{from_human}<|im_end|>\n"
            gpt_text = f"<|im_start|>assistant\n{from_gpt}<|im_end|>"
            

            user_ids = self.tokenizer.encode(user_text).ids
            gpt_ids = self.tokenizer.encode(gpt_text).ids

            if len(user_ids) >= self.max_length:
                self.stats["user_too_long"] += 1
                continue


            if len(user_ids) + len(gpt_ids) > self.max_length:
                self.stats["too_long"] += 1
                continue

            input_ids = user_ids + gpt_ids

            labels = [-100] * len(user_ids) + gpt_ids

            input_tensor = torch.tensor(input_ids, dtype=torch.long)
            label_tensor = torch.tensor(labels, dtype=torch.long)
                
            self.buffer.append((input_tensor, label_tensor))

            self.stats["kept"] += 1

        print(f"SFTDataset: {self.stats}")
            

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]


class SFTCollator:

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        """
            batch 是 DataLoader 根据 batch_size=12 从 Dataset 中取出的 12 个样本组成的 list。
            batch = [[input_ids, label_ids], [input_ids, label_ids], ...]
        """
        # >>> batch = [[1, 'a'], [2, 'b']]
        # >>> c, d = zip(*a)
        # >>> c
        # (1, 2)
        # >>> d
        # ('a', 'b')

        input_ids, label_ids = zip(*batch)

        max_len = max(x.size(0) for x in input_ids)

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for ids, lbs in zip(input_ids, label_ids):
            pad_len = max_len - len(ids)

            batch_input_ids.append(
                torch.cat([
                    ids, 
                    torch.full((pad_len,), self.pad_token_id, dtype=torch.long)  # NOTE: 此处使用pad_token_id填充，此处要通过Embeddings层计算token的embedding， 所以需要使用pad_token_id的embedding
                ])
            )
            batch_labels.append(
                torch.cat([
                    lbs, 
                    torch.full((pad_len,), -100, dtype=torch.long)  # NOTE: 关键修正, label补齐长度时不要使用pad_token_id, 直接使用-100， 方便CrossEntropyLoss计算loss时ignore_index=-100的token
                ])  
            )

            batch_attention_mask.append(
                torch.cat([
                    torch.ones(
                        len(ids),
                        dtype=torch.long
                    ),
                    torch.zeros(
                        pad_len,
                        dtype=torch.long
                    )
                ])
            )
        
        return {
            "input_ids": torch.stack(
                batch_input_ids
            ),
            "labels": torch.stack(
                batch_labels
            ),
            "attention_mask": torch.stack(
                batch_attention_mask
            ),
        }


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


def evaluate(model, eval_loader):

    model.eval()
    with torch.no_grad():
        count = 0
        total_loss = 0.0
        for i, batch in enumerate(eval_loader):
            
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            logits = model.forward(input_ids, past_key_values=None, use_cache=False, attention_mask=attention_mask)
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=-100)

            total_loss = total_loss + loss.item()
            count += 1

    model.train()
    return total_loss / max(count, 1)



def train():


    epochs = 1
    batch_size = 12
    device = "cuda:1"


    tokenizer = Tokenizer.from_file(f"{CUR_DIR}/model/tokenizer_sft.json")
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    im_end_id = tokenizer.token_to_id("<|im_end|>")
    print(f"""tokenizer loaded, <|im_start|> token_id = {im_start_id}, <|im_end|> token_id = {im_end_id} """)


    data_files = [

        f"{CUR_DIR}/data/sft/Magpie-Pro-300K-Filtered/data/train-00000-of-00003.parquet",
        f"{CUR_DIR}/data/sft/Magpie-Pro-300K-Filtered/data/train-00001-of-00003.parquet",
        f"{CUR_DIR}/data/sft/Magpie-Pro-300K-Filtered/data/train-00002-of-00003.parquet"
    ]


    dataset_dict = load_dataset("parquet", data_files=data_files)

    split = dataset_dict["train"].train_test_split(test_size=0.1, seed=42)
    train_split, eval_split = split["train"], split["test"]

    train_dataset_len = len(train_split)
    eval_dataset_len = len(eval_split)
    print(f"train_dataset_len = {train_dataset_len}, eval_dataset_len = {eval_dataset_len}")    

    start_time = time.time()

    train_ds = SFTDataset(train_split, tokenizer)
    eval_ds = SFTDataset(eval_split, tokenizer)

    collator = SFTCollator(
        pad_token_id=tokenizer.token_to_id("<pad>")
    )


    eva_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )

    for item in eva_loader:
        input_ids = item["input_ids"]
        labels = item["labels"]
        attention_mask = item["attention_mask"]
        print(input_ids.shape)
        print(labels.shape)
        print(attention_mask.shape)
        break


    print(f"train/val dataset prepare done, cost: {time.time() - start_time: .3f}s")

  
    with open(f"{CUR_DIR}/model/config.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    cfg = NSConfig(**data)
    model = NsModel(cfg)
    model.to(device)

    sft_model_path = f"{CUR_DIR}/model/sft_model.pt"
    model.load_state_dict(torch.load(sft_model_path, map_location=device))
    model.to(device)
    model.train()

    print(f"load model: {sft_model_path} done")
    total_steps = (train_dataset_len // batch_size) * epochs   
    warmup_steps = total_steps * 0.05
    max_lr = 3e-5
    min_lr = 3e-6

    eval_step = 1000
    log_step = 10
    curve_loss_step = 100

    save_steps = (int(total_steps * 0.5), int(total_steps * 0.8)) 

    step = 0

    print(f"total_steps = {total_steps}, warmup_steps = {warmup_steps}, max_lr = {max_lr}, min_lr = {min_lr}")


    train_step_history = []
    train_loss_history = [] 
    eval_step_history = []
    eval_loss_history = []
    save_path = f"{CUR_DIR}/sft_loss_curve.png"
    smooth_window = 10


    optimzer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)

    for epoch in range(epochs):
        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimzer.zero_grad()

            lr = get_lr(step, max_lr, min_lr, warmup_steps, total_steps)

            for param_group in optimzer.param_groups:
                param_group["lr"] = lr


            logits = model.forward(input_ids, past_key_values=None, use_cache=False, attention_mask=attention_mask)

            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=-100)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimzer.step()

            step += 1


            train_step_history.append(step)
            train_loss_history.append(loss.item())

            if step % log_step == 0:
                print(f"Epoch: {epoch + 1}/{epochs}, Step: {step}, Loss: {loss.item(): .6f}, LR: {lr:.8f}")

            if step % eval_step == 0:
                eval_loss = evaluate(model, eva_loader)
                print(f"Epoch: {epoch + 1}/{epochs}, Step: {step}, Eval Loss: {eval_loss: .6f}")
                eval_step_history.append(step)
                eval_loss_history.append(eval_loss)

            if step % curve_loss_step == 0:
                # TODO 更新loss曲线
                plot_and_save(
                    step, epoch, epochs, train_step_history, train_loss_history, eval_step_history, eval_loss_history, save_path, smooth_window
                )

            if step in save_steps:
                print(f"save model weight at step {step}")
                torch.save(model.state_dict(), f"{CUR_DIR}/model/sft_model_step_{step}.pt")

    torch.save(model.state_dict(), f"{CUR_DIR}/model/sft_model_step_{step}.pt")



if __name__ == "__main__": 
    # prepare_sft_model(f"{CUR_DIR}/model/ns_model_v2_epoch_1_step_10w.bin", device="cuda:1")
    train()