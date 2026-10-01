from pathlib import Path
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAYON_NUM_THREADS"] = "40"

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

data_home = "/home/sllm_scratch/D2L/nsllm/data"


# 1. 创建 BPE 模型
tokenizer = Tokenizer(
    BPE(
        unk_token="<unk>"
    )
)


# 2. 设置预分词器
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)


# 3. 创建 BPE Trainer
trainer = BpeTrainer(
    vocab_size=32000,
    min_frequency=2,

    special_tokens=[
        "<unk>",
        "<pad>",
        "<bos>",
        "<eos>",
    ],

    initial_alphabet=ByteLevel.alphabet()
)




def iter_pretrain_en_fineweb():

    dataset = load_dataset(
        "parquet",
        data_files=f"{data_home}/pretrain/general_en/fineweb-6b/fineweb-6b.parquet",
        split="train",
    )

    for row in dataset:
        text = row["text"]
        if text:
            yield text


# def iter_pretrain_security_apt_data():
#     directory = Path(f"{data_home}/pretrain/security/apt")

#     for path in directory.glob("*.txt"):
#         print("Loading:", path)

#         with path.open(
#             "r",
#             encoding="utf-8",
#             errors="ignore",
#         ) as f:

#             for line in f:
#                 line = line.strip()

#                 if line:
#                     yield line


# def iter_pretrain_security_megavul():

#     dataset = load_dataset(
#         "parquet",
#         data_files=f"{data_home}/pretrain/security/megavul/data/*.parquet",
#         split="train",
#     )

#     for row in dataset:
#         if row:
#             yield json.dumps(row, ensure_ascii=False)



# 4. 开始训练
tokenizer.train_from_iterator(
    iter_pretrain_en_fineweb(),
    trainer=trainer,
)

        


# 5. 保存
tokenizer.save("/home/sllm_scratch/D2L/nsllm/model/tokenizer.json")