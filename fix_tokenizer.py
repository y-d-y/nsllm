from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

tokenizer_path = "/home/sllm_scratch/D2L/nsllm/model/tokenizer.json"
tokenizer_path_fixed = "/home/sllm_scratch/D2L/nsllm/model/tokenizer_fixed.json"

tokenizer = Tokenizer.from_file(tokenizer_path)

print("Before:")
print("pre_tokenizer:", tokenizer.pre_tokenizer)
print("decoder:", tokenizer.decoder)

# 修复 decoder
tokenizer.decoder = ByteLevelDecoder()

# 保存
tokenizer.save(tokenizer_path_fixed)

print("After:")
print("pre_tokenizer:", tokenizer.pre_tokenizer)
print("decoder:", tokenizer.decoder)