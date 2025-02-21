from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch

# model_name = "Qwen/Qwen2.5-14B-Instruct-1M"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu")

prompt = "请谈谈大预言模型的发展与未来"
input = tokenizer(prompt, return_tensors="pt").to(model.device)

# 显性设置attention_mask
if "attention_mask" not in input:
    input["attention_mask"] = (input.input_ids != tokenizer.pad_token_id).int()

generate_kwargs = {
    "max_length": 100,  # 生成的最大长度
    "num_beams": 5,     # Beam Search 的 beam 数量
    "temperature": 0.7, # 控制生成多样性
    "top_k": 50,        # Top-k 采样
    "top_p": 0.9,       # Top-p 采样
    "do_sample": True,  # 是否使用采样
    "pad_token_id": tokenizer.eos_token_id,  # 设置填充符
}

with torch.no_grad():
    output = model.generate(
        input_ids=input["input_ids"], 
        attention_mask=input["attention_mask"],  # 显式传递 attention_mask
        **generate_kwargs
        )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
