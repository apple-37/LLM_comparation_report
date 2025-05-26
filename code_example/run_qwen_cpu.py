from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

model_path = "/mnt/data/Qwen-7B-Chat"  # 本地模型路径，注意修改路径
# model_path = "/mnt/data/chatglm3-6B"
prompt = "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"
# ""中内容可以修改为任意提示词

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True   #可删，意在告诉它只用本地文件，不联网
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    # 可改为 torch_dtype="auto",
    local_files_only=True  #可删，意在告诉它只用本地文件，不联网
).eval()

inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

print("生成中：")
outputs = model.generate(
    inputs,
    streamer=streamer,
    max_new_tokens=300,
    do_sample=False
)
