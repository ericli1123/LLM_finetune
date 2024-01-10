import flask
import json
import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from component.utils import ModelUtils

app = flask.Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # 防止返回中文乱码


@app.route('/ljl', methods=['POST'])
def ds_llm():
    params = flask.request.get_json()
    inputs = params.pop('inputs').strip()

    # 生成超参配置
    max_new_tokens = 1024  # 每轮对话最多生成多少个token
    history_max_len = 1024  # 模型记忆的最大token长度
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0

    # chatglm使用官方的数据组织格式
    if model.config.model_type == 'chatglm':
        text = '[Round 1]\n\n问：{}\n\n答：'.format(inputs)
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    # 为了兼容qwen-7b，因为其对eos_token进行tokenize，无法得到对应的eos_token_id
    else:
        input_ids = tokenizer(inputs, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        bos_token_id = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
        eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(device)
        input_ids = torch.concat([bos_token_id, input_ids, eos_token_id], dim=1)

    logger.info(params)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_p=top_p,
                temperature=temperature, repetition_penalty=repetition_penalty, eos_token_id=tokenizer.eos_token_id)
    outputs = outputs.tolist()[0][len(input_ids[0]):]
    # response = tokenizer.batch_decode(outputs)
    response = tokenizer.decode(outputs)
    response = response.strip().replace(tokenizer.eos_token, "").strip()

    result = {
        'input': inputs,
        'output': response
    }
    with open(log_file, 'a', encoding='utf8') as f:
        data = json.dumps(result, ensure_ascii=False)
        f.write('{}\n'.format(data))

    return result


if __name__ == '__main__':
    # 参数设置
    model_name_or_path = "./premodel/"
    adapter_name_or_path = "output/sft/"

    log_file = 'service_history.txt'
    port = 8877

    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    device = 'cuda'

    logger.info(f"Starting to load the model {model_name_or_path} into memory")

    # 加载model和tokenizer
    model = ModelUtils.load_model(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id

    logger.info(f"Successfully loaded the model {model_name_or_path} into memory")

    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    print("Total model params: %.2fM" % (total / 1e6))
    model.eval()

    app.run(port=port)