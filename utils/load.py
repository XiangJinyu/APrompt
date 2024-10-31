import yaml
import random


def load_llm():
    # 读取上一级目录中的 YAML 配置文件
    with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config


def load_meta_data(k):
    # 读取 YAML 文件
    with open('../meta.yaml', 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    qa = []

    # 提取问题和答案
    for item in data['faq']:
        question = item['question']
        answer = item['answer']
        qa.append({'question': question, 'answer': answer})

    prompt = data['prompt']

    # 随机选择三组问答
    random_qa = random.sample(qa, min(k, len(qa)))  # 确保不超过列表长度

    return prompt, random_qa

