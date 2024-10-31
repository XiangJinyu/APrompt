import csv
import os
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import time
from utils import load


config = load.load_llm()
# 初始化 OpenAI 客户端
client = OpenAI(api_key=config['openai']['api_key'],
                base_url=config['openai']['base_url'])


# 假设你有一个用于生成模型响应的函数
def responser(messages, model="deepseek-chat", temperature=0.3, max_tokens=4096, max_retries=10):
    retries = 0
    while retries < max_retries:
        try:
            # 调用 OpenAI API 生成回答
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
                max_tokens=max_tokens
            )

            response = completion.choices[0].message.content
            return response
        except Exception as e:
            print(f"Error occurred: {e}. Retrying... ({retries+1}/{max_retries})")
            retries += 1
            time.sleep(5)

    print("Max retries reached. Failed to get a response.")
    return None


def extract_content(xml_string, tag):
    # 构建正则表达式，匹配指定的标签内容
    pattern = f'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, xml_string)
    # 如果匹配成功，返回内容，否则返回None
    return match.group(1) if match else None



