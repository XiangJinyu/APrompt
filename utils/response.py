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
async def responser(messages, model, temperature=0.3, max_tokens=4096, max_retries=10):
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
    pattern = rf'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, xml_string, re.DOTALL)  # 使用 re.DOTALL 以匹配换行符
    return match.group(1).strip() if match else None


if __name__ == "__main__":
    content = """<analyse>
分析参考的prompt产生的结果还有哪些缺点以及如何改进。

1. **冗长性**：参考prompt的回答过于详细，包含了许多不必要的细节，如工作时间、社交媒体账号等，这些信息在某些情况下可能并不需要。
2. **一致性**：不同问题的回答格式不一致，有的使用编号，有的没有，这可能会让用户感到困惑。
3. **简洁性**：参考prompt的回答可以更加简洁明了，直接给出关键步骤，避免过多的解释和背景信息。
4. **用户友好性**：回答中可以增加一些用户友好的提示，如“如果您在操作过程中遇到任何问题，请随时联系我们的客户支持团队获取帮助。”

改进方向：简化回答，保持一致的格式，增加用户友好的提示。
</analyse>

<modification>
简化回答，保持一致的格式，增加用户友好的提示。
</modification>

<prompt>
你是一个常见问题解答 (FAQ) 系统。给出简洁且准确的回复。根据用户问题给出关键步骤的答复。用户问题：{question}

例如：
- 如何重置我的密码？
  1. 访问登录页面。
  2. 点击“忘记密码？”链接。
  3. 输入您的注册邮箱地址。
  4. 检查您的邮箱，查收重置密码的邮件。
  5. 按照邮件中的链接和说明进行操作。

- 如何更新我的账户信息？
  1. 登录您的账户。
  2. 进入“账户设置”页面。

</prompt>"""
    prompt = extract_content(content, tag="prompt")
    print(prompt)





