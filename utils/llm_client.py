from openai import OpenAI
from utils import load
from dataclasses import dataclass
from typing import Dict, List, Optional
import functools
import tiktoken
import time
from utils.token_manager import get_token_tracker
import re

# 配置加载
config = load.load_llm()
client = OpenAI(api_key=config['openai']['api_key'],
                base_url=config['openai']['base_url'])


@dataclass
class TokenStats:
    """Token统计信息"""
    input_tokens: int
    response_tokens: int
    total_tokens: int


@dataclass
class LLMResponse:
    """LLM响应结果，包含响应内容和token统计"""
    content: Optional[str]
    token_stats: TokenStats


def check_tokens(max_input_tokens: Optional[int] = None,
                 encoding_name: str = "cl100k_base"):
    """
    装饰器: 用于检查和计算LLM输入消息的token数量，并返回token统计
    """

    def calculate_message_tokens(messages: List[Dict], encoding) -> int:
        """计算消息列表的总token数"""
        total_tokens = 0
        for message in messages:
            content = message.get("content", "")
            role = message.get("role", "")
            tokens = len(encoding.encode(content)) + len(encoding.encode(role))
            total_tokens += tokens
        return total_tokens

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # 获取token跟踪器实例
            token_tracker = get_token_tracker()

            # 获取model参数
            model = kwargs.get("model", "unknown-model")

            messages = None
            if args:
                messages = args[0]
            elif "messages" in kwargs:
                messages = kwargs["messages"]

            if not messages:
                response = await func(*args, **kwargs)
                return LLMResponse(
                    content=response,
                    token_stats=TokenStats(0, 0, 0)
                )

            encoding = tiktoken.get_encoding(encoding_name)
            input_tokens = calculate_message_tokens(messages, encoding)

            if max_input_tokens and input_tokens > max_input_tokens:
                raise ValueError(
                    f"Input messages token count ({input_tokens}) "
                    f"exceeds limit ({max_input_tokens})"
                )

            response = await func(*args, **kwargs)
            response_tokens = len(encoding.encode(response)) if response else 0
            total_tokens = input_tokens + response_tokens

            # 更新全局统计
            token_tracker.add_usage(model, input_tokens, response_tokens)

            return LLMResponse(
                content=response,
                token_stats=TokenStats(
                    input_tokens=input_tokens,
                    response_tokens=response_tokens,
                    total_tokens=total_tokens
                )
            )

        return wrapper

    return decorator

def extract_content(xml_string, tag):
    # 构建正则表达式，匹配指定的标签内容
    pattern = rf'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, xml_string, re.DOTALL)  # 使用 re.DOTALL 以匹配换行符
    return match.group(1).strip() if match else None


@check_tokens()
async def responser(messages, model, temperature=0.3, max_tokens=4096, max_retries=3):
    """LLM响应器"""
    retries = 0
    while retries < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
                max_tokens=max_tokens
            )
            response = completion.choices[0].message.content
            return response
        except Exception as e:
            print(f"Error occurred: {e}. Retrying... ({retries + 1}/{max_retries})")
            retries += 1
            time.sleep(5)

    print("Max retries reached. Failed to get a response.")
    return "None"


# 使用示例
async def main():
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]

    # 多次调用API测试
    for model in ["gpt-4o", "gpt-4o", "gpt-4o-mini"]:
        result = await responser(messages, model=model)
        print(f"\nModel: {model}")
        print(f"Response: {result.content}")
        print(f"Input tokens: {result.token_stats.input_tokens}")
        print(f"Response tokens: {result.token_stats.response_tokens}")

    # 打印最终统计信息
    get_token_tracker().print_usage_report()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())