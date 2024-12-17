
import re
from openai import OpenAI
from utils import load
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict
import functools
import tiktoken
import time
from datetime import datetime


config = load.load_llm()
# 初始化 OpenAI 客户端
client = OpenAI(api_key=config['openai']['api_key'],
                base_url=config['openai']['base_url'])



@dataclass
class TokenStats:
    """Token统计信息"""
    input_tokens: int
    response_tokens: int
    total_tokens: int


@dataclass
class ModelUsage:
    """单个模型的使用统计"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0

    def add_usage(self, input_tokens: int, output_tokens: int):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += (input_tokens + output_tokens)
        self.call_count += 1


@dataclass
class TokenTracker:
    """Token使用跟踪器"""
    model_usage: Dict[str, ModelUsage] = field(default_factory=lambda: defaultdict(ModelUsage))
    start_time: datetime = field(default_factory=datetime.now)

    def add_usage(self, model: str, input_tokens: int, output_tokens: int):
        self.model_usage[model].add_usage(input_tokens, output_tokens)

    def get_total_usage(self) -> Dict[str, Dict]:
        """获取所有模型的使用统计"""
        usage = {}
        for model, stats in self.model_usage.items():
            usage[model] = {
                "input_tokens": stats.input_tokens,
                "output_tokens": stats.output_tokens,
                "total_tokens": stats.total_tokens,
                "call_count": stats.call_count
            }
        return usage

    def print_usage_report(self):
        """打印使用报告"""
        duration = datetime.now() - self.start_time
        print("\n=== Token Usage Report ===")
        print(f"Duration: {duration}")
        print("\nPer Model Statistics:")

        for model, stats in self.model_usage.items():
            print(f"\n{model}:")
            print(f"  Calls: {stats.call_count}")
            print(f"  Input Tokens: {stats.input_tokens:,}")
            print(f"  Output Tokens: {stats.output_tokens:,}")
            print(f"  Total Tokens: {stats.total_tokens:,}")

        total_cost = self.calculate_estimated_cost()
        if total_cost:
            print(f"\nEstimated Total Cost: ${total_cost:.2f}")

    def calculate_estimated_cost(self) -> float:
        """计算估算成本 (可根据实际价格调整)"""
        # 模型价格配置 (USD per 1K tokens)
        prices = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }

        total_cost = 0.0
        for model, stats in self.model_usage.items():
            if model in prices:
                price = prices[model]
                input_cost = (stats.input_tokens / 1000) * price["input"]
                output_cost = (stats.output_tokens / 1000) * price["output"]
                total_cost += input_cost + output_cost

        return total_cost


# 创建全局token跟踪器实例
token_tracker = TokenTracker()


@dataclass
class LLMResponse:
    """LLM响应结果，包含响应内容和token统计"""
    content: Optional[str]
    token_stats: TokenStats


def check_tokens(max_input_tokens: Optional[int] = None,
                 encoding_name: str = "cl100k_base"):

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
            # 获取model参数
            model = kwargs.get("model", "unknown-model")

            # 原有的token计算逻辑
            messages = None
            if args:
                messages = args[0]
            elif "messages" in kwargs:
                messages = kwargs["messages"]

            if not messages:
                response = await func(*args, **kwargs)
                return LLMResponse(content=response, token_stats=TokenStats(0, 0, 0))

            encoding = tiktoken.get_encoding(encoding_name)
            input_tokens = calculate_message_tokens(messages, encoding)

            if max_input_tokens and input_tokens > max_input_tokens:
                raise ValueError(f"Input messages token count ({input_tokens}) exceeds limit ({max_input_tokens})")

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


# 使用示例:
@check_tokens()
async def responser(messages, model, temperature=0.3, max_tokens=4096, max_retries=10):
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
    return None


# 使用示例:
async def main():
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]

    # 调用API
    result = await responser(messages, model="gpt-4o")
    print(f"Response: {result.content}")
    print(result.token_stats.input_tokens)
    result = await responser(messages, model="gpt-4o")
    print(f"Response: {result.content}")
    print(result.token_stats.input_tokens)
    result = await responser(messages, model="gpt-4o-mini")
    print(f"Response: {result.content}")
    print(result.token_stats.input_tokens)

    # 在程序结束时打印统计信息
    token_tracker.print_usage_report()


# 运行程序
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())



# if __name__ == "__main__":
#     content = """<analyse>
# 分析参考的prompt产生的结果还有哪些缺点以及如何改进。
#
# 1. **冗长性**：参考prompt的回答过于详细，包含了许多不必要的细节，如工作时间、社交媒体账号等，这些信息在某些情况下可能并不需要。
# 2. **一致性**：不同问题的回答格式不一致，有的使用编号，有的没有，这可能会让用户感到困惑。
# 3. **简洁性**：参考prompt的回答可以更加简洁明了，直接给出关键步骤，避免过多的解释和背景信息。
# 4. **用户友好性**：回答中可以增加一些用户友好的提示，如“如果您在操作过程中遇到任何问题，请随时联系我们的客户支持团队获取帮助。”
#
# 改进方向：简化回答，保持一致的格式，增加用户友好的提示。
# </analyse>
#
# <modification>
# 简化回答，保持一致的格式，增加用户友好的提示。
# </modification>
#
# <prompt>
# 你是一个常见问题解答 (FAQ) 系统。给出简洁且准确的回复。根据用户问题给出关键步骤的答复。用户问题：{question}
#
# 例如：
# - 如何重置我的密码？
#   1. 访问登录页面。
#   2. 点击“忘记密码？”链接。
#   3. 输入您的注册邮箱地址。
#   4. 检查您的邮箱，查收重置密码的邮件。
#   5. 按照邮件中的链接和说明进行操作。
#
# - 如何更新我的账户信息？
#   1. 登录您的账户。
#   2. 进入“账户设置”页面。
#
# </prompt>"""
#     prompt = calculate_tokens(content)
#     print(prompt)





