EVALUATE_PROMPT_0 = """根据原始需求，评估A和B的两种回答，哪种方式输出更满足要求。如果有参考答案，则严格参考标准答案的格式/内容。
# Requirement
{requirement}

# A
{sample}

# B
{new_sample}

# Golden answer
{answers}

给出你的分析和你认为更好的选择，使用XML标签封装

<analyse>一些分析</analyse>
<choose>A/B(你认为更好的答案)</choose>
"""

EVALUATE_PROMPT = """
Based on the original requirements, evaluate the two responses, A and B, and determine which one better meets the requirements. If a reference answer is provided, strictly follow the format/content of the reference answer.

# Requirement
{requirement}

# A
{sample}

# B
{new_sample}

# Golden answer
{answers}

Provide your analysis and the choice you believe is better, using XML tags to encapsulate your response.

<analyse>Some analysis</analyse>
<choose>A/B (the better answer in your opinion)</choose>
"""
