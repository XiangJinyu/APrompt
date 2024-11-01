EVALUATE_PROMPT = """根据原始需求，评估A和B的两种回答，哪种方式输出更满足要求。如果有参考答案，则严格参考标准答案的格式/内容。
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
