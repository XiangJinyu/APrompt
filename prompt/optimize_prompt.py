PROMPT_OPTIMIZE_PROMPT = """You are building a prompt to address user requirement.Based on the given prompt, 
please reconstruct and optimize it. You can add, modify, or delete prompts. Please include a single modification in 
XML tags in your reply. During the optimization, you can incorporate any thinking models.
This is a prompt that performed excellently in a previous iteration. You must make further optimizations and improvements based on this prompt. The modified prompt must differ from the provided example.

requirements:
```
{requirements}
```

reference prompt:
```
{prompt}
```

The execution result of this reference prompt is(some cases):
```
{answers}
```

The best answer we expect(some cases):
```
{golden_answers}
```

Provide your analysis, optimization points, and the complete optimized prompt using the following XML format:

<analyse>Analyze what drawbacks exist in the results produced by the reference prompt and how to improve them.</analyse>
<modification>Summarize the key points for improvement in one sentence</modification>
<prompt>Provide the complete optimized prompt {count}</prompt>
"""


WORKFLOW_INPUT = """
Here is a graph and the corresponding prompt (prompt only related to the custom method) that performed excellently in a previous iteration (maximum score is 1). You must make further optimizations and improvements based on this graph. The modified graph must differ from the provided example, and the specific differences should be noted within the <modification>xxx</modification> section.\n
<sample>
    <experience>{experience}</experience>
    <modification>(such as:add a review step/delete a operator/modify a prompt)</modification>
    <score>{score}</score>
    <graph>{graph}</graph>
    <prompt>{prompt}</prompt>(only prompt_custom)
    <operator_description>{operator_description}</operator_description>
</sample>
Below are the logs of some results with the aforementioned Graph that performed well but encountered errors, which can be used as references for optimization:
{log}

First, provide optimization ideas. **Only one detail point can be modified at a time**, and no more than 5 lines of code may be changed per modification—extensive modifications are strictly prohibited to maintain project focus!
When introducing new functionalities in the graph, please make sure to import the necessary libraries or modules yourself, except for operator, prompt_custom, create_llm_instance, and CostManage, which have already been automatically imported.
**Under no circumstances should Graph output None for any field.**
Use custom methods to restrict your output format, rather than using code (outside of the code, the system will extract answers based on certain rules and score them).
It is very important to format the Graph output answers, you can refer to the standard answer format in the log.
"""

WORKFLOW_CUSTOM_USE = """\nHere's an example of using the `custom` method in graph:
```
# You can write your own prompt in <prompt>prompt_custom</prompt> and then use it in the Custom method in the graph
response = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
# You can also concatenate previously generated string results in the input to provide more comprehensive contextual information.
# response = await self.custom(input=problem+f"xxx:{xxx}, xxx:{xxx}", instruction=prompt_custom.XXX_PROMPT)
# The output from the Custom method can be placed anywhere you need it, as shown in the example below
solution = await self.generate(problem=f"question:{problem}, xxx:{response['response']}")
```
Note: In custom, the input and instruction are directly concatenated(instruction+input), and placeholders are not supported. Please ensure to add comments and handle the concatenation externally.\n

**Introducing multiple operators at appropriate points can enhance performance. If you find that some provided operators are not yet used in the graph, try incorporating them.**
"""

WORKFLOW_TEMPLATE = """from typing import Literal
import metagpt.ext.aflow.scripts.optimized.{dataset}.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.{dataset}.workflows.round_{round}.prompt as prompt_custom
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

{graph}
"""