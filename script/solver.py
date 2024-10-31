from utils import response
from prompt.solver_prompt import SOLVE, CHECK


class Solver:
    def __init__(self, question="", model="gpt-4o", temperature=0.1, max_tokens=4096):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = [{"role": "system", "content": SOLVE}, {"role": "user", "content": question}, {"role": "assistant", "content": "<step_1>让我们开始一步一步解决</step_1>"}]
        self.question = question
        self.answer = None
        self.full_answer = "<step_1>让我们开始一步一步解决</step_1>"

    def check(self):
        # 构造消息内容

        messages = [{"role": "system", "content": CHECK},
                    {"role": "user", "content": f"原问题：\n{self.question}\n历史步骤：\n{self.full_answer}\n当前评审步骤：\n{self.answer}\n提示：\n{CHECK}"}]

        # 调用 response 函数获取答案
        check_answer = response.responser(
            messages=messages,
            model=self.model,
            temperature=0.3,
            max_tokens=self.max_tokens
        )

        judge = response.extract_content(check_answer, "judge")
        feedback = response.extract_content(check_answer, "feedback")
        if judge == "False" and feedback is not None:
            self.messages.append({"role": "user", "content": feedback})
            self.full_answer += f"\n<think>\n{feedback}/n</think>"
        if judge is None:
            judge = "None"

        print("////////////////////////////////////////////\n\n", self.full_answer)

        return judge, feedback

    def solve(self):
        # 构造消息内容

        # 调用 response 函数获取答案
        answer = response.responser(
            messages=self.messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        self.messages.append({"role": "assistant", "content": answer})
        self.answer = answer
        self.full_answer += f"\n{answer}"

        final_answer = response.extract_content(answer, "answer")

        return answer, final_answer



def main():
    question = """A list of positive integers has the following properties: $\bullet$ The sum of the items in the list is $30$. $\bullet$ The unique mode of the list is only $9$ without others. $\bullet$ The median of the list is must a positive integer that does not appear in the list itself. Find the sum of the squares of all the items in the list. """

    solver = Solver(question=question)
    for i in range(100):
        answer, final_answer = solver.solve()
        judge, _ = solver.check()
        if final_answer is not None and judge == "True":
            print(f"问题已解决，答案为：{final_answer}")
            return final_answer

    return "哦豁"


if __name__ == "__main__":
    main()
