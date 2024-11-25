# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 20:00 PM
# @Author  : didi
# @Desc    : Entrance of AFlow.

from script.optimizer import Optimizer


if __name__ == "__main__":

    optimizer = Optimizer(
        optimized_path="workspace",
        initial_round=10,
        max_rounds=10,
        name="paper_classify_4omini",
        optimize_model="claude-3-5-sonnet-20240620",
        execute_model="gpt-4o-mini"
    )

    optimizer.optimize()