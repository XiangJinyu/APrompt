# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 20:00 PM
# @Author  : didi
# @Desc    : Entrance of AFlow.

import argparse
from typing import Dict, List
from script.optimizer import Optimizer


# def parse_args():
#     parser = argparse.ArgumentParser(description="AFlow Optimizer")
#     parser.add_argument("--sample", type=int, default=4, help="Sample count")
#     parser.add_argument(
#         "--optimized_path",
#         type=str,
#         default="workspace",
#         help="Optimized result save path",
#     )
#     parser.add_argument("--initial_round", type=int, default=1, help="Initial round")
#     parser.add_argument("--max_rounds", type=int, default=20, help="Max iteration rounds")
#     parser.add_argument("--check_convergence", type=bool, default=True, help="Whether to enable early stop")
#     parser.add_argument("--validation_rounds", type=int, default=5, help="Validation rounds")
#     parser.add_argument(
#         "--if_first_optimize",
#         type=lambda x: x.lower() == "true",
#         default=True,
#         help="Whether to download dataset for the first time",
#     )
#     return parser.parse_args()


if __name__ == "__main__":
    # args = parse_args()

    # optimizer = Optimizer(
    #     optimized_path=args.optimized_path,
    #     initial_round=args.initial_round,
    #     max_rounds=args.max_rounds,
    # )

    optimizer = Optimizer(
        optimized_path="workspace",
        initial_round=22,
        max_rounds=30,
        name="gsm8k_wo_short",
        optimize_model="claude-3-5-sonnet-20240620",
        execute_model="gpt-4o-mini"
    )

    # Optimize workflow via setting the optimizer's mode to 'Graph'
    optimizer.optimize()

    # Test workflow via setting the optimizer's mode to 'Test'
    # optimizer.optimize("Test")