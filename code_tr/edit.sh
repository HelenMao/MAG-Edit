#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python edit.py --source_prompt="there is a set of sofas on the red carpet in the living room"\
                --target_prompt="there is a set of sofas on the yellow carpet in the living room" \
                --target_word="yellow" \
                --img_path="examples/1/1.jpg"\
                --mask_path="examples/1/mask.png"\
                --result_dir="result"\
                --max_iteration=15\
                --scale=2.5