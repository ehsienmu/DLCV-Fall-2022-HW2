#!/bin/bash
wget https://www.dropbox.com/s/vv8lbj1yxgajwig/part1_Best.pth?dl=0 -O hw2_1_best_r10921a36.pt
# TODO - run your inference Python3 code
python3 part1_test.py --output_dir $1 --model_file hw2_1_best_r10921a36.pt