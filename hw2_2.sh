#!/bin/bash
wget https://www.dropbox.com/s/0o49vkzhdg8fkyt/part2.pth?dl=0 -O hw2_2_best_r10921a36.pt
# TODO - run your inference Python3 code
python3 part2_test.py --output_dir $1 --model_file hw2_2_best_r10921a36.pt