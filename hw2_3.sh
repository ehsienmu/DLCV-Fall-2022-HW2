#!/bin/bash
wget https://www.dropbox.com/s/ifggkjxsl6t822n/part3_mnistm_svhn_best.pt?dl=0 -O hw2_3_svhn_r10921a36.pt
wget https://www.dropbox.com/s/yaa5kz09hc2noiw/part3_mnistm_usps_best.pt?dl=0 -O hw2_3_usps_r10921a36.pt
# TODO - run your inference Python3 code
python3 part3_test.py --input_dir $1 --output_file $2 --svhn_model_file hw2_3_svhn_r10921a36.pt --usps_model_file hw2_3_usps_r10921a36.pt