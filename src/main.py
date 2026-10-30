# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SPARC', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, nargs='+', default=['baby'], help='name of datasets')
    # parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--mg', type=bool, default=False)
    
    config_dict = {
        'gpu_id': 0,
    }

    args, _ = parser.parse_known_args()
    # 遍历数据集列表
    for dataset in args.dataset:
        print(f"Training on dataset: {dataset}")
        quick_start(model=args.model, dataset=dataset, config_dict=config_dict, save_model=True, mg=args.mg)
    # quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True, mg=args.mg)

"""
autoDL:
cd /root/autodl-tmp/mmrec
python src/main.py --model SPARC --dataset baby
"""
