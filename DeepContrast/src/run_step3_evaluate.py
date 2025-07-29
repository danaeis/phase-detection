import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt
import glob
from time import gmtime, strftime
from datetime import datetime
import timeit
import yaml
import argparse
import random
from tensorflow.keras.optimizers import Adam
from go_model.evaluate_model import evaluate_model
from statistics.write_txt import write_txt
from statistics.get_stats_plots import get_stats_plots
from opts import parse_opts
import tensorflow as tf



if __name__ == '__main__':

    opt = parse_opts()

    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    tf.random.set_seed(opt.manual_seed)

    import sys
    dataset = 'Abdomen'  # default
    for i, arg in enumerate(sys.argv):
        if arg == '--dataset' and i+1 < len(sys.argv):
            dataset = sys.argv[i+1]
    print(f'Using dataset: {dataset}')

    if opt.root_dir is not None:
        if dataset == 'HeadNeck':
            out_dir = os.path.join(opt.root_dir, opt.HN_out)
            model_dir = os.path.join(opt.root_dir, opt.HN_model)
            pro_data_dir = os.path.join(opt.root_dir, opt.HN_pro_data)
        elif dataset == 'Chest':
            out_dir = os.path.join(opt.root_dir, opt.CH_out)
            model_dir = os.path.join(opt.root_dir, opt.CH_model)
            pro_data_dir = os.path.join(opt.root_dir, opt.CH_pro_data)
        elif dataset == 'Abdomen':
            out_dir = os.path.join(opt.root_dir, opt.AB_out)
            model_dir = os.path.join(opt.root_dir, opt.AB_model)
            pro_data_dir = os.path.join(opt.root_dir, opt.AB_pro_data)
        else:
            raise ValueError(f'Unknown dataset: {dataset}')
        for d in [out_dir, model_dir, pro_data_dir]:
            if not os.path.exists(d):
                os.makedirs(d)
     
    print('\n--- STEP 3 - MODEL EVALUATION ---\n')   
    
    for opt.run_type in ['val', 'test']:
        # evalute model
        loss, acc = evaluate_model(
            run_type=opt.run_type,
            model_dir=model_dir,
            pro_data_dir=pro_data_dir,
            saved_model=opt.run_model,
            threshold=opt.thr_img,
            activation=opt.activation)
        if opt.stats_plots:
            get_stats_plots(
                pro_data_dir=pro_data_dir,
                root_dir=opt.root_dir,
                run_type=opt.run_type,
                run_model=opt.run_model,
                loss=loss,
                acc=acc,
                saved_model=opt.run_model,
                epoch=opt.epoch,
                batch_size=opt.batch_size,
                lr=opt.lr,
                thr_img=opt.thr_img,
                thr_prob=opt.thr_prob,
                thr_pos=opt.thr_pos,
                bootstrap=opt.n_bootstrap)
