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
import random
import argparse
from tensorflow.keras.optimizers import Adam
from go_model.tune_model import tune_model
from statistics.write_txt import write_txt
from statistics.get_stats_plots import get_stats_plots
from go_model.evaluate_model import evaluate_model
from train_data.tune_dataset import tune_pat_dataset
from train_data.tune_dataset import tune_img_dataset
from opts import parse_opts
import tensorflow as tf

  

if __name__ == '__main__':

    opt = parse_opts()

    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    tf.random.set_seed(opt.manual_seed)

    import sys
    dataset = 'Chest'  # default for tuning
    for i, arg in enumerate(sys.argv):
        if arg == '--dataset' and i+1 < len(sys.argv):
            dataset = sys.argv[i+1]
    print(f'Using target dataset for tuning: {dataset}')

    source_dataset = 'HeadNeck'
    if dataset == source_dataset:
        raise ValueError('Target dataset same as source')

    if opt.root_dir is not None:
        # Source directories
        if source_dataset == 'HeadNeck':
            source_model_dir = os.path.join(opt.root_dir, opt.HN_model)
        else:
            raise ValueError(f'Unknown source dataset: {source_dataset}')

        # Target directories
        if dataset == 'Chest':
            target_out_dir = os.path.join(opt.root_dir, opt.CH_out)
            target_data_dir = os.path.join(opt.root_dir, opt.CH_data)
            target_label_dir = os.path.join(opt.root_dir, opt.CH_label)
            target_pro_data_dir = os.path.join(opt.root_dir, opt.CH_pro_data)
            target_pre_data_dir = os.path.join(opt.root_dir, opt.CH_pre_data)
            target_model_dir = os.path.join(opt.root_dir, opt.CH_model)
            label_file = opt.CH_label_file
            crop_shape = opt.CH_crop_shape
            slice_range = opt.CH_slice_range
        elif dataset == 'Abdomen':
            target_out_dir = os.path.join(opt.root_dir, opt.AB_out)
            target_data_dir = os.path.join(opt.root_dir, opt.AB_data)
            target_label_dir = os.path.join(opt.root_dir, opt.AB_label)
            target_pro_data_dir = os.path.join(opt.root_dir, opt.AB_pro_data)
            target_pre_data_dir = os.path.join(opt.root_dir, opt.AB_pre_data)
            target_model_dir = os.path.join(opt.root_dir, opt.AB_model)
            label_file = opt.AB_label_file
            crop_shape = opt.AB_crop_shape
            slice_range = opt.AB_slice_range
        else:
            raise ValueError(f'Unknown target dataset: {dataset}')

        for d in [target_out_dir, target_data_dir, target_label_dir, target_pro_data_dir, target_pre_data_dir, target_model_dir]:
            if not os.path.exists(d):
                os.makedirs(d)
        if not os.path.exists(source_model_dir):
            os.makedirs(source_model_dir)
    
    print('\n--- STEP 4 - MODEL FINE TUNE ---\n')   
    
    # get target CT data and do preprocessing
    if opt.get_CH_data:
        tune_pat_dataset(
            data_dir=target_data_dir,
            pre_data_dir=target_pre_data_dir,
            pro_data_dir=target_pro_data_dir,
            label_dir=target_label_dir,
            label_file=label_file,
            crop_shape=crop_shape)
        tune_img_dataset(
            pro_data_dir=target_pro_data_dir,
            pre_data_dir=target_pre_data_dir,
            slice_range=slice_range)

    ## fine tune models by freezing some layers
    if opt.fine_tune:
        opt.run_type = 'tune'
        tuned_model, model_fn = tune_model(
            HN_model_dir=source_model_dir,
            CH_model_dir=target_model_dir, 
            pro_data_dir=target_pro_data_dir, 
            HN_model=opt.run_model,
            batch_size=opt.batch_size, 
            epoch=opt.epoch, 
            freeze_layer=opt.freeze_layer)
    
    ## evaluate finetuned model on target CT data
    if opt.test:
        opt.run_type = 'tune'
        loss, acc = evaluate_model(
            run_type=opt.run_type,
            model_dir=target_model_dir,
            pro_data_dir=target_pro_data_dir,
            saved_model=opt.tuned_model)
        if opt.stats_plots:
            get_stats_plots(
                pro_data_dir=target_pro_data_dir,
                root_dir=opt.root_dir,
                run_type=opt.run_type,
                run_model=opt.run_model,
                loss=0,
                acc=0,
                saved_model=opt.tuned_model,
                epoch=opt.epoch,
                batch_size=opt.batch_size,
                lr=opt.lr)


    
