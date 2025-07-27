import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from time import gmtime, strftime
from datetime import datetime
import timeit
import random
import argparse
from opts import parse_opts
import tensorflow as tf
from train_data.get_img_dataset import get_img_dataset
from train_data.get_pat_dataset import get_pat_dataset
from train_data.preprocess_data import preprocess_data



if __name__ == '__main__':
    
    opt = parse_opts()
    
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    tf.random.set_seed(opt.manual_seed)

    # Add dataset selection: HeadNeck, Chest, Abdomen
    import sys
    dataset = 'Abdomen'  # default
    for i, arg in enumerate(sys.argv):
        if arg == '--dataset' and i+1 < len(sys.argv):
            dataset = sys.argv[i+1]
    print(f'Using dataset: {dataset}')

    if opt.root_dir is not None:
        if dataset == 'HeadNeck':
            out_dir = os.path.join(opt.root_dir, opt.HN_out)
            data_dir = os.path.join(opt.root_dir, opt.HN_data)
            label_dir = os.path.join(opt.root_dir, opt.HN_label)
            pro_data_dir = os.path.join(opt.root_dir, opt.HN_pro_data)
            pre_data_dir = os.path.join(opt.root_dir, opt.HN_pre_data)
            label_file = opt.HN_label_file
            crop_shape = opt.HN_crop_shape
            slice_range = opt.HN_slice_range
        elif dataset == 'Chest':
            out_dir = os.path.join(opt.root_dir, opt.CH_out)
            data_dir = os.path.join(opt.root_dir, opt.CH_data)
            label_dir = os.path.join(opt.root_dir, opt.CH_label)
            pro_data_dir = os.path.join(opt.root_dir, opt.CH_pro_data)
            pre_data_dir = os.path.join(opt.root_dir, opt.CH_pre_data)
            label_file = opt.CH_label_file
            crop_shape = opt.CH_crop_shape
            slice_range = opt.CH_slice_range
        elif dataset == 'Abdomen':
            out_dir = os.path.join(opt.root_dir, opt.AB_out)
            data_dir = os.path.join(opt.root_dir, opt.AB_data)
            label_dir = os.path.join(opt.root_dir, opt.AB_label)
            pro_data_dir = os.path.join(opt.root_dir, opt.AB_pro_data)
            pre_data_dir = os.path.join(opt.root_dir, opt.AB_pre_data)
            label_file = opt.AB_label_file
            crop_shape = opt.AB_crop_shape
            slice_range = opt.AB_slice_range
        else:
            raise ValueError(f'Unknown dataset: {dataset}')
        for d in [out_dir, data_dir, label_dir, pro_data_dir, pre_data_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

    print('\n--- STEP 1 - GET DATA ---\n')

    if opt.preprocess_data:
        preprocess_data(
            data_dir=data_dir,
            pre_data_dir=pre_data_dir,
            new_spacing=opt.new_spacing,
            data_exclude=opt.data_exclude,
            crop_shape=crop_shape)

    data_tot, label_tot, ID_tot = get_pat_dataset(
        data_dir=data_dir,
        pre_data_dir=pre_data_dir,
        label_dir=label_dir,
        label_file=label_file,
        pro_data_dir=pro_data_dir)

    get_img_dataset(
        pro_data_dir=pro_data_dir,
        run_type=None,
        data_tot=data_tot,
        ID_tot=ID_tot,
        label_tot=label_tot,
        slice_range=slice_range)



