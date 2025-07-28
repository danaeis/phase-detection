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
import pydot
import pydotplus
import graphviz
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import plot_model
from go_model.data_generator import train_generator
from go_model.data_generator import val_generator
from go_model.get_model import get_model
from go_model.train_model import train_model
from go_model.train_model import callbacks
from opts import parse_opts



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
            model_dir = os.path.join(opt.root_dir, opt.HN_model)
            label_dir = os.path.join(opt.root_dir, opt.HN_label)
            pro_data_dir = os.path.join(opt.root_dir, opt.HN_pro_data)
            log_dir = os.path.join(opt.root_dir, opt.HN_log)
        elif dataset == 'Chest':
            out_dir = os.path.join(opt.root_dir, opt.CH_out)
            model_dir = os.path.join(opt.root_dir, opt.CH_model)
            label_dir = os.path.join(opt.root_dir, opt.CH_label)
            pro_data_dir = os.path.join(opt.root_dir, opt.CH_pro_data)
            log_dir = os.path.join(opt.root_dir, opt.CH_log)
        elif dataset == 'Abdomen':
            out_dir = os.path.join(opt.root_dir, opt.AB_out)
            model_dir = os.path.join(opt.root_dir, opt.AB_model)
            label_dir = os.path.join(opt.root_dir, opt.AB_label)
            pro_data_dir = os.path.join(opt.root_dir, opt.AB_pro_data)
            log_dir = os.path.join(opt.root_dir, opt.AB_log)
        else:
            raise ValueError(f'Unknown dataset: {dataset}')
            
        for d in [out_dir, model_dir, label_dir, pro_data_dir, log_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

    print('\n--- STEP 2 - TRAIN MODEL ---\n')

    # data generator for train and val data
    train_gen = train_generator(
        pro_data_dir=pro_data_dir,
        batch_size=opt.batch_size)
    x_val, y_val, val_gen = val_generator(
        pro_data_dir=pro_data_dir,
        batch_size=opt.batch_size)

    # get CNN model 
    my_model = get_model(
        out_dir=out_dir,
        run_model=opt.run_model, 
        activation=opt.activation, 
        input_shape=opt.input_shape,
        freeze_layer=opt.freeze_layer, 
        transfer=opt.transfer)

    ### train model
    if not opt.no_train:
        if opt.optimizer_function == 'adam':
            optimizer = Adam(learning_rate=opt.lr)
        train_model(
            root_dir=opt.root_dir,
            out_dir=out_dir,
            log_dir=log_dir,
            model_dir=model_dir,
            model=my_model,
            run_model=opt.run_model,
            train_gen=train_gen,
            val_gen=val_gen,
            x_val=x_val,
            y_val=y_val,
            batch_size=opt.batch_size,
            epoch=opt.epoch,
            optimizer=optimizer,
            loss_function=opt.loss_function,
            lr=opt.lr)

