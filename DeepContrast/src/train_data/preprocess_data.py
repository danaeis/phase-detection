import glob
import shutil
import os
import pandas as pd
#import nrrd  # No longer needed for NIfTI
import re
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from time import gmtime, strftime
from datetime import datetime
import timeit
from utils.respacing import respacing
from utils.nrrd_reg import nrrd_reg_rigid_ref
from utils.crop_image import crop_image
from utils.resize_3d import resize_3d
from utils.crop_image import crop_image
import SimpleITK as sitk


def preprocess_data(data_dir, pre_data_dir, new_spacing, data_exclude=None,
                    crop_shape=[192, 192, 10], interp_type='linear', region='abdomen'):
    """
    Preprocess data including: respacing, ROI crop, registration, final crop;

    Args:
        data_dir {path} -- path to CT data;
        pre_data_dir {path} -- path to save final preprocessed outputs;
    Keyword args:
        new_spacing {tuple} -- respacing size;
        data_exclude {str} -- exclude patient data due to data issue, default: None;
        crop_shape {np.array} -- numpy array size after cropping;
        interp_type {str} -- interpolation type for respacing, default: 'linear';
        region {str} -- body region: 'abdomen', 'chest', 'head_neck'
    Return:
        save nifti image data;
    """
    
    # Recursively find all .nii.gz files in nested folders
    fns = [fn for fn in sorted(glob.glob(os.path.join(data_dir, '*', '*.nii.gz')))]
    fns += [fn for fn in sorted(glob.glob(os.path.join(data_dir, '*', '*.nii')))]

    # Select template based on region (assume .nii.gz templates exist in data_dir)
    # if region == 'head_neck':
    #     reg_temp_img = fns[0]
    # elif region == 'abdomen':
    #     reg_temp_img = fns[0]
    # elif region == 'chest':
    #     reg_temp_img = fns[0]
    # else:
    #     raise ValueError('Unknown region')
    reg_temp_img = fns[0]
    # Use the filename (without extension) as the ID
    IDs = [os.path.splitext(os.path.basename(fn))[0].replace('.nii.gz', '') for fn in fns]

    # PMH dataframe
    df = pd.DataFrame({'ID': IDs, 'file': fns})
    for fn, ID in zip(df['file'], df['ID']):
        print(ID)
        # respacing
        img_respaced = respacing(
            nrrd_dir=fn,  # param name kept for compatibility, but it's now a NIfTI file
            interp_type=interp_type,
            new_spacing=new_spacing,
            patient_id=ID,
            return_type='nifti',
            save_dir=None)
        print("data dir", data_dir)
        # Construct seg_path
        seg_dir = os.path.join(os.path.dirname(data_dir), 'segmentation_masks')
        seg_path = os.path.join(seg_dir, f"{ID}_seg.nii.gz")
        
        # ROI crop using TotalSegmentator or pre-existing segmentation
        img_roi = crop_image(
            nrrd_file=img_respaced,
            patient_id=ID,
            crop_shape=crop_shape,  # crop_shape not used in 'roi' mode, but passed
            return_type='nifti',
            save_dir=None,  # don't save intermediate
            region=region,
            mode='roi',
            seg_path=seg_path)
        
        # registration on ROI cropped image
        img_reg = nrrd_reg_rigid_ref(
            img_nrrd=img_roi,
            fixed_img_dir=reg_temp_img,
            patient_id=ID,
            save_dir=None)
        
        # final crop to exact shape
        img_final = crop_image(
            nrrd_file=img_reg,
            patient_id=ID,
            crop_shape=crop_shape,
            return_type='nifti',
            save_dir=pre_data_dir,
            region=region,  # optional for final mode
            mode='final')


