import os
import itertools
import operator
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import nrrd
from PIL import Image


#--------------------------------------------------------------------------------------
# crop image
#-------------------------------------------------------------------------------------
def crop_image(nrrd_file, patient_id, crop_shape, return_type, save_dir):
    print(f"[Crop] Starting image cropping for patient {patient_id}...")
    
    ## load stik and arr
    print("[Crop] Loading image array...")
    img_arr = sitk.GetArrayFromImage(nrrd_file)
    ## Return top 25 rows of 3D volume, centered in x-y space / start at anterior (y=0)?
#    img_arr = np.transpose(img_arr, (2, 1, 0))
#    print("image_arr shape: ", img_arr.shape)
    c, y, x = img_arr.shape
#    x, y, c = image_arr.shape
#    print('c:', c)
#    print('y:', y)
#    print('x:', x)
    
    ## Get center of mass to center the crop in Y plane
    print("[Crop] Calculating center of mass...")
    try:
        print("[Crop] Creating binary mask...")
        mask_arr = np.copy(img_arr)
        # Handle NaN values first
        mask_arr = np.nan_to_num(mask_arr, nan=-1024)
        # Adjust threshold for abdomen tissue (-100 to +200 HU typical for soft tissue)
        mask_arr[mask_arr > -100] = 1
        mask_arr[mask_arr <= -100] = 0
        print(f"[Crop] Mask created with range: {np.amin(mask_arr):.2f} to {np.amax(mask_arr):.2f}")
        
        print("[Crop] Computing center of mass...")
        # Get initial center of mass for all slices
        centermass_3d = ndimage.measurements.center_of_mass(mask_arr)
        # Use the slice with maximum tissue content for x-y centering
        tissue_sums = np.sum(mask_arr, axis=(1,2))
        max_tissue_slice = np.argmax(tissue_sums)
        centermass = ndimage.measurements.center_of_mass(mask_arr[max_tissue_slice, :, :])
        print(f"[Crop] Center of mass computed at: ({centermass[0]:.1f}, {centermass[1]:.1f})")
        print(f"[Crop] Using slice {max_tissue_slice} of {c} for centering (maximum tissue content)")
    except Exception as e:
        print(f"[Crop] Error during mask creation or center calculation: {str(e)}")
        raise
    startx = int(centermass[0] - crop_shape[0]//2)
    starty = int(centermass[1] - crop_shape[1]//2)
    
    # Center the z-axis crop around the slice with maximum tissue content
    startz = max(0, int(max_tissue_slice - crop_shape[2]//2))
    
    # Ensure we don't exceed image boundaries
    startx = max(0, min(startx, x - crop_shape[0]))
    starty = max(0, min(starty, y - crop_shape[1]))
    startz = max(0, min(startz, c - crop_shape[2]))
    
    print(f"[Crop] Crop region - X: {startx}:{startx + crop_shape[0]}, Y: {starty}:{starty + crop_shape[1]}, Z: {startz}:{startz + crop_shape[2]}")
    
    print("[Crop] Cropping image to shape", crop_shape, "...")
    ## crop image using crop shape
    try:
        # Extract the region, handling boundary cases
        print(f"[Crop] Extracting region: z={startz}:{startz + crop_shape[2]}, y={starty}:{starty + crop_shape[1]}, x={startx}:{startx + crop_shape[0]}")
        
        # Calculate required padding for each dimension
        pad_x = max(0, (startx + crop_shape[0]) - x)
        pad_y = max(0, (starty + crop_shape[1]) - y)
        pad_z = max(0, (startz + crop_shape[2]) - c)
        
        if pad_x > 0 or pad_y > 0 or pad_z > 0:
            print(f"[Crop] Adding padding - X: {pad_x}, Y: {pad_y}, Z: {pad_z}")
            img_arr = np.pad(
                img_arr,
                ((0, pad_z), (0, pad_y), (0, pad_x)),
                'constant',
                constant_values=-1024  # Standard air HU value
            )
            print(f"[Crop] Padded image shape: {img_arr.shape}")
        
        # Extract the region
        img_crop_arr = img_arr[
            startz:startz + crop_shape[2],
            starty:starty + crop_shape[1],
            startx:startx + crop_shape[0]
        ]
        
        # Handle any NaN values in the cropped region
        img_crop_arr = np.nan_to_num(img_crop_arr, nan=-1024)
        
        print(f"[Crop] Cropped shape: {img_crop_arr.shape}")
        print(f"[Crop] Value range: {np.min(img_crop_arr):.1f} to {np.max(img_crop_arr):.1f} HU")
    except Exception as e:
        print(f"[Crop] Error during cropping/padding: {str(e)}")
        raise
    try:
        print("[Crop] Converting cropped array to NRRD format...")
        img_crop_nrrd = sitk.GetImageFromArray(img_crop_arr)
        img_crop_nrrd.SetSpacing(nrrd_file.GetSpacing())
        img_crop_nrrd.SetOrigin(nrrd_file.GetOrigin())
        print("[Crop] NRRD conversion complete")

        if save_dir != None:
            print(f"[Crop] Saving cropped image for patient {patient_id}...")
            fn = str(patient_id) + '.nrrd'
            writer = sitk.ImageFileWriter()
            writer.SetFileName(os.path.join(save_dir, fn))
            writer.SetUseCompression(True)
            writer.Execute(img_crop_nrrd)
            print(f"[Crop] Successfully saved cropped image for patient {patient_id}")

        print(f"[Crop] Returning result as {return_type}...")
        if return_type == 'nrrd':
            return img_crop_nrrd
        elif return_type == 'npy':
            return img_crop_arr
        else:
            raise ValueError(f"Unsupported return type: {return_type}")
            
    except Exception as e:
        print(f"[Crop] Error during final processing: {str(e)}")
        raise
 
#-----------------------------------------------------------------------
# run codes to test
#-----------------------------------------------------------------------
if __name__ == '__main__':

#    output_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/output'
    output_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/test'
    file_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_PMH'
    test_file = 'PMH_OPC-00050_CT-SIM_raw_raw_raw_xx.nrrd'
    return_type = 'sitk'
    crop_shape = [192, 192, 110]

    train_file = os.path.join(file_dir, test_file)

    img_crop = crop_image(
        nrrd_file=train_file,
        crop_shape=crop_shape,
        return_type=return_type,
        output_dir=output_dir
        )
    print('crop arr shape:', img_crop)
#    arr_crop = image_arr_crop[0, :, :]
#    img_dir = os.path.join(output_dir, 'arr_crop.jpg')
#    plt.imsave(img_dir, arr_crop, cmap='gray')
    print('successfully save image!!!')
    
