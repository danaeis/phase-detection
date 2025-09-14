import sys, os, glob
import SimpleITK as sitk
#import pydicom
import numpy as np

def registration_callback(method):
    print(f'Iteration: {method.GetOptimizerIteration()}, Metric: {method.GetMetricValue():.6f}')


def nrrd_reg_rigid_ref(img_nrrd, fixed_img_dir, patient_id, save_dir):
    print(f'\nStarting registration for patient {patient_id}...')
    print('Reading fixed image...')
    fixed_img = sitk.ReadImage(fixed_img_dir, sitk.sitkFloat32)
    print('Setting up moving image...')
    moving_img = img_nrrd
#    moving_img = sitk.ReadImage(img_nrrd, sitk.sitkUInt32)
    #moving_img = sitk.ReadImage(input_path, sitk.sitkFloat32)
    
    transform = sitk.CenteredTransformInitializer(
        fixed_img, 
        moving_img, 
        sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    
    # multi-resolution rigid registration using Mutual Information
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.05)  # Increased from 0.01 for better speed/accuracy balance
    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsGradientDescent(
        learningRate=2.0,  # Increased from 1.0 for faster convergence
        numberOfIterations=50,  # Reduced from 100
        convergenceMinimumValue=1e-5,  # Relaxed from 1e-6
        convergenceWindowSize=5  # Reduced from 10
        )
    
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2])  # Increased for faster initial alignment
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1])  # Adjusted accordingly
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(transform)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: registration_callback(registration_method))
    print('Executing registration (this may take a while)...')
    final_transform = registration_method.Execute(fixed_img, moving_img)
    print('Registration complete, resampling image...')                               
    moving_img_resampled = sitk.Resample(
        moving_img, 
        fixed_img, 
        final_transform, 
        sitk.sitkLinear, 
        0.0, 
        moving_img.GetPixelID()
        )
    img_reg = moving_img_resampled
    
    if save_dir != None:   
        nrrd_fn = str(patient_id) + '_reg.nrrd'
        print(f'Saving registered image to {os.path.join(save_dir, nrrd_fn)}...')
        sitk.WriteImage(img_reg, os.path.join(save_dir, nrrd_fn))
        print('Save complete.')

    return img_reg
    #return fixed_img, moving_img, final_transform
