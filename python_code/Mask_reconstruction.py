#%% Libraries
import nibabel as nb
import numpy as np
import os
from tensorflow.keras.models import load_model
from skimage.transform import resize

#%% Data

#Function to get the list of all the files to load with their respective paths
def list_files(adc, dwi, flair):
    #List of file
    path_raw = 'C:/Users/cdarc/OneDrive/Documents/KTH/Deep learning/Projet/Stroke project/rawdata/'
    path_mask = 'C:/Users/cdarc/OneDrive/Documents/KTH/Deep learning/Projet/Stroke project/derivatives/'
    adc_list = []
    dwi_list = []
    flair_list = []
    mask_list = []
    
    for folder in os.listdir(path_raw):
        current_path = path_raw + folder + '/'
        if len(os.listdir(current_path))>1 or len(os.listdir(current_path))<1:
            print('Problem with folder :' + folder)
        else:
            current_path = current_path + os.listdir(current_path)[0] + '/'
            for file in os.listdir(current_path):
                if '.nii.gz' in file:
                    if 'adc' in file:
                        if adc:
                            adc_list.append(current_path+file)
                    elif 'dwi' in file:
                        if dwi:
                            dwi_list.append(current_path+file)
                    elif 'flair' in file:
                        if flair:
                            flair_list.append(current_path+file)
                    else:
                        print('Problem with subject : ' + folder)
                else:
                    pass
    
    for folder in os.listdir(path_mask):
        current_path = path_mask + folder + '/'
        if len(os.listdir(current_path))>1 or len(os.listdir(current_path))<1:
            print('Problem with folder :' + folder)
        else:
            current_path = current_path + os.listdir(current_path)[0] + '/'
            for file in os.listdir(current_path):
                if '.nii.gz' in file:
                    mask_list.append(current_path+file)
                else:
                    pass
    
    return adc_list, dwi_list, flair_list, mask_list

#Function to binarize an array according an input threshold
def binarize_array(input_array, threshold, dim2 = False, dim3 = False):
    if dim2:
        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):
                if input_array[i, j] >= threshold:
                    input_array[i, j] = 1
                else:
                    input_array[i, j] = 0
    elif dim3:
        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):
                for k in range(input_array.shape[2]):
                    if input_array[i, j, k] >= threshold:
                        input_array[i, j, k] = 1
                    else:
                        input_array[i, j, k] = 0
    else:
        print('Precise dim array')
    return input_array
                    
    
#%% Dice coeff
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 0.0001) / (K.sum(y_true_f) + K.sum(y_pred_f) + 0.0001)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

#Function to compute the dice coefficient between 2 arrays, it was given in the challenge
def compute_dice(im1, im2, empty_value=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size as im1. If not boolean, it will be converted.
    empty_value : scalar, float.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        If both images are empty (sum equal to zero) = empty_value
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    This function has been adapted from the Verse Challenge repository:
    https://github.com/anjany/verse/blob/main/utils/eval_utilities.py
    """

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_value
    
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2.0 * intersection.sum() / im_sum


#%% Mask reconstruction for 1 subject and compute dice_coef at each layer
#This part of the code evaluate a trained model on each layer of one subject

#Parameter
n_base = 8
LR = 0.0001
n_epochs = 150
batchsize = 8
img_w, img_h, img_ch = 112, 112, 3
subject = 3

#Load path of data 
adc_list, dwi_list, flair_list, mask_list = list_files(True, True, True)

#Load the model to be used
path_model = 'C:/Users/cdarc/OneDrive/Documents/KTH/Deep learning/Projet/Stroke project/models/2D/'
model_name = '5_slices_dropout'
u_net_model = load_model(path_model + model_name, custom_objects = {"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})


#Nifti to array
adc_nii = nb.load(adc_list[subject])
adc = adc_nii.get_fdata()
adc = (adc-np.min(adc))/(np.max(adc)-np.min(adc))
dwi_nii = nb.load(dwi_list[subject])
dwi = dwi_nii.get_fdata()
dwi = (dwi-np.min(dwi))/(np.max(dwi)-np.min(dwi))
flair_nii = nb.load(flair_list[subject])
flair = flair_nii.get_fdata()
flair = (flair-np.min(flair))/(np.max(flair)-np.min(flair))
mask_nii = nb.load(mask_list[subject])
mask = mask_nii.get_fdata()
del adc_nii, dwi_nii, flair_nii, mask_nii

#Initialization of the prediction structure
n = mask.shape[2]
y_pred = np.zeros((img_h, img_w, n, 1), dtype = np.float32) 

#List of dice coeff per layer
dice_coeff= []

for layer in range(mask.shape[2]):
    #Selecting the corresponding layer in the other modalities
    adc_layer = adc[:, :, int(layer*adc.shape[2]/n)]
    adc_layer = resize(adc_layer, (img_h, img_w), anti_aliasing = True).astype('float32')
    dwi_layer = dwi[:, :, int(layer*dwi.shape[2]/n)]
    dwi_layer = resize(dwi_layer, (img_h, img_w), anti_aliasing = True).astype('float32')
    flair_layer = flair[:, :, int(layer*flair.shape[2]/n)]
    flair_layer = resize(flair_layer, (img_h, img_w), anti_aliasing = True).astype('float32')
    mask_layer = resize(mask[:, :, layer], (img_h, img_w), anti_aliasing = True).astype('float32')
   
    #Construction of the input of the model
    x = np.zeros((1, img_h, img_w, 3), dtype = np.float32)
    x[0, :, :, 0] = adc_layer
    x[0, :, :, 1] = dwi_layer
    x[0, :, :, 2] = flair_layer

    
    #Prediction
    y_pred[:, :, layer, :] = u_net_model.predict(x)
    
    dice_coeff.append(compute_dice(binarize_array(y_pred[:, :, layer, 0], 0.5, dim2 = True), mask_layer))
    
    #Progression
    print(str(layer+1) + '/' + str(n))

mask_pred = binarize_array(y_pred[:, :, :, 0], 0.5, dim3 = True)
overall_dice_coef = compute_dice(mask_pred, mask)

#In the end the list dice_coeff contains all the dice_coeff comparing predicted layers to true layers
#And overall_dice_coeff contains the overall dice coeff of the reconstructed mask layer by layer with the true mask

#%% Mask reconstruction of all subject
#This part evaluate a trained model for all subject

#Parameter
n_base = 8
LR = 0.0001
n_epochs = 150
batchsize = 8
img_w, img_h, img_ch = 112, 112, 3

#Load path of data 
adc_list, dwi_list, flair_list, mask_list = list_files(True, True, True)

#Load model
path_model = 'C:/Users/cdarc/OneDrive/Documents/KTH/Deep learning/Projet/Stroke project/models/2D/'
model_name = '5_slices_dropout'
u_net_model = load_model(path_model + model_name, custom_objects = {"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})

#List of dice coeff per subject
dice_coeff= []

for subject in range(len(mask_list)):
    
    #Nifti to array
    adc_nii = nb.load(adc_list[subject])
    adc = adc_nii.get_fdata()
    adc = (adc-np.min(adc))/(np.max(adc)-np.min(adc))
    dwi_nii = nb.load(dwi_list[subject])
    dwi = dwi_nii.get_fdata()
    dwi = (dwi-np.min(dwi))/(np.max(dwi)-np.min(dwi))
    flair_nii = nb.load(flair_list[subject])
    flair = flair_nii.get_fdata()
    flair = (flair-np.min(flair))/(np.max(flair)-np.min(flair))
    mask_nii = nb.load(mask_list[subject])
    mask = mask_nii.get_fdata()
    del adc_nii, dwi_nii, flair_nii, mask_nii
    
    #Initialization of the prediction structure
    n = mask.shape[2]
    x = np.zeros((n, img_h, img_w, 3), dtype = np.float32)
    y_pred = np.zeros((n, img_h, img_w, 1), dtype = np.float32) 
    mask_resize = np.zeros((img_h, img_w, n), dtype = np.float32)
    
    for layer in range(n):
        #Selecting the corresponding layer in the other modalities
        adc_layer = adc[:, :, int(layer*adc.shape[2]/n)]
        adc_layer = resize(adc_layer, (img_h, img_w), anti_aliasing = True).astype('float32')
        dwi_layer = dwi[:, :, int(layer*dwi.shape[2]/n)]
        dwi_layer = resize(dwi_layer, (img_h, img_w), anti_aliasing = True).astype('float32')
        flair_layer = flair[:, :, int(layer*flair.shape[2]/n)]
        flair_layer = resize(flair_layer, (img_h, img_w), anti_aliasing = True).astype('float32')
       
        #Construction of the input of the model
        x[layer, :, :, 0] = adc_layer
        x[layer, :, :, 1] = dwi_layer
        x[layer, :, :, 2] = flair_layer
    
        #Resize mask
        mask_layer = resize(mask[:, :, layer], (img_h, img_w), anti_aliasing = True).astype('float32')
        mask_resize[:, :, layer] = mask_layer 
        
        # dice_coeff.append(compute_dice(binarize_array(y_pred[:, :, layer, 0], 0.5, dim2 = True), mask_layer))
    
    y_pred[:, :, :, :] = u_net_model.predict(x)
        
    
    mask_pred = binarize_array(y_pred[:, :, :, 0], 0.5, dim3 = True)
    mask_pred = np.swapaxes(np.swapaxes(mask_pred, 0, 1), 1, 2)
    dice_coeff.append(compute_dice(mask_pred, mask_resize))
    #Progression
    print(str(subject+1) + '/' + str(len(mask_list)))


#In the end dice_coeff contains the dice coeff between the predicted mask and the true mask for all subjects






