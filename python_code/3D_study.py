#%%Libraries
import nibabel as nb
import numpy as np
import os
from random import shuffle
import skimage.transform as skTrans
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Input, Activation, Dropout, BatchNormalization, Conv3DTranspose, Concatenate
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt



#%% Data
def nifti2array(nifti_img, img_w, img_h, img_d):
    
    #Convert nifti into array
    img_array = nifti_img.get_fdata()
    
    if img_array.shape == [img_w, img_h, img_d]:
        rescaled_array = img_array
    else :
        #Rescaling the image
        rescaled_array = skTrans.resize(img_array, (img_w, img_h, img_d), order=1, preserve_range=True)
    return rescaled_array


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


def get_data(img_w, img_h, img_d, adc = True, dwi = True, flair = False):
    
    #Loading filenames
    adc_list, dwi_list, flair_list, mask_list = list_files(adc, dwi, flair)
    shuffle(mask_list)
    
    #Initialisation of the arrays
    if adc :
        adc_array = np.zeros((len(adc_list), img_w, img_h, img_d), dtype = np.float32)
    if dwi :
        dwi_array = np.zeros((len(dwi_list), img_w, img_h, img_d), dtype = np.float32)
    if flair :
        flair_array = np.zeros((len(flair_list), img_w, img_h, img_d), dtype = np.float32)
    mask_array = np.zeros((len(mask_list), img_w, img_h, img_d), dtype = np.float32)
    
    
    for ind, mask in enumerate(mask_list):
        
        # Saving the ID of the current patient
        ID = mask.split('derivatives')[1].split('_msk.nii.gz')[0]
        
        # Creating mask array
        mask_nii = nb.load(mask)
        mask_small_array = nifti2array(mask_nii, img_w, img_h, img_d) # Transforming the image into a rescaled array
        mask_small_array = (mask_small_array > 0).astype(np.int_) #Binarizing the mask
        mask_array[ind] = mask_small_array
        
        
        # Creating adc array
        if adc:
            name_subj = [i for i in adc_list if ID in i]
            if len(name_subj) == 1:
                adc_nii = nb.load(name_subj[0])
                adc_small_array = nifti2array(adc_nii, img_w, img_h, img_d) # Transforming the image into a rescaled array
                adc_array[ind] = (adc_small_array-np.min(adc_small_array))/(np.max(adc_small_array)-np.min(adc_small_array)) # Normalizing the array
            else :
                print('Problem with adc for ID : '+ID)
        
        # Creating dwi array
        if dwi:
            name_subj = [i for i in dwi_list if ID in i]
            if len(name_subj) == 1:
                dwi_nii = nb.load(name_subj[0])
                dwi_small_array = nifti2array(dwi_nii, img_w, img_h, img_d) # Transforming the image into a rescaled array
                dwi_array[ind] = (dwi_small_array-np.min(dwi_small_array))/(np.max(dwi_small_array)-np.min(dwi_small_array)) # Normalizing the array
            else :
                print('Problem with dwi for ID : '+ID)
                
        # Creating flair array
        if flair:
            name_subj = [i for i in flair_list if ID in i]
            if len(name_subj) == 1:
                flair_nii = nb.load(name_subj[0])
                flair_small_array = nifti2array(flair_nii, img_w, img_h, img_d) # Transforming the image into a rescaled array
                flair_array[ind] = (flair_small_array-np.min(flair_small_array))/(np.max(flair_small_array)-np.min(flair_small_array)) # Normalizing the array 
            else :
                print('Problem with flair for ID : '+ID)
        
        # Progression
        print(str(ind+1) + '/' + str(len(mask_list)))
    
    # Build the output 
    ans = []
    if adc:
        adc_array = np.expand_dims(adc_array, axis = 4)
        ans.append(adc_array)
    if dwi:
        dwi_array = np.expand_dims(dwi_array, axis = 4)
        ans.append(dwi_array)
    if flair:
        flair_array = np.expand_dims(flair_array, axis = 4)
        ans.append(flair_array)
    mask_array = np.expand_dims(mask_array, axis = 4)
    ans.append(mask_array)
    return ans

def get_all_data(img_w, img_h, img_d):
    [adc, dwi, flair, y_data] = get_data(img_w, img_h, img_d, True, True, True)
    x_data = np.zeros((adc.shape[0], img_w, img_h, img_d, 3),  dtype = np.float32)
    x_data[:, :, :, :, 0] = adc[:, :, :, :, 0]
    x_data[:, :, :, :, 1] = dwi[:, :, :, :, 0]
    x_data[:, :, :, :, 2] = flair[:, :, :, :, 0]
    return x_data, y_data
    

def get_train_test(list_array, split_ratio = 0.8):
    train_list = []
    test_list = []
    for img_array in list_array:
        index = int(split_ratio * img_array.shape[0])
        train = img_array[:index, :, :, :, :]
        test = img_array[index:, :, :, :, :]
        train_list.append(train)
        test_list.append(test)
    return train_list, test_list

#%% Unet architecture

# Convolutianal block
def conv_block(input_layer, n_base, batchnorm = False):
    output_layer = Conv3D(filters= n_base, kernel_size=(3,3,3), strides=(1,1,1), padding='same')(input_layer)
    if batchnorm :
        output_layer = BatchNormalization()(output_layer)
    output_layer = Activation('relu')(output_layer)
    output_layer = Conv3D(filters=n_base, kernel_size=(3,3, 3), strides=(1,1,1), padding='same')(output_layer)
    if batchnorm :
        output_layer = BatchNormalization()(output_layer)
    output_layer = Activation('relu')(output_layer)
    return output_layer

# Encoder block
def encoder_block(input_layer,n_base , batchnorm = False, dropout = False):
    out = conv_block(input_layer, n_base, batchnorm = batchnorm)
    out2 = MaxPooling3D(pool_size=(2,2,2))(out)
    if dropout:
        out2 = Dropout(0.2)(out2)
    return out, out2

# Decoder block
def decoder_block(input_layer, layer2conc, n_base, batchnorm = False, dropout = False):
    output_layer = Conv3DTranspose(filters = n_base,  kernel_size=(3,3,3), strides=(2,2,2), padding="same")(input_layer)
    output_layer = Concatenate()([output_layer, layer2conc])
    if dropout:
        output_layer = Dropout(0.2)(output_layer)
    output_layer = conv_block(output_layer, n_base, batchnorm = batchnorm)
    return output_layer

# U-net model
def get_unet(img_ch, img_width, img_height, img_depth, n_base, dropout = False, batchnormal = False, binary = True, class_num = 2):
    input_layer = Input(shape=(img_width, img_height, img_depth, img_ch))
    
    #Encoder
    e1, em1 = encoder_block(input_layer, n_base, batchnorm = batchnormal, dropout = dropout)
    e2, em2 = encoder_block(em1, n_base*2, batchnorm = batchnormal, dropout = dropout)
    e3, em3 = encoder_block(em2, n_base*4, batchnorm = batchnormal, dropout = dropout)
    e4, em4 = encoder_block(em3, n_base*8, batchnorm = batchnormal, dropout = dropout)

    #Bottleneck 
    bottleneck = conv_block(em4, n_base*16, batchnorm = batchnormal)
    print(bottleneck.shape)
    #Decoder
    d_block1 = decoder_block(bottleneck, e4, n_base*8, batchnorm = batchnormal, dropout = dropout)
    d_block2 = decoder_block(d_block1, e3, n_base*4, batchnorm = batchnormal, dropout = dropout)
    d_block3 = decoder_block(d_block2, e2, n_base*2, batchnorm = batchnormal, dropout = dropout)
    d_block4 = decoder_block(d_block3, e1, n_base, batchnorm = batchnormal, dropout = dropout)
    
    #Output
    if binary:
        out = Conv3D(filters=1, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation = 'sigmoid')(d_block4)
    else:
        out = Conv3D(filters=class_num, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation = 'sigmoid')(d_block4)
        
    clf = Model(inputs=input_layer, outputs=out)
    clf.summary()
    
    return clf 

#%% Dice coeff
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 0.0001) / (K.sum(y_true_f) + K.sum(y_pred_f) + 0.0001)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

#%% Run 3D in each modality

# Path where the different files will be saved
path = 'C:/Users/cdarc/OneDrive/Documents/KTH/Deep learning/Projet/Stroke project/'

# Parameters of the model
n_base = 8
LR = 0.0001
n_epochs = 60
n_steps = 5
batchsize = 4
img_w, img_h, img_d, img_ch = 112, 112, 64, 1

clear_session()

# ADC
## Load data
[adc_array, mask_array] = get_data(img_w, img_h, img_d, True, False, False)
[x_train, y_train], [x_test, y_test] = get_train_test([adc_array, mask_array])
del adc_array, mask_array

## Build architecture
clf = get_unet(img_ch, img_w, img_h, img_d, n_base, batchnormal =True, dropout = False)
clf.compile(loss=[dice_coef_loss], optimizer = Adam(lr = LR), metrics=[dice_coef, Precision(), Recall()])
clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, steps_per_epoch = n_steps, batch_size = batchsize, validation_data=(x_test, y_test))
clf.save(path + 'models/3D_new_dice_no_swap')

## Free memory
del x_train
del y_train
del x_test
del y_test

## Plots
###Accuracy
plt.figure(figsize=(4, 4))
plt.title("Accuracy")
plt.plot(clf_hist.history["dice_coef"], label="dice_coef")
plt.plot(clf_hist.history["val_dice_coef"], label="val_dice_coef")
xmax = np.argmax(clf_hist.history["val_dice_coef"])
ymax = np.max(clf_hist.history["val_dice_coef"])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Dice coefficient")
plt.legend();
plt.savefig(path + 'Results/Dice_coef-adc.png', dpi = 200)

###Loss
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(clf_hist.history["loss"], label="loss")
plt.plot(clf_hist.history["val_loss"], label="val_loss")
xmin = np.argmin(clf_hist.history["val_loss"])
ymin = np.min(clf_hist.history["val_loss"])
plt.plot( xmin, ymin, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmin) + ', '+ str(round(ymin, 2)) + ')', xy = (xmin, ymin - 0.01),
             horizontalalignment = "center", verticalalignment = "top", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig(path + 'Results/Loss-adc.png', dpi = 200)

### Precision
plt.figure(figsize=(4, 4))
plt.title("Precision")
plt.plot(clf_hist.history['precision'], label="Precision")
plt.plot(clf_hist.history["val_precision"], label="val_precision")
xmax = np.argmax(clf_hist.history["val_precision"])
ymax = np.max(clf_hist.history["val_precision"])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Precision")
plt.legend();
plt.savefig(path + 'Results/Precision-adc.png', dpi = 200)
    
### Recall
plt.figure(figsize=(4, 4))
plt.title("Recall")
plt.plot(clf_hist.history['recall'], label="Recall")
plt.plot(clf_hist.history['val_recall'], label="val_Recall")
xmax = np.argmax(clf_hist.history['val_recall'])
ymax = np.max(clf_hist.history['val_recall'])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Recall")
plt.legend();
plt.savefig(path + 'Results/Recall-adc.png', dpi = 200)
del clf, clf_hist
clear_session()




# DWI
## Load data
[dwi_array, mask_array] = get_data(img_w, img_h, img_d, False, True, False)
[x_train, y_train], [x_test, y_test] = get_train_test([dwi_array, mask_array])
del dwi_array, mask_array

## Build architecture
clf = get_unet(img_ch, img_w, img_h, img_d, n_base, batchnormal =True, dropout = False)
clf.compile(loss=[dice_coef_loss], optimizer = Adam(lr = LR), metrics=[dice_coef, Precision(), Recall()])
clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, steps_per_epoch = n_steps, batch_size = batchsize, validation_data=(x_test, y_test))
clf.save(path + 'models/dwi_model')

## Free memory
del x_train
del y_train
del x_test
del y_test

## Plots
###Accuracy
plt.figure(figsize=(4, 4))
plt.title("Accuracy")
plt.plot(clf_hist.history["dice_coef"], label="dice_coef")
plt.plot(clf_hist.history["val_dice_coef"], label="val_dice_coef")
xmax = np.argmax(clf_hist.history["val_dice_coef"])
ymax = np.max(clf_hist.history["val_dice_coef"])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Dice coefficient")
plt.legend();
plt.savefig(path + 'Results/Dice_coef-dwi.png', dpi = 200)

###Loss
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(clf_hist.history["loss"], label="loss")
plt.plot(clf_hist.history["val_loss"], label="val_loss")
xmin = np.argmin(clf_hist.history["val_loss"])
ymin = np.min(clf_hist.history["val_loss"])
plt.plot( xmin, ymin, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmin) + ', '+ str(round(ymin, 2)) + ')', xy = (xmin, ymin - 0.01),
             horizontalalignment = "center", verticalalignment = "top", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig(path + 'Results/Loss-dwi.png', dpi = 200)

### Precision
plt.figure(figsize=(4, 4))
plt.title("Precision")
plt.plot(clf_hist.history['precision'], label="Precision")
plt.plot(clf_hist.history["val_precision"], label="val_precision")
xmax = np.argmax(clf_hist.history["val_precision"])
ymax = np.max(clf_hist.history["val_precision"])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Precision")
plt.legend();
plt.savefig(path + 'Results/Precision-dwi.png', dpi = 200)
    
### Recall
plt.figure(figsize=(4, 4))
plt.title("Recall")
plt.plot(clf_hist.history['recall'], label="Recall")
plt.plot(clf_hist.history['val_recall'], label="val_Recall")
xmax = np.argmax(clf_hist.history['val_recall'])
ymax = np.max(clf_hist.history['val_recall'])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Recall")
plt.legend();
plt.savefig(path + 'Results/Recall-dwi.png', dpi = 200)
del clf, clf_hist
clear_session()





# Flair
## Load data
[flair_array, mask_array] = get_data(img_w, img_h, img_d, False, False, True)
[x_train, y_train], [x_test, y_test] = get_train_test([flair_array, mask_array])
del flair_array, mask_array

## Build architecture
clf = get_unet(img_ch, img_w, img_h, img_d, n_base, batchnormal =True, dropout = False)
clf.compile(loss=[dice_coef_loss], optimizer = Adam(lr = LR), metrics=[dice_coef, Precision(), Recall()])
clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, steps_per_epoch = n_steps, batch_size = batchsize, validation_data=(x_test, y_test))
clf.save(path + 'models/flair_model')

## Free memory
del x_train
del y_train
del x_test
del y_test

## Plots
###Accuracy
plt.figure(figsize=(4, 4))
plt.title("Accuracy")
plt.plot(clf_hist.history["dice_coef"], label="dice_coef")
plt.plot(clf_hist.history["val_dice_coef"], label="val_dice_coef")
xmax = np.argmax(clf_hist.history["val_dice_coef"])
ymax = np.max(clf_hist.history["val_dice_coef"])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Dice coefficient")
plt.legend();
plt.savefig(path + 'Results/Dice_coef-flair.png', dpi = 200)

###Loss
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(clf_hist.history["loss"], label="loss")
plt.plot(clf_hist.history["val_loss"], label="val_loss")
xmin = np.argmin(clf_hist.history["val_loss"])
ymin = np.min(clf_hist.history["val_loss"])
plt.plot( xmin, ymin, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmin) + ', '+ str(round(ymin, 2)) + ')', xy = (xmin, ymin - 0.01),
             horizontalalignment = "center", verticalalignment = "top", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig(path + 'Results/Loss-flair.png', dpi = 200)

### Precision
plt.figure(figsize=(4, 4))
plt.title("Precision")
plt.plot(clf_hist.history['precision'], label="Precision")
plt.plot(clf_hist.history["val_precision"], label="val_precision")
xmax = np.argmax(clf_hist.history["val_precision"])
ymax = np.max(clf_hist.history["val_precision"])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Precision")
plt.legend();
plt.savefig(path + 'Results/Precision-flair.png', dpi = 200)
    
### Recall
plt.figure(figsize=(4, 4))
plt.title("Recall")
plt.plot(clf_hist.history['recall'], label="Recall")
plt.plot(clf_hist.history['val_recall'], label="val_Recall")
xmax = np.argmax(clf_hist.history['val_recall'])
ymax = np.max(clf_hist.history['val_recall'])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Recall")
plt.legend();
plt.savefig(path + 'Results/Recall-flair.png', dpi = 200)
del clf, clf_hist
clear_session()



#%% Run 3D in all modality

#Load data
x_data, y_data = get_all_data(112, 112, 64)
[x_train, y_train], [x_test, y_test] = get_train_test([x_data, y_data])
del x_data, y_data

#%%

# Path where the different files will be saved
path = 'C:/Users/cdarc/OneDrive/Documents/KTH/Deep learning/Projet/Stroke project/'

# Parameters of the model
n_base = 8
LR = 0.0001
n_epochs = 80
n_steps = 5
batchsize = 4
img_w, img_h, img_d, img_ch = 112, 112, 64, 3

clear_session()

## Build architecture
clf = get_unet(img_ch, img_w, img_h, img_d, n_base, batchnormal =True, dropout = False)
clf.compile(loss=[dice_coef_loss], optimizer = Adam(lr = LR), metrics=[dice_coef, Precision(), Recall()])
clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, steps_per_epoch = n_steps, batch_size = batchsize, validation_data=(x_test, y_test))
clf.save(path + 'models/3D_all')

## Free memory
del x_train
del y_train
del x_test
del y_test

## Plots
###Accuracy
plt.figure(figsize=(4, 4))
plt.title("Accuracy")
plt.plot(clf_hist.history["dice_coef"], label="dice_coef")
plt.plot(clf_hist.history["val_dice_coef"], label="val_dice_coef")
xmax = np.argmax(clf_hist.history["val_dice_coef"])
ymax = np.max(clf_hist.history["val_dice_coef"])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Dice coefficient")
plt.legend();
plt.savefig(path + 'Results/Dice_coef-all3D.png', dpi = 200)

###Loss
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(clf_hist.history["loss"], label="loss")
plt.plot(clf_hist.history["val_loss"], label="val_loss")
xmin = np.argmin(clf_hist.history["val_loss"])
ymin = np.min(clf_hist.history["val_loss"])
plt.plot( xmin, ymin, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmin) + ', '+ str(round(ymin, 2)) + ')', xy = (xmin, ymin - 0.01),
             horizontalalignment = "center", verticalalignment = "top", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.savefig(path + 'Results/Loss-all3D.png', dpi = 200)

### Precision
plt.figure(figsize=(4, 4))
plt.title("Precision")
plt.plot(clf_hist.history['precision'], label="Precision")
plt.plot(clf_hist.history["val_precision"], label="val_precision")
xmax = np.argmax(clf_hist.history["val_precision"])
ymax = np.max(clf_hist.history["val_precision"])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Precision")
plt.legend();
plt.savefig(path + 'Results/Precision-all3D.png', dpi = 200)
    
### Recall
plt.figure(figsize=(4, 4))
plt.title("Recall")
plt.plot(clf_hist.history['recall'], label="Recall")
plt.plot(clf_hist.history['val_recall'], label="val_Recall")
xmax = np.argmax(clf_hist.history['val_recall'])
ymax = np.max(clf_hist.history['val_recall'])
plt.plot( xmax, ymax, marker="x", color="r", label="best model")
plt.annotate('(' + str(xmax) + ', '+ str(round(ymax,2)) + ')', xy = (xmax, ymax + 0.01),
             horizontalalignment = "center", verticalalignment = "bottom", color = "red")
plt.xlabel("Epochs")
plt.ylabel("Recall")
plt.legend();
plt.savefig(path + 'Results/Recall-all3D.png', dpi = 200)
del clf, clf_hist
clear_session()

