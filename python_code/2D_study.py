#%%Libraries
import nibabel as nb
import numpy as np
import os
from random import shuffle
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Activation, Dropout, BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from skimage.transform import resize


#%% Data

#Function used to split the data according to the split ratio
def train_test_data(x_data, y_data, split_ratio = 0.8):
    split_index = int(split_ratio*x_data.shape[0])
    x_train = x_data[:split_index, :, :, :]
    x_test = x_data[split_index:, :, :, :]
    y_train = y_data[:split_index, :, :, :]
    y_test = y_data[split_index:, :, :, :]
    return x_train, x_test, y_train, y_test

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

#Function to return a chosen number of layer affected by the strokes in each subject
def select_layer(mask_array, number):
    number_of_layer = mask_array.shape[2]
    current_max = 0
    most_affected_layer = 0
    layers = []
    selected_layer = []
    for layer in range(mask_array.shape[2]):
        c0 = 0
        c1 = 0
        for i in range(mask_array.shape[0]):
            for j in range(mask_array.shape[1]):
                if mask_array[i, j, layer] == 1:
                    c1 = c1 +1
                    layers.append(layer)
                elif mask_array[i, j, layer] == 0:
                    c0 = c0 +1
                else :
                    print('Problem with value in mask', mask_array[i,j,layer])
        if current_max < c1:
            current_max = c1
            most_affected_layer = layer
    selected_layer.append(most_affected_layer)
    if len(layers)<number-1:
        for layer in layers:
            selected_layer.append(layer)
    else:
        shuffle(layers)
        for ind in range(number-1):
            selected_layer.append(layers[ind])
    return selected_layer, number_of_layer
    
#Function used to load the data
def get_data(img_w, img_h, number_of_layer_per_subj):
    
    #Loading filenames
    adc_list, dwi_list, flair_list, mask_list = list_files(True, True, True)
    shuffle(mask_list)
    
    #Initialisation of the arrays
    x_data = np.zeros((len(mask_list)*number_of_layer_per_subj, img_h, img_w, 3), dtype = np.float32)
    y_data = np.zeros((len(mask_list)*number_of_layer_per_subj, img_h, img_w, 1), dtype = np.float32)
    c = 0
    #Filling the arrays
    for ind, mask in enumerate(mask_list):
        
        # Saving the ID of the current patient
        ID = mask.split('derivatives')[1].split('_msk.nii.gz')[0]
        
        #Creation of the mask array
        mask_nii = nb.load(mask)
        mask_array = mask_nii.get_fdata()
        layers, n = select_layer(mask_array, number_of_layer_per_subj) #select affected layers
        
        for layer in layers:
            #Loading this layer in each modality
            layer_mask = mask_array[:, :, layer]
            layer_mask = resize(layer_mask, (img_h, img_w), anti_aliasing = True).astype('float32')
            
            #adc
            name_subj = [i for i in adc_list if ID in i]
            if len(name_subj) == 1:
                adc_nii = nb.load(name_subj[0])
                adc_array = adc_nii.get_fdata()
                adc_array = (adc_array - np.min(adc_array))/(np.max(adc_array)-np.min(adc_array))
                layer_adc = adc_array[:, :, int(layer*adc_array.shape[2]/n)]
                layer_adc = resize(layer_adc, (img_h, img_w), anti_aliasing = True).astype('float32')
            else :
                print('Problem with adc for ID : '+ID)
            
            #dwi
            name_subj = [i for i in dwi_list if ID in i]
            if len(name_subj) == 1:
                dwi_nii = nb.load(name_subj[0])
                dwi_array = dwi_nii.get_fdata()
                dwi_array = (dwi_array - np.min(dwi_array))/(np.max(dwi_array)-np.min(dwi_array))
                layer_dwi = dwi_array[:, :, int(layer*dwi_array.shape[2]/n)]
                layer_dwi = resize(layer_dwi, (img_h, img_w), anti_aliasing = True).astype('float32')
            else :
                print('Problem with dwi for ID : '+ID)
                
            #flair
            name_subj = [i for i in flair_list if ID in i]
            if len(name_subj) == 1:
                flair_nii = nb.load(name_subj[0])
                flair_array = flair_nii.get_fdata()
                flair_array = (flair_array - np.min(flair_array))/(np.max(flair_array)-np.min(flair_array))
                layer_flair = flair_array[:, :, int(layer*flair_array.shape[2]/n)]
                layer_flair = resize(layer_flair, (img_h, img_w), anti_aliasing = True).astype('float32')
            else :
                print('Problem with flair for ID : '+ID)
            x_data[c, :, :, 0] = layer_adc
            x_data[c, :, :, 1] = layer_dwi
            x_data[c, :, :, 2] = layer_flair
            y_data[c, :, :, 0] = layer_mask
            c = c+1
        print(str(ind+1)+'/'+str(len(mask_list)))
    
    return x_data, y_data

#%% U-net

# Convolutianal block
def conv_block(input_layer, n_base, batchnorm = False):
    output_layer = Conv2D(filters= n_base, kernel_size=(3,3), strides=(1,1), padding='same')(input_layer)
    if batchnorm :
        output_layer = BatchNormalization()(output_layer)
    output_layer = Activation('relu')(output_layer)
    output_layer = Conv2D(filters=n_base, kernel_size=(3,3), strides=(1,1), padding='same')(output_layer)
    if batchnorm :
        output_layer = BatchNormalization()(output_layer)
    output_layer = Activation('relu')(output_layer)
    return output_layer

# Encoder block
def encoder_block(input_layer,n_base , batchnorm = False, dropout = False):
    out = conv_block(input_layer, n_base, batchnorm = batchnorm)
    out2 = MaxPooling2D(pool_size=(2,2))(out)
    if dropout:
        out2 = Dropout(0.2)(out2)
    return out, out2

# Decoder block
def decoder_block(input_layer, layer2conc, n_base, batchnorm = False, dropout = False):
    output_layer = Conv2DTranspose(filters = n_base,  kernel_size=(3,3), strides=(2, 2), padding="same")(input_layer)
    output_layer = Concatenate()([output_layer, layer2conc])
    if dropout:
        output_layer = Dropout(0.2)(output_layer)
    output_layer = conv_block(output_layer, n_base, batchnorm = batchnorm)
    return output_layer

# Unet model
def get_unet(img_ch, img_width, img_height, n_base, dropout = False, batchnormal = False, binary = True, class_num = 2):
    input_layer = Input(shape=(img_width, img_height, img_ch))
    
    #Encoder
    e1, em1 = encoder_block(input_layer, n_base, batchnorm = batchnormal, dropout = dropout)
    e2, em2 = encoder_block(em1, n_base*2, batchnorm = batchnormal, dropout = dropout)
    e3, em3 = encoder_block(em2, n_base*4, batchnorm = batchnormal, dropout = dropout)
    e4, em4 = encoder_block(em3, n_base*8, batchnorm = batchnormal, dropout = dropout)

    #Bottleneck 
    bottleneck = conv_block(em4, n_base*16, batchnorm = batchnormal)

    #Decoder
    d_block1 = decoder_block(bottleneck, e4, n_base*8, batchnorm = batchnormal, dropout = dropout)
    d_block2 = decoder_block(d_block1, e3, n_base*4, batchnorm = batchnormal, dropout = dropout)
    d_block3 = decoder_block(d_block2, e2, n_base*2, batchnorm = batchnormal, dropout = dropout)
    d_block4 = decoder_block(d_block3, e1, n_base, batchnorm = batchnormal, dropout = dropout)
    
    #Output
    if binary:
        out = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same', activation = 'sigmoid')(d_block4)
    else:
        out = Conv2D(filters=class_num, kernel_size=(3,3), strides=(1,1), padding='same', activation = 'sigmoid')(d_block4)
        
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


#%%  Training and testing 1 slice

#Load data
x_data, y_data = get_data(112, 112, 1) #get 1 slice per subject
x_train, x_test, y_train, y_test = train_test_data(x_data, y_data) #split the data
del x_data, y_data #free memory

# Path where the different files will be saved
path = 'C:/Users/cdarc/OneDrive/Documents/KTH/Deep learning/Projet/Stroke project/'

# Parameters of the model
n_base = 8
LR = 0.0001
n_epochs = 150
batchsize = 8
img_w, img_h, img_ch = 112, 112, 3

clear_session()

## Build architecture and train data with it
clf = get_unet(img_ch, img_w, img_h, n_base, batchnormal =True, dropout = True)
clf.compile(loss=[dice_coef_loss], optimizer = Adam(lr = LR), metrics=[dice_coef, Precision(), Recall()])
clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data=(x_test, y_test))
clf.save(path + 'models/2D_model')

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
plt.savefig(path + 'Results/Dice_coef-2D.png', dpi = 200)

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
plt.savefig(path + 'Results/Loss-2D.png', dpi = 200)

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
plt.savefig(path + 'Results/Precision-2D.png', dpi = 200)
    
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
plt.savefig(path + 'Results/Recall-2D.png', dpi = 200)


#%% Testing and training multiple slices

        
#Load data
x_data, y_data = get_data(112, 112, 5) #get 5 slices per subject
x_train, x_test, y_train, y_test = train_test_data(x_data, y_data)#split the data
del x_data, y_data #free memory

# Path where the different files will be saved
path = 'C:/Users/cdarc/OneDrive/Documents/KTH/Deep learning/Projet/Stroke project/'

# Parameters of the model
n_base = 8
LR = 0.0001
n_epochs = 150
batchsize = 8
img_w, img_h, img_ch = 112, 112, 3

clear_session()

## Build architecture and train data with it
clf = get_unet(img_ch, img_w, img_h, n_base, batchnormal =True, dropout = True)
clf.compile(loss=[dice_coef_loss], optimizer = Adam(lr = LR), metrics=[dice_coef, Precision(), Recall()])
clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data=(x_test, y_test))
clf.save(path + 'models/5_slices_dropout')

del clf
        
clear_session()

## Build architecture and train data with it
clf = get_unet(img_ch, img_w, img_h, n_base, batchnormal =True, dropout = True)
clf.compile(loss=[dice_coef_loss], optimizer = Adam(lr = LR), metrics=[dice_coef, Precision(), Recall()])
clf_hist = clf.fit(x_train, y_train, epochs = n_epochs, batch_size = batchsize, validation_data=(x_test, y_test))


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
plt.savefig(path + 'Results/Dice_coef-2D_all.png', dpi = 200)

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
plt.savefig(path + 'Results/Loss-2D.png', dpi = 200)

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
plt.savefig(path + 'Results/Precision-2D_all.png', dpi = 200)
    
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
plt.savefig(path + 'Results/Recall-2D_all.png', dpi = 200)

        
        
        
        
        
        
        
        
        