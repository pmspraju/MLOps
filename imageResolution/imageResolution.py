#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pathlib
from glob import glob

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pickle

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import *

print(f'Pandas:{pd.__version__}, Numpy:{np.__version__}, Tensorflow:{tf.__version__}')


# In[2]:


# Resize an image based on a scale factor. Take in to consideration that it receives an image represented as Numpy array.
def resize_image(image_array, factor):
    original_image = Image.fromarray(image_array)
    new_size = np.array(original_image.size) * factor
    new_size = new_size.astype(np.int32)
    new_size = tuple(new_size)

    resized = original_image.resize(new_size)
    resized = img_to_array(resized)
    resized = resized.astype(np.uint8)

    return resized


# In[3]:


# Tightly crop an image. We need the image to fit nicely when we apply a sliding window to extract patches later. 
# SCALE is the actor we want the network to learn how to enlarge images by
def tight_crop_image(image, scale):
    height, width = image.shape[:2]
    width -= int(width % scale)
    height -= int(height % scale)

    return image[:height, :width]


# In[4]:


# Reduce resolution of an imae by downsizing and then upsizing 
def downsize_upsize_image(image, scale):
    scaled = resize_image(image, 1.0 / scale)
    scaled = resize_image(scaled, scale / 1.0)

    return scaled


# In[5]:


# Crop patches from input images. input_dim is the height&width of the images that is input to the network
def crop_input(image, x, y, input_dim):
    x_slice = slice(x, x + input_dim)
    y_slice = slice(y, y + input_dim)

    return image[y_slice, x_slice]


# In[68]:


# Crop patches of output images. label_size is the height&width of the images output by the network. 
# pad is the number of pixels used as padding to ensure we are cropping the roi accurately
def crop_output(image, x, y, label_size, pad):
    y_slice = slice(y + pad, y + pad + label_size)
    x_slice = slice(x + pad, x + pad + label_size)
    #print(x, y , x_slice, y_slice)

    return image[y_slice, x_slice]


# In[71]:


# method to prepare data
def prepare_data(path):
    SEED = 999
    SUBSET_SIZE = 100 #1500
    SCALE = 2.0
    INPUT_DIM = 33
    LABEL_SIZE = 21
    PAD = int((INPUT_DIM - LABEL_SIZE) / SCALE)
    STRIDE = 14
    
    np.random.seed(SEED)
    pattern = (path / 'images' / '*.png')
    file_patterns = str(pattern)
    dataset_paths = [*glob(file_patterns)]
    
    dataset_paths = np.random.choice(dataset_paths, SUBSET_SIZE)

    data = []; labels = []; cnt = 0

    for image_path in dataset_paths:
        image = load_img(image_path)
        image = img_to_array(image)
        image = image.astype(np.uint8); #Image.fromarray(image).show()
        image = tight_crop_image(image, SCALE); #Image.fromarray(image).show()
        scaled = downsize_upsize_image(image, SCALE); #Image.fromarray(scaled).show()

        height, width = image.shape[:2]
        #print(f'height:{height},width:{width}')
        
        for y in range(0, height - INPUT_DIM + 1, STRIDE):
            for x in range(0, width - INPUT_DIM + 1, STRIDE):
                crop = crop_input(scaled, x, y, INPUT_DIM); #Image.fromarray(crop).show()
                target = crop_output(image, x, y, LABEL_SIZE, PAD);#Image.fromarray(target).show()
                
                #cnt = cnt + 1
                
                data.append(crop)#data.append(np.array(crop).flatten()) #use np.reshape(fi,(33,33,3)) to read        
                labels.append(target)#labels.append(np.array(target).flatten());#use np.reshape(fi,(33,33,3)) to read  

                #fname = f'train/images/image_{y}_{x}.png' 
                #Image.fromarray(crop).save(os.path.join(path,fname))   

                #fname = f'train/labels/label_{y}_{x}.png'              
                #Image.fromarray(target).save(os.path.join(path,fname)) 
                #break
            #break
        #break
    #print(cnt)    
    return [data, labels]
    #pd.DataFrame({'image': data, 'label': labels})


# In[73]:


# # Test data
# path = pathlib.Path('/mnt/c/Users/pmspr/Documents/Machine Learning/Courses/Tensorflow Cert/Data/dogscats')
# [data, labels] = prepare_data(path)
# print(f'shape:{np.array(labels).shape}')
# print(f'shape:{np.array(data).shape}')
#labels = tf.convert_to_tensor(labels, np.int32)
#print(labels.shape)


# In[14]:


def build_model(height, width, depth):
    input = Input(shape=(height, width, depth))

    x = Conv2D(filters=64, 
               kernel_size=(9,9),
               kernel_initializer='he_normal'
               ) (input)
    
    x = ReLU()(x)

    x = Conv2D(filters=32, 
               kernel_size=(1,1),
               kernel_initializer='he_normal'
                ) (x)

    x = ReLU()(x)

    output = Conv2D(filters=depth, 
                    kernel_size=(5,5),
                    kernel_initializer='he_normal'
                    ) (x)

    return Model(input, output)


# In[39]:


def train_model(dim):
    path = pathlib.Path('/mnt/c/Users/pmspr/Documents/Machine Learning/Courses/Tensorflow Cert/Data/dogscats')
    [data, labels] = prepare_data(path)
    data = tf.convert_to_tensor(data, np.int32)
    labels = tf.convert_to_tensor(labels, np.int32)
    print(f'Data shape:{data.shape}, Labels shape"{labels.shape}')

    EPOCHS = 12
    optimizer = Adam(learning_rate=1e-3, decay=1e-3 / EPOCHS)
    model = build_model(dim, dim, 3)
    model.compile(loss='mse', optimizer=optimizer)
    BATCH_SIZE = 64
    
    model.fit(data, labels, batch_size=BATCH_SIZE, epochs = EPOCHS)


# In[74]:


if __name__ == "__main__":
    dim = 33
    train_model(dim)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script imageResolution.ipynb')


# In[ ]:


path = pathlib.Path('/mnt/c/Users/pmspr/Documents/Machine Learning/Courses/Tensorflow Cert/Data/dogscats')
    # df = prepare_data(path)
    # with open(os.path.join(path,'train.txt'), 'wb') as fp:
    #     pickle.dump(df[0], fp, pickle.HIGHEST_PROTOCOL)
    #np.savetxt(os.path.join(path,'train.txt'), df.values, fmt='%d')
    # with open(os.path.join(path,'train.txt'), 'a') as f:
    #     dfAsString = df.to_string(header=False, index=False)
    #     f.write(dfAsString)

