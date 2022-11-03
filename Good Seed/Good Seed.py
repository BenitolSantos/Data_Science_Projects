#!/usr/bin/env python
# coding: utf-8

# # Good Seed

# The supermarket chain Good Seed would like to explore whether Data Science can help them adhere to alcohol laws by making sure they do not sell alcohol to people underage. You are asked to conduct that evaluation, so as you set to work, keep the following in mind:
# The shops are equipped with cameras in the checkout area which are triggered when a person is buying alcohol
# Computer vision methods can be used to determine age of a person from a photo
# The task then is to build and evaluate a model for verifying people's age
# To start working on the task, you'll have a set of photographs of people with their ages indicated.

# ## Initialization

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam


# ## Load Data

# The dataset is stored in the `/datasets/faces/` folder, there you can find
# - The `final_files` folder with 7.6k photos
# - The `labels.csv` file with labels, with two columns: `file_name` and `real_age`
# 
# Given the fact that the number of image files is rather high, it is advisable to avoid reading them all at once, which would greatly consume computational resources. We recommend you build a generator with the ImageDataGenerator generator. This method was explained in Chapter 3, Lesson 7 of this course.
# 
# The label file can be loaded as an usual CSV file.

# In[2]:


labels = pd.read_csv('/datasets/faces/labels.csv')

train_datagen = ImageDataGenerator(rescale=1./255)

train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=labels,
        directory='/datasets/faces/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        seed=12345) 


# ## EDA

# In[ ]:


#from PIL import image

#Look at the dataset size.
#Explore the age distribution in the dataset.
#Print 10-15 photos for different ages on the screen to get an overall impression of the dataset.



print(labels)
print()
print('duplicated elements:',labels.duplicated().sum())
print(labels['file_name'])
print()
print('real age NaNs:',labels[labels['real_age'].isna() == True].size)
print('file name NaNs',labels[labels['file_name'].isna() == True].size)
labels.hist(bins=100)
plt.show()
labels.boxplot()
plt.show()

by_age = labels.pivot_table(index='real_age',values='file_name',aggfunc='count')

by_age.plot()

by_age.boxplot()

#I really shouldn't try to load and show all 7.6k images in the notebook, since it used to be range(len(train_gen_flow.filenames)):
for i in range(10):
     image, label = train_gen_flow.next()
     # display the image from the iterator
     for j in range(0,14):
        plt.imshow(image[j])
        plt.show()


# ### Findings

# Nothing bad so far. The median is early 20's which is perfect since the legal drinking age is 21.  Outliers are past 60. This is a right skewed graph. A data distribution with additional values on the right is said to skew to the right. This is often called positive skew.

# ## Modeling

# Define the necessary functions to train your model on the GPU platform and build a single script containing all of them along with the initialization section.
# 
# To make this task easier, you can define them in this notebook and run a ready code in the next section to automatically compose the script.
# 
# The definitions below will be checked by project reviewers as well, so that they can understand how you built the model.

# In[2]:


def load_train(path):
    
    """
    It loads the train part of dataset from path
    
    Remember you still need to load what you need seperately from the path.
    Use flow_from_dataframe instead of flow_from_directory.
    The path is /datasets/faces/ and we don't need the / at the end for folders.
    Remember to add the subsets to split it!
    
    Reminder to me: horizontal_flip should be used as an argument when ImageDataGenerator is initialized, not as argument of flow_from_dataframe method
    Any augmentations should only be applied to the training data, so horizontal flip shouldn't be applied in load_test function (otherwise we can end up with an overly optimistic estimate of the model's generalization performance, and the goal of augmentations is just to create more traning data basically for free)
    """
    
    # place your code here
    labels = pd.read_csv(path+'labels.csv')                                                     
    train_datagen = ImageDataGenerator(rescale= 1/255, validation_split=0.25, horizontal_flip=True)  
    train_datagen_flow = train_datagen.flow_from_dataframe(
        dataframe = labels,
        directory = path + 'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=42)
    return train_datagen_flow


# In[3]:


def load_test(path):
    
    """
    It loads the validation/test part of dataset from path
    Reminder to me:In load_test it should be subset='validation', otherwise we're evaluating the model on the same data it was trained on :)
    """
    
    # place your code here
    labels = pd.read_csv(path+'labels.csv')                                                     
    train_datagen = ImageDataGenerator(rescale= 1/255, validation_split=0.25)  
    train_datagen_flow = train_datagen.flow_from_dataframe(
        dataframe = labels,
        directory = path + 'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=42)
    return test_datagen_flow


# In[4]:


def create_model(input_shape):
    
    """
    It defines the model
    As this is the regression task, you may want to use either the MSE loss function or the MAE one. However, neural networks with the MSE loss function are typically trained faster than with the MAE ones. In either of the cases, MAE remains the evaluation metric.
    input_shape is gotten from the parameter not inputed yourself as (150, 150, 3)
    """
    
    # place your code here
    backbone = ResNet50(input_shape= input_shape,
                    weights='imagenet', 
                    include_top= False)

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))
    optimizer = Adam(lr=0.0005)
    model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['mae']) 
    return model


# In[6]:


def train_model(model, train_data, test_data, batch_size=None, epochs=10,
                steps_per_epoch=None, validation_steps=None):

    """
    Trains the model given the parameters
    """
    
    # place your code here
    
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)
    model.fit(train_data, 
              validation_data= test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)
    return model


# ## Prepare the Script to Run on the GPU Platform

# Given you've defined the necessary functions you can compose a script for the GPU platform, download it via the "File|Open..." menu, and to upload it later for running on the GPU platform.
# 
# N.B.: The script should include the initialization section as well. An example of this is shown below.

# In[ ]:


# prepare a script to run on the GPU platform

init_str = """
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
"""

import inspect

with open('run_model_on_gpu.py', 'w') as f:
    
    f.write(init_str)
    f.write('\n\n')
        
    for fn_name in [load_train, load_test, create_model, train_model]:
        
        src = inspect.getsource(fn_name)
        f.write(src)
        f.write('\n\n')


# ### Output

# Place the output from the GPU platform as an Markdown cell here.

# 2022-11-01 04:00:02.256853: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer.so.6
# 2022-11-01 04:00:02.380907: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer_plugin.so.6
# Using TensorFlow backend.
# Found 5694 validated image filenames.
# Found 1897 validated image filenames.
# 2022-11-01 04:00:07.287929: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
# 2022-11-01 04:00:07.399108: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2022-11-01 04:00:07.399329: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
# pciBusID: 0000:00:1e.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
# coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.78GiB deviceMemoryBandwidth: 836.37GiB/s
# 2022-11-01 04:00:07.399382: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2022-11-01 04:00:07.399419: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2022-11-01 04:00:07.494208: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
# 2022-11-01 04:00:07.513218: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
# 2022-11-01 04:00:07.687024: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
# 2022-11-01 04:00:07.707176: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
# 2022-11-01 04:00:07.707250: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 2022-11-01 04:00:07.707408: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2022-11-01 04:00:07.707700: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2022-11-01 04:00:07.707870: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
# 2022-11-01 04:00:07.708315: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# 2022-11-01 04:00:07.751700: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2300015000 Hz
# 2022-11-01 04:00:07.754005: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x4f620d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
# 2022-11-01 04:00:07.754046: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
# 2022-11-01 04:00:07.923526: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2022-11-01 04:00:07.923852: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x2da0720 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
# 2022-11-01 04:00:07.923876: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
# 2022-11-01 04:00:07.924134: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2022-11-01 04:00:07.924337: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
# pciBusID: 0000:00:1e.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
# coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.78GiB deviceMemoryBandwidth: 836.37GiB/s
# 2022-11-01 04:00:07.924386: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2022-11-01 04:00:07.924405: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2022-11-01 04:00:07.924445: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
# 2022-11-01 04:00:07.924461: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
# 2022-11-01 04:00:07.924476: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
# 2022-11-01 04:00:07.924490: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
# 2022-11-01 04:00:07.924500: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 2022-11-01 04:00:07.924586: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2022-11-01 04:00:07.924798: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2022-11-01 04:00:07.924964: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
# 2022-11-01 04:00:07.926968: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2022-11-01 04:00:10.123127: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
# 2022-11-01 04:00:10.123179: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0 
# 2022-11-01 04:00:10.123190: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N 
# 2022-11-01 04:00:10.125037: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2022-11-01 04:00:10.125320: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2022-11-01 04:00:10.125492: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
# 2022-11-01 04:00:10.125547: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14988 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:1e.0, compute capability: 7.0)
# Downloading data from https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# 
#     8192/94765736 [..............................] - ETA: 1s
#  8871936/94765736 [=>............................] - ETA: 0s
# 18292736/94765736 [====>.........................] - ETA: 0s
# 29827072/94765736 [========>.....................] - ETA: 0s
# 37756928/94765736 [==========>...................] - ETA: 0s
# 49274880/94765736 [==============>...............] - ETA: 0s
# 52871168/94765736 [===============>..............] - ETA: 0s
# 58793984/94765736 [=================>............] - ETA: 0s
# 66363392/94765736 [====================>.........] - ETA: 0s
# 75505664/94765736 [======================>.......] - ETA: 0s
# 87220224/94765736 [==========================>...] - ETA: 0s
# 94773248/94765736 [==============================] - 1s 0us/step
# <class 'tensorflow.python.keras.engine.sequential.Sequential'>
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# Train for 178 steps, validate for 60 steps
# Epoch 1/10
# 2022-11-01 04:00:28.643427: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2022-11-01 04:00:29.831296: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 178/178 - 63s - loss: 203.1177 - mae: 10.6800 - val_loss: 358.4354 - val_mae: 14.0400
# Epoch 2/10
# 178/178 - 39s - loss: 113.0923 - mae: 8.0743 - val_loss: 525.2079 - val_mae: 17.7680
# Epoch 3/10
# 178/178 - 39s - loss: 88.4932 - mae: 7.1680 - val_loss: 423.2995 - val_mae: 15.3790
# Epoch 4/10
# 178/178 - 39s - loss: 74.6991 - mae: 6.5607 - val_loss: 170.1841 - val_mae: 10.1973
# Epoch 5/10
# 178/178 - 40s - loss: 60.3922 - mae: 5.9806 - val_loss: 159.6877 - val_mae: 10.1872
# Epoch 6/10
# 178/178 - 40s - loss: 50.2668 - mae: 5.3520 - val_loss: 84.8826 - val_mae: 6.9760
# Epoch 7/10
# Epoch 8/10
# 178/178 - 39s - loss: 44.3392 - mae: 5.0997 - val_loss: 91.1241 - val_mae: 7.1879
# 178/178 - 39s - loss: 37.8437 - mae: 4.7105 - val_loss: 100.4241 - val_mae: 7.7159
# Epoch 9/10
# 178/178 - 39s - loss: 31.0166 - mae: 4.2301 - val_loss: 81.8848 - val_mae: 6.6861
# Epoch 10/10
# 178/178 - 39s - loss: 25.5607 - mae: 3.8835 - val_loss: 83.7814 - val_mae: 6.7650
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# 60/60 - 10s - loss: 83.7814 - mae: 6.7650
# Test MAE: 6.7650
# 2022-11-01 04:00:02.256853: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer.so.6
# 2022-11-01 04:00:02.380907: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer_plugin.so.6
# Using TensorFlow backend.
# Found 5694 validated image filenames.
# Found 1897 validated image filenames.
# 2022-11-01 04:00:07.287929: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
# 2022-11-01 04:00:07.399108: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2022-11-01 04:00:07.399329: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
# pciBusID: 0000:00:1e.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
# coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.78GiB deviceMemoryBandwidth: 836.37GiB/s
# 2022-11-01 04:00:07.399382: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2022-11-01 04:00:07.399419: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2022-11-01 04:00:07.494208: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
# 2022-11-01 04:00:07.513218: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
# 2022-11-01 04:00:07.687024: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
# 2022-11-01 04:00:07.707176: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
# 2022-11-01 04:00:07.707250: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 2022-11-01 04:00:07.707408: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2022-11-01 04:00:07.707700: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2022-11-01 04:00:07.707870: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
# 2022-11-01 04:00:07.708315: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# 2022-11-01 04:00:07.751700: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2300015000 Hz
# 2022-11-01 04:00:07.754005: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x4f620d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
# 2022-11-01 04:00:07.754046: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
# 2022-11-01 04:00:07.923526: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2022-11-01 04:00:07.923852: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x2da0720 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
# 2022-11-01 04:00:07.923876: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
# 2022-11-01 04:00:07.924134: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2022-11-01 04:00:07.924337: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
# pciBusID: 0000:00:1e.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
# coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.78GiB deviceMemoryBandwidth: 836.37GiB/s
# 2022-11-01 04:00:07.924386: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2022-11-01 04:00:07.924405: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2022-11-01 04:00:07.924445: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
# 2022-11-01 04:00:07.924461: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
# 2022-11-01 04:00:07.924476: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
# 2022-11-01 04:00:07.924490: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
# 2022-11-01 04:00:07.924500: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 2022-11-01 04:00:07.924586: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2022-11-01 04:00:07.924798: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2022-11-01 04:00:07.924964: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
# 2022-11-01 04:00:07.926968: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2022-11-01 04:00:10.123127: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
# 2022-11-01 04:00:10.123179: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0 
# 2022-11-01 04:00:10.123190: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N 
# 2022-11-01 04:00:10.125037: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2022-11-01 04:00:10.125320: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2022-11-01 04:00:10.125492: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
# 2022-11-01 04:00:10.125547: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14988 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:1e.0, compute capability: 7.0)
# Downloading data from https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# 
#     8192/94765736 [..............................] - ETA: 1s
#  8871936/94765736 [=>............................] - ETA: 0s
# 18292736/94765736 [====>.........................] - ETA: 0s
# 29827072/94765736 [========>.....................] - ETA: 0s
# 37756928/94765736 [==========>...................] - ETA: 0s
# 49274880/94765736 [==============>...............] - ETA: 0s
# 52871168/94765736 [===============>..............] - ETA: 0s
# 58793984/94765736 [=================>............] - ETA: 0s
# 66363392/94765736 [====================>.........] - ETA: 0s
# 75505664/94765736 [======================>.......] - ETA: 0s
# 87220224/94765736 [==========================>...] - ETA: 0s
# 94773248/94765736 [==============================] - 1s 0us/step
# <class 'tensorflow.python.keras.engine.sequential.Sequential'>
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# Train for 178 steps, validate for 60 steps
# Epoch 1/10
# 2022-11-01 04:00:28.643427: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2022-11-01 04:00:29.831296: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 178/178 - 63s - loss: 203.1177 - mae: 10.6800 - val_loss: 358.4354 - val_mae: 14.0400
# Epoch 2/10
# 178/178 - 39s - loss: 113.0923 - mae: 8.0743 - val_loss: 525.2079 - val_mae: 17.7680
# Epoch 3/10
# 178/178 - 39s - loss: 88.4932 - mae: 7.1680 - val_loss: 423.2995 - val_mae: 15.3790
# Epoch 4/10
# 178/178 - 39s - loss: 74.6991 - mae: 6.5607 - val_loss: 170.1841 - val_mae: 10.1973
# Epoch 5/10
# 178/178 - 40s - loss: 60.3922 - mae: 5.9806 - val_loss: 159.6877 - val_mae: 10.1872
# Epoch 6/10
# 178/178 - 40s - loss: 50.2668 - mae: 5.3520 - val_loss: 84.8826 - val_mae: 6.9760
# Epoch 7/10
# 178/178 - 39s - loss: 44.3392 - mae: 5.0997 - val_loss: 91.1241 - val_mae: 7.1879
# Epoch 8/10
# 178/178 - 39s - loss: 37.8437 - mae: 4.7105 - val_loss: 100.4241 - val_mae: 7.7159
# Epoch 9/10
# 178/178 - 39s - loss: 31.0166 - mae: 4.2301 - val_loss: 81.8848 - val_mae: 6.6861
# Epoch 10/10
# 178/178 - 39s - loss: 25.5607 - mae: 3.8835 - val_loss: 83.7814 - val_mae: 6.7650
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# 60/60 - 10s - loss: 83.7814 - mae: 6.7650
# Test MAE: 6.7650

# ## Conclusions

# Running a neural network model requires a lot of computing power so it needs to be done on the GPU platform. Results can be captured by printing the model's output. The output was saved and returned to me.
# 
# I used a pre-trained model for the task, specifically, one which was pre-trained on a large dataset of images called imagenet. You can check the content of that dataset at: http://www.image-net.org/. This is not a dataset that's strictly about faces, so the pre-trained weights might not work the best for faces. However, it is somewhat valid to use this dataset for face classification because the imagenet dataset is mostly about natural objects and contains a subset of faces too. By setting the 'weights' parameter to 'imagenet,' we're able to load weights of this pre-trained neural network for the ResNet50 architecture.
# By the way, there are more than 23 million trainable parameters in the ResNet50 model. This is the reason why using it without a GPU platform would cause a very, very long wait time for a model to be trained, even if it has been pre-trained.
# 
# In one article about this dataset I am working with (http://people.ee.ethz.ch/~timofter/publications/Agustsson-FG-2017.pdf), the lowest MAE value reached is 5.4. Meaning, if you get MAE less than 7, it would be a great result! 
# 
# With an MAE less than 7 (6.7650), the model is more than enough to detect underage people.

# # Checklist

# - [X]  Notebook was opened
# - [X]  The code is error free
# - [X]  The cells with code have been arranged by order of execution
# - [X]  The exploratory data analysis has been performed
# - [X]  The results of the exploratory data analysis are presented in the final notebook
# - [X]  The model's MAE score is not higher than 8
# - [X]  The model training code has been copied to the final notebook
# - [X]  The model training output has been copied to the final notebook
# - [X]  The findings have been provided based on the results of the model training

# In[ ]:




