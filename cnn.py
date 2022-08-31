#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# load and evaluate a saved model
from numpy import loadtxt
from tensorflow.keras.models import load_model


# In[2]:


model=load_model('model.h5')
model.summary()


# In[3]:


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


# In[4]:


score=model.evaluate(test_images,test_labels)
print(score)


# In[ ]:





# In[5]:


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()


# In[6]:


score=model.evaluate(test_images[20:30],test_labels[20:30])
print(score)


# In[7]:


print(test_images[35].shape)


# In[8]:


img=test_images[35].reshape(1,32,32,3)


# In[9]:


output=model.predict(img)
print(output)


# In[10]:




