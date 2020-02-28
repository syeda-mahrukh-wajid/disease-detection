# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 13:01:51 2020

@author: hp
"""

from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# load the model we saved
model = load_model('group.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


test_image = image.load_img('C:/Users/hp/Desktop/infimage.jpeg', 
                            target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
if result [0][0] == 1:
    prediction = 'pneumonia'
    print("drink turmeric tea")
else:
    prediction = 'normal'
  