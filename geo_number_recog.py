#need to import pictures as image arrays
from PIL import Image
import numpy as np
#start with your own created test data
#here we start with a bitmap from paint 
#shape,size,view numpy array attributes
train_img_0 = np.array(Image.open('D:\\Documents\\programming\\ocr\\training_data\\0_training.bmp'), dtype=int)
train_img_1 = np.array(Image.open('D:\\Documents\\programming\\ocr\\training_data\\1_training.bmp'), dtype=int)

trian_img = np.array(Image.open('D:\\Documents\\programming\\ocr\\training_data\\triangle.bmp'), dtype=int)
#.reshape(512,512,3)
#dtype by default is a bool
#use binary properties, isnt that just some kind of filter I guess so.
#Do binary operations on the data. Do an exclusive or of the two images.
#The two images are one a triangle, The other is a number.
# With an "AND" bitwise operation of all the rows we can see what values are towards the left and what values are towards the right.
#Start with a test to differentiate against 0 and 1 using simple geometry
#The model is what the bitwise "AND" typically would look like for each number.

#bitwise AND of the training and triangle bitmap, collection of these could make the model for each number.
#https://numpy.org/doc/2.2/reference/generated/numpy.bitwise_and.html
img_comb = np.bitwise_and(train_img_0, trian_img)

#find area in different quadrants of 64x64. Area= number of pixels for now.
#count the number of 0s or 1s found in 32x32 quadrants for different kinds of images.
#Start off with 0 and 1. numpy split
#https://numpy.org/doc/stable/reference/generated/numpy.split.html
#count_quadrant function
img_split_h1 = [x[0:32] for x in img_comb]
img_split_h2 = [x[32:] for x in img_comb]

img_split_q1 = img_split_h1[0:32]
img_split_q3 = img_split_h1[32:]

img_split_q2 = img_split_h2[0:32]
img_split_q4 = img_split_h2[32:]

#16x16
#img_split_q5 = np.array([img_split_q1[28:],img_split_q2[0:4],img_split_q3[28:],img_split_q4[0:4]])

#numpy count/search for the number of elements in array
#the number of zeros in the array
#the number of pixels overlapping with the triangle in each quadrant
num_0_q1 = np.unique(img_split_q1, return_counts=True)[1][0]
num_0_q2 = np.unique(img_split_q2, return_counts=True)[1][0]
num_0_q3 = np.unique(img_split_q3, return_counts=True)[1][0]
num_0_q4 = np.unique(img_split_q4, return_counts=True)[1][0]
#num_0_q5 = np.unique(img_split_q5, return_counts=True)[1][0]

num_0_pixels = np.array([num_0_q1,num_0_q2,num_0_q3,num_0_q4])
#num_0_pixels = np.array([num_0_q1,num_0_q2,num_0_q3,num_0_q4,num_0_q5])


#bitwise AND of the training and triangle bitmap, collection of these could make the model for each number.
#https://numpy.org/doc/2.2/reference/generated/numpy.bitwise_and.html
img_comb_1 = np.bitwise_and(train_img_1, trian_img)

#find area in different quadrants of 64x64. Area= number of pixels for now.
#count the number of 0s or 1s found in 32x32 quadrants for different kinds of images.
#Start off with 0 and 1. numpy split
#https://numpy.org/doc/stable/reference/generated/numpy.split.html
#count_quadrant function
img_split_h1 = [x[0:32] for x in img_comb_1]
img_split_h2 = [x[32:] for x in img_comb_1]

img_split_q1 = img_split_h1[0:32]
img_split_q3 = img_split_h1[32:]

img_split_q2 = img_split_h2[0:32]
img_split_q4 = img_split_h2[32:]

#16x16
#img_split_q5 = np.array([img_split_q1[28:],img_split_q2[0:4],img_split_q3[28:],img_split_q4[0:4]])

#numpy count/search for the number of elements in array
#the number of zeros in the array
#the number of pixels overlapping with the triangle in each quadrant
num_1_q1 = np.unique(img_split_q1, return_counts=True)[1][0]
num_1_q2 = np.unique(img_split_q2, return_counts=True)[1][0]
num_1_q3 = np.unique(img_split_q3, return_counts=True)[1][0]
num_1_q4 = np.unique(img_split_q4, return_counts=True)[1][0]
#num_1_q5 = np.unique(img_split_q5, return_counts=True)[1][0]

num_1_pixels = np.array([num_1_q1,num_1_q2,num_1_q3,num_1_q4])
#num_1_pixels = np.array([num_1_q1,num_1_q2,num_1_q3,num_1_q4,num_1_q5])

import pdb;pdb.set_trace()
print(img_comb[1])
