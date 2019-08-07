#!/usr/bin/env python
# coding: utf-8

# In[3]:


import PIL
import numpy as np 
import matplotlib as plt
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
import sys 


# In[4]:


if len(sys.argv)< 4 : 
    print (" need more arguments")
    sys.exit()
else :
     kmeans = int(sys.argv[1])
     if kmeans<3 :
         print(" need more clusters, increase number")
     input= sys.argv[2]
     output = sys.argv[3]

image = Image.open(input)
f1=image


# In[33]:


# file='C:/Users/Rutuja Moharil/kmeansimage.jpg'
# #img=stage_1_PIL(file);
# f1=Image.open(file)
# #pixels=f1.load()
# plt.imshow(f1)


# In[37]:


width, height = f1.size
k = kmeans
# 5 is the number of feature vectcors i.e R,G,B,x,y positions 
Vector = np.ndarray(shape=(width * height, 5), dtype=float)
for y in range(0,height):
    for x in range(0,width):
        #cpixel = get.pixels(x, y)
        #all_pixels.append(cpixel)
       # xy=(x,y)
        #rgb = get.pixels(x,y)
        rgb=f1.getpixel((x,y))
        Vector[x + y * width, 0] = rgb[0]
        Vector[x + y * width, 1] = rgb[1]
        Vector[x + y * width, 2] = rgb[2]
        Vector[x + y * width, 3] = x
        Vector[x + y * width, 4] = y
#Standardising the features 
Vector_normal= preprocessing.scale(Vector)

#Initialize Random values for centroids 
mean = np.mean(Vector_normal, axis = 0)
std = np.std(Vector_normal, axis = 0)
# k is the number of clusters 5 is the feature vector space
#centers stores the centroid values
centers = np.random.randn(k,5)*std + mean
print(centers.shape)
print(Vector_normal.shape)
plt.scatter(Vector_normal[:,0], Vector_normal[:,1], s=2)
plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)


# In[38]:


old_center = np.zeros(centers.shape) # to store old centers
update_center =np.zeros(centers.shape) # Store new centers
iterations=0
clusters = np.zeros(width)
distances = np.zeros((width*height,k))
error = np.linalg.norm(update_center - old_center)
Max_iter =200
# If the estimate of that center stays the same, exit loop
while (iterations < Max_iter):
    #print (iterations)
    for i in range(k):
# Finding euclidean distance and trying to minimize the error
        if iterations==0:
            distances[:,i] = np.linalg.norm(Vector_normal - centers[i],axis=1)
        else:
            distances[:,i] = np.linalg.norm(Vector_normal - update_center[i],axis=1)
        
 # Assign all training data to closest center 

    clusters = np.argmin(distances, axis = 1)
 ##Cluster empty condition    
    clusterToCheck = np.arange(k)
    clustersEmpty = np.in1d(clusterToCheck, clusters)
    for index, item in enumerate(clustersEmpty):
        if item == False:
           clusters[np.random.randint(len(clusters))] = index
    old_center = update_center
    
 # Calculate mean for every cluster and update the center
    for i in range(k):
        #print(Vector_normal[clusters == i])
        update_center[i] = np.mean(Vector_normal[clusters == i],axis=0)
    error = np.linalg.norm(update_center - old_center)
    iterations =iterations+1


# In[39]:


## Image reconstruction
for index, item in enumerate(clusters):
    Vector[index][0] = int(round(update_center[item][0]*255 ))
    Vector[index][1] = int(round(update_center[item][1]*255 ))
    Vector[index][2] = int(round(update_center[item][2] *255))

#Save image
image = Image.new("RGB", (width, height))

for y in range(height):
    for x in range(width):
        image.putpixel((x, y), (int(Vector[y * width + x][0]),int(Vector[y * width + x][1]),int(Vector[y * width + x][2])))
plt.imshow(image)
#image.save('powerpuffcorrect.jpg')
image.save(output)

