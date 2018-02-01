import numpy as np
import pandas as pd
import glob
import os
from skimage import io,color,filters,img_as_ubyte,img_as_uint
from skimage.transform import resize
from skimage.feature import hog,match_template,canny
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
import cv2
import sys

import pickle

#from cyvlfeat import sift
#import cyvlfeat
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from scipy.cluster.vq import vq,kmeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib





def compute_dress_similarity(url_path):
#### constants, and load models
    rx = 256
    ry = 256
    cx=200
    cy=100
    
    PROJECT_ROOT = os.path.realpath(__file__).split('model')[0]

#    IMAGE_ROOT = '/home/miracle/Work/Projects/SIFT_classifie_modified/HOG_classifier_sleeve'
    dress_length_path = os.path.join(PROJECT_ROOT,'db/SVM_dress_length_model_1_25.pkl')
    SVM_length = joblib.load(dress_length_path)

    knn = joblib.load(os.path.join(PROJECT_ROOT,'db/knn_harrods.pkl'))
    pca_color = joblib.load(os.path.join(PROJECT_ROOT,'db/PCA_harrods.pkl'))

    bk0 = joblib.load(os.path.join(PROJECT_ROOT,'db/mbk.pkl'))
    scaler0 = joblib.load(os.path.join(PROJECT_ROOT,'db/scaler.pkl'))
    svm0 = joblib.load(os.path.join(PROJECT_ROOT,'db/SVM_sleevelength.pkl'))


#### image preprocessing
    aspect_ratio, gray_cropped , color_cropped = image_preprocessing(PROJECT_ROOT,url_path)

#### compute color features
    color_path = os.path.join(PROJECT_ROOT,'static/images','color_cropped_0.jpg')
    binary_path = os.path.join(PROJECT_ROOT,'static/images','binary_cropped_0.jpg')
    color_hist,main_color = compute_colorhist_oneframe(color_path,binary_path)
    color_hist1 = pca_color.transform(color_hist)
    color_similarity = knn.kneighbors(color_hist1.reshape(1,-1),3,return_distance=False)[0]
   
	
    return color_similarity, main_color[0]

def resize_image(img,rx,ry):

    ratio = rx/ry
    white = img[0,0,:]
    x= img.shape[0]
    y = img.shape[1]
    if(x>int(y*ratio)):
        padx = np.zeros((x,int(x/ratio/2) - int(y/2),3),dtype = np.uint8)
        padx[:,:,:] = white
        img1 = np.concatenate((padx,img,padx),axis = 1)
        img1 = resize(img1,(rx,ry),mode = 'constant')
    elif (x == int(y*ratio)):
        img1 = resize(img,(rx,ry),mode = 'constant')
    else:
        pady = np.zeros((int(y*ratio/2) - int(x/2),x,3),dtype = np.uint8)
        pady[:,:,:] = white
        img1 = np.concatenate((pady,img,pady), axis = 0)
        img1 = resize(img1,(rx,ry),mode ='constant')
    return img1


def image_preprocessing(PROJECT_ROOT,fname):

    rx = 256
    ry = 256
    cx=200
    cy=100
    
#    filename = os.path.join('/home/miracle/flask/myflask',fname)
    im = io.imread(fname)
    im = resize_image(im,rx,ry)

    gray = color.rgb2gray(im)
    th_edge = filters.sobel(gray)
    edged = th_edge>0
# binary image
    test_im_binary = edged
    
    _,cnts, _ = cv2.findContours(img_as_ubyte(edged), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    dress = gray[y:y + h, x:x + w]
    dress = cv2.resize(dress,(cy,cx))
# grayscale cropped image
    test_im_gray_cropped = dress

# binary cropped image
    dress_b = edged[y:y + h, x:x + w]
    test_im_binary_cropped = resize(dress_b,(cx,cy))
    print(PROJECT_ROOT)
    print(os.path.join(PROJECT_ROOT,'static/images/binary_cropped_0.jpg'))
    io.imsave(os.path.join(PROJECT_ROOT,'static/images/binary_cropped_0.jpg'),
              img_as_uint(edged))
    

# color cropped image
    test_im_color_cropped = im[y:y + h, x:x + w,:]
    test_im_color_cropped = resize(test_im_color_cropped,(cx,cy))
    io.imsave(os.path.join(PROJECT_ROOT,'static/images/color_cropped_0.jpg'),
              img_as_uint(test_im_color_cropped))
    

    hw = np.sum(dress_b[int(h/2),:])
    
    aspect_ratio = x = np.hstack((h,w,hw))
    
# compute color histogram and main color
#     color_hist,main_color = compute_colorhist_oneframe(test_im_color_cropped,test_im_binary_cropped)
    
#     return aspect_ratio,test_im_gray_cropped,color_hist,main_color
    return aspect_ratio,test_im_gray_cropped,test_im_color_cropped


def compute_colorhist_oneframe(color_path,binary_path):
    n_img = 1
#     cimg = np.zeros((n_img,20000))
#     bimg = np.zeros((n_img,20000))

#     hue_upper= np.zeros((n_img,180))
#     sat_upper = np.zeros((n_img,256))
#     val_upper = np.zeros((n_img,256))

#     hue_lower= np.zeros((n_img,180))
#     sat_lower = np.zeros((n_img,256))
#     val_lower = np.zeros((n_img,256))
    
    l_upper= np.zeros((n_img,180))
    a_upper = np.zeros((n_img,256))
    b_upper = np.zeros((n_img,256))
    
    l_mid= np.zeros((n_img,180))
    a_mid = np.zeros((n_img,256))
    b_mid = np.zeros((n_img,256))

    l_lower= np.zeros((n_img,180))
    a_lower = np.zeros((n_img,256))
    b_lower = np.zeros((n_img,256))
    
    main_color = np.zeros((n_img,9))

    imb = cv2.imread(binary_path)
    imb_gray= cv2.cvtColor(imb,cv2.COLOR_RGB2GRAY)
#     bimg = binary.reshape(1,20000)
#     imb_gray= cv2.cvtColor(img_as_ubyte(binary),cv2.COLOR_RGB2GRAY)

    im_color = cv2.imread(color_path)
#     imc_lab = cv2.cvtColor(img_as_ubyte(color),cv2.COLOR_RGB2LAB)
    imc_lab = cv2.cvtColor(im_color,cv2.COLOR_RGB2LAB)
    # fill the holes in binarnized image    
    _, cnts, _ = cv2.findContours(imb_gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    ctr = np.array(c).reshape((-1,1,2)).astype(np.int32)
    mask = cv2.drawContours(imb,[ctr],0,(255,255,255),-1)
    mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)

    mask_upper = mask[40:60,40:60]
    mask_mid = mask[90:110,40:60]
    mask_lower = mask[160:180,40:60]
        
    l_upper = cv2.calcHist([imc_lab[40:60,40:60]],[0],None,[180],[0,180]).transpose()
    a_upper = cv2.calcHist([imc_lab[40:60,40:60]],[1],None,[256],[0,256]).transpose()
    b_upper = cv2.calcHist([imc_lab[40:60,40:60]],[2],None,[256],[0,256]).transpose()
        
    l_mid = cv2.calcHist([imc_lab[90:110,40:60]],[0],None,[180],[0,180]).transpose()
    a_mid = cv2.calcHist([imc_lab[90:110,40:60]],[1],None,[256],[0,256]).transpose()
    b_mid = cv2.calcHist([imc_lab[90:110,40:60]],[2],None,[256],[0,256]).transpose()
    
     
    l_lower = cv2.calcHist([imc_lab[160:180,40:60]],[0],None,[180],[0,180]).transpose()
    a_lower = cv2.calcHist([imc_lab[160:180,40:60]],[1],None,[256],[0,256]).transpose()
    b_lower = cv2.calcHist([imc_lab[160:180,40:60]],[2],None,[256],[0,256]).transpose()
        
    imc_lab1 = color.rgb2lab(img_as_uint(im_color))
    imc_lab2= imc_lab.reshape((imc_lab.shape[0]*imc_lab.shape[1],3))
    clt = MiniBatchKMeans(n_clusters=8)
    labels = clt.fit_predict(imc_lab2)

    idx = clt.counts_.argsort()[-3:][::-1]
    main_color = clt.cluster_centers_[idx].reshape((1,-1))
    
    color_hist = np.hstack((l_upper,a_upper,b_upper,l_mid,a_mid,b_mid,l_lower,a_lower,b_lower))    

    return color_hist,main_color





def load_image(img_file):
    im = io.imread(img_file)
#    im = resize(im,(int(im.shape[0] / 3), int(im.shape[1] / 3)))
    
    return im


def compute_rgb(im):
    

 #   n_seg = 0
 #   th = 1

  #   while n_seg<2:
 #        labels1 = segmentation.slic(im, compactness= th , n_segments=100)
  #       out1 = color.label2rgb(labels1, im, kind='avg')
   #      g = graph.rag_mean_color(im, labels1, mode='similarity')
    #     labels2 = graph.cut_normalized(labels1, g,num_cuts = 10)
     #    out2 = color.label2rgb(labels2, im, kind='avg')
 #        value_unique, value_counts = np.unique(labels2,return_counts = True)
  #       n_seg = len(value_unique)
 #        th = th/10
  #       if(th<0.001):
  #           break
            
 #    value_unique, value_counts = np.unique(labels2,return_counts = True)
 #    seq = np.argsort(value_counts)
 #    if n_seg == 2:
 #        mask = labels2 == value_unique[seq[-1]]
 #    else:
 #        mask = labels2 == value_unique[seq[-2]]
    xr = range(220,301)
    yr = range(220,301)

    x = im[xr,yr,0]
    y = im[xr,yr,1]
    z = im[xr,yr,2]
    rgb = [np.average(x),np.average(y),np.average(z)]
    return rgb


