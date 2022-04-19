import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import pickle
import re
import math
#camera calibration 
def chunk(x):
    try:
        return int(x)
    except:
        return x
     
def sort_key(s):
   
    return [chunk(c) for c in re.split('([0-9]+)', s)]

def sort_fn(l):
   
    l.sort(key= sort_key)
    
def plot_images(data, layout='row', cols=2, figsize=(20, 12)):
  
    rows = math.ceil(len(data) / cols)
    f, ax = plt.subplots(figsize=figsize)
    if layout == 'row':
        for idx, d in enumerate(data):
            img, title = d

            plt.subplot(rows, cols, idx+1)
            plt.title(title, fontsize=20)
            plt.axis('off')
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
                #len(img.shape) gives you two, it has a single channel a 2D array (grayscale image).
               
            elif len(img.shape) == 3:
                plt.imshow(img)
                
    elif layout == 'col':
        counter = 0
        for r in range(rows):
            for c in range(cols):
                img, title = data[r + rows*c]
                nb_channels = len(img.shape)
                
                plt.subplot(rows, cols, counter+1)
                plt.title(title, fontsize=20)
                plt.axis('off')
                if len(img.shape) == 2:
                    plt.imshow(img, cmap='gray')
                
                elif len(img.shape) == 3:
                    plt.imshow(img)
              
                counter += 1

    return ax
  
  def calibrate_camera():
    
    imgpaths = glob.glob('camera_cal/calibration*.jpg')
    sort_fn(imgpaths)

  
    image = cv2.imread(imgpaths[0])
    imshape = image.shape[:2]   
    plt.imshow(image)
    plt.show()
    print('Image shape: {}'.format(image.shape))
    objpoints = []
    imgpoints = []
    nx = 9 
    ny = 6 
    objp = np.zeros([ny*nx, 3], dtype=np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    for idx, imgpath in enumerate(imgpaths):
        img = cv2.imread(imgpath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            imgpoints.append(corners)
            objpoints.append(objp)

            cv2.imshow('img', img)
            cv2.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imshape[::-1], None, None)
    cv2.destroyAllWindows()
    return mtx, dist
if os.path.exists('camera_calib.p'):
    with open('camera_calib.p', mode='rb') as f:
        data = pickle.load(f)
        mtx, dist = data['mtx'], data['dist']
        print('camera calibration matrix & dist coefficients are saved !')
else:
    mtx, dist = calibrate_camera()
    with open('camera_calib.p', mode='wb') as f:
        pickle.dump({'mtx': mtx, 'dist': dist}, f)

def undistort(img, mtx, dist):
   
    
    undistort = cv2.undistort(img, mtx, dist, None, mtx)
    return undistort
  
 

ccimg = cv2.imread('camera_cal/calibration1.jpg')
ccimg_undist = undistort(ccimg, mtx, dist)

plot_images([
    (ccimg, 'Original Image'),
    (ccimg_undist, 'Undistorted Image')
])

img_orig = mpimg.imread(test_img_paths[6])
img = undistort(img_orig, mtx, dist)

plot_images([
    (img_orig, 'Original Image'),
    (img, 'Undistorted Image')
])
