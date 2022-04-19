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

###############Perspective Transformation
img_dimensions = (720, 1280)
#crop unwanted parts of img by setting intensty=0
def modify_img(img, vertices):
    vertices = np.array(vertices, ndmin=3, dtype=np.int32)
    if len(img.shape) == 3:
        fill_color = (255,) * 3
    else:
        fill_color = 255         
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, vertices, fill_color)
    return cv2.bitwise_and(img, mask)

#wrap the img (convert to bird's eye view)
def wrap_img(img, warp_shape, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, warp_shape, flags=cv2.INTER_LINEAR)
    return warped, M, invM

#prepare img (correct the distortion,convert to bird's eye view,crop unwanted parts)
def preprocess_image(img, visualise=False):
    ysize = img.shape[0]
    xsize = img.shape[1]
    undist = undistort(img, mtx, dist)
    src = np.float32([
        (696,455),    
        (587,455), 
        (235,700),  
        (1075,700)
    ])
    dst = np.float32([
        (xsize - 350, 0),
        (350, 0),
        (350, ysize),
        (xsize - 350, ysize)
    ])
    warped, M, invM = wrap_img(undist, (xsize, ysize), src, dst)
    vertices = np.array([
        [200, ysize],
        [200, 0],
        [1100, 0],
        [1100, ysize]
    ])
    roi = modify_img(warped, vertices)
    if visualise:
        img_copy = np.copy(img)
        roi_copy = np.copy(roi)
        
        cv2.polylines(img_copy, [np.int32(src)], True, (255, 0, 0), 3)
        cv2.polylines(roi_copy, [np.int32(dst)], True, (255, 0, 0), 3)
        
        plot_images([
            (img_copy, 'Original Image'),
            (roi_copy, 'Bird\'s Eye View')
        ])
    return roi, (M, invM)

def get_image(img_path, visualise=False):
    img = mpimg.imread(img_path)
    return preprocess_image(img, visualise=visualise)

#plotting test imgs
for path in test_img_paths[:]:
    get_image(path, visualise=True)
