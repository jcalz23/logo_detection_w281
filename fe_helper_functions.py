import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable




########-------- IMAGE PROCESSING --------########

def overlay_bboxes(im, img_bbox_file, color='gray'):
    '''Take image and bbox info from roboflow, overlay bbox rectangle on image'''
    
    # get image shape
    y_shape, x_shape = im.shape[:2]
    im_bboxes = im.copy()

    # read bbox info, format into cv2 format (needs corner coords)
    with open(img_bbox_file) as f:
        rf_output = f.readlines()

    # unpack, format each box from roboflow output to cv2 corner coords
    bboxes = []
    for box in rf_output:
        # unpack, format roboflow line
        box = box.replace('\n', '')
        box = box.split(' ')
        box = list(map(float, box)) #convert all to floats
        label, mid_x_p, mid_y_p, w_box, h_box = box

        # convert to box corners
        middle_x = int(x_shape*mid_x_p)
        middle_y = int(y_shape*mid_y_p)
        l_x = middle_x - int(x_shape*w_box/2)
        r_x = middle_x + int(x_shape*w_box/2)
        b_y = middle_y + int(y_shape*h_box/2)
        t_y = middle_y - int(y_shape*h_box/2)
        draw_rect = cv2.rectangle(im_bboxes, (l_x, t_y), (r_x, b_y), (0,0,255), 2)
        bboxes.append([label, l_x, r_x, t_y, b_y]) # save out the bbox corners
    
    return im_bboxes, bboxes
    



########-------- KEYPOINT DESCRIPTOR FEATURES (SIFT, ORB, ETC.) --------########

# input a grayscale image and draw keypoints
def drawKeyPts(grayImage, keyp):
    
    out_im = (grayImage * 255).astype(np.uint8)
    out_im = np.concatenate((out_im[:,:,np.newaxis], 
                             out_im[:,:,np.newaxis], 
                             out_im[:,:,np.newaxis]), axis=2)
    
    for curKey in keyp:
        x=np.int32(curKey.pt[0])
        y=np.int32(curKey.pt[1])
        size = np.int32(curKey.size)
        cv2.circle(out_im, (x,y), 2, [255, 0, 0], 
                   thickness=1, shift=0)

    return out_im
    

# compute the key points from input grayscale images
def HarrisKeypointDetector(in_image, n=2, w=3, k=0.04, p=0.9, verbose=True):
    
    # STEP 1:
    # compute the points where there are good corners in the image
    # the score for good corners are computed as follows:
    # harrisImage = det(H) - k(trace(H))^2
    # where H = A.T*A seen in the async lecture
    
    # parameters to cv2.cornerHarris are:
    #   grayscale image
    #   n = size of the window to compute the A
    #   w = size of the kernel to compute the gradients
    #   k = value for k in det(H) - k(trace(H))^2
    harrisImage = cv2.cornerHarris(in_image, n, w, k)
    
    # STEP 2:
    # threshold the scores to keep only interesting features
    thresh = (1-p) * harrisImage.max()
    harrisMaxImage = harrisImage > thresh
    
    # STEP 3:
    # get keypoints structure from the detected features
    height, width = in_image.shape[:2]
    features = []
    for y in range(height):
        for x in range(width):
            # do not include if it is not in the good features
            if not harrisMaxImage[y, x]:
                continue

            # fill in the parameters
            f = cv2.KeyPoint()
            f.pt = (x, y)
            f.response = harrisImage[y,x]
            features.append(f)

    if verbose:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,7))
        ax[0].imshow(drawKeyPts(in_image, features)); ax[0].axis('off')
        ax[0].set_title('image')
        har_im = ax[1].imshow(harrisImage, cmap='gray'); ax[1].axis('off')
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(har_im, cax=cax)
        ax[1].set_title('score')
        #plt.suptitle('KEYPOINTS')
        plt.show()
            
    return features


# get simple feature detector with only pixel values around wxw window
def SimpleFeatureDescriptor(grayImage, keypoints, w=5):
    
    # for every keypoint get the pixel values around 5x5 window
    grayImage1 = np.pad(grayImage, [(w,w), (w,w)], mode='constant')
    desc = np.zeros((len(keypoints), w*w), np.float32)

    for i, f in enumerate(keypoints):
        x, y = f.pt
        x, y = np.int32(x), np.int32(y)
        x=x+w
        y=y+w

        each_desc = grayImage1[y - w//2 : y + w//2 + 1, 
                               x - w//2 : x + w//2 + 1].reshape([1, w*w])
        desc[i, :] = each_desc

    return desc


# more robust feature descriptor using ORB, SIFT
def ORB_SIFT_FeatureDescriptor(grayImage, use_orb = True, use_harris=True, nfeatures=10, harris_kp=None):
    # select which feature descriptor method: orb or sift
    if use_orb:
        mod = cv2.ORB_create(nfeatures, edgeThreshold=2) # edgeThresh tells orb not to look within x pixels of border
    else:
        mod = cv2.SIFT_create(nfeatures, edgeThreshold=2)
    
    # select keypoint generator: harris or ORB/SIFT
    if use_harris:
        kp = harris_kp
    else:
        kp = mod.detect(grayImage,None)
    
    # derive feature descriptor
    kp, des = mod.compute(grayImage, kp) # or use orb keypoints

    return des


########-------- COLOR MOMENT FEATURES --------########

def color_moments(img, mom=0):
  copy_img=img.copy()
  copy_img=copy_img.astype(np.float32)

  if mom==0: 
    out=copy_img.mean(axis=(0,1))
  else:
    out=[]
    for i in range(3): # loop through channels
      list_image=list(copy_img[:,:,i].flatten()) # channel color histogram
      mean_i=copy_img[:,:,i].mean() # mean color for channel
      res=[(x - mean_i)**mom for x in list_image] # moment difference measure
      res=sum(res)/len(res) # average moment difference
      res=math.copysign(1, res) * np.power(abs(res), 1/mom)
      out.append(res)

  return(out)

def extract_color_moments(im_bbox):
    bbox_cm_0 = color_moments(im_bbox, mom=0)
    bbox_cm_2 = color_moments(im_bbox, mom=2)
    bbox_cm_3 = color_moments(im_bbox, mom=3)

    return bbox_cm_0, bbox_cm_2, bbox_cm_3


########-------- HU MOMENT FEATURES --------########
# Calculate Hu moments for input img
# pass contrast-normalized bbox image

def hu_moments(img):
  img_edges = cv2.Canny(image=img, threshold1=100, threshold2=200)  
  moments = cv2.moments(img_edges)
  hu_moments = cv2.HuMoments(moments)

  for i in range(0,7):
    hu_moments[i] = -1 * math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]))

  return np.array(hu_moments).T






