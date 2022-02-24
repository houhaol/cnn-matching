import cv2
import numpy as np
from skimage import transform

def img_wrap(img, t_matrix):
    img_wrap = transform.warp(img, t_matrix)
    return img_wrap

def image_addon(img1, img2, alpha=0.5):
    img_merge = np.uint8(img1 * alpha + img2 * (1 - alpha))
    # img_merge = np.uint8(img1 * alpha + img2 * 0.9)
    # Display images
    return img_merge

def pano_img(img1, img2, t_matrix):
    dst = cv2.warpPerspective(img1, t_matrix,((img1.shape[1] + img2.shape[1]), img2.shape[0])) #wraped image
    dst[0:img2.shape[0], 0:img2.shape[1]] = img2 #stitched image
    cv2.imwrite('output.jpg',dst)
    # import pdb; pdb.set_trace()