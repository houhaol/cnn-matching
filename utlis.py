import cv2
import numpy as np
from skimage import transform

def img_wrap(img1, t_matrix):
    img_wrap = transform.warp(img1, t_matrix)
    return img_wrap

def image_addon(img1, img2, alpha=0.5):
    img_merge = np.uint8(img1 * alpha + img2 * (1 - alpha))
    # img_merge = np.uint8(img1 * alpha + img2 * 0.9)
    # Display images
    return img_merge