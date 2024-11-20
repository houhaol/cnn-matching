import argparse
from asyncio import wrap_future
import cv2
import numpy as np
import imageio
import plotmatch
from lib.cnn_feature import cnn_feature_extract
import matplotlib.pyplot as plt
import time
from skimage import measure
from skimage import transform
from utlis import img_wrap, image_addon, pano_img


pts = "pt6"
target = "/home/houhao/workspace/bim_view_o3d/ros_data/validation2/rs_images/" + pts + ".jpg"
src = (
    "/home/houhao/workspace/bim_view_o3d/ros_data/validation2/BIM_perspective_revised/images_light_on/"
    + pts
    + "_color.png"
)
matches_path = "/home/houhao/workspace/cnn-matching/matches_light_off/" + pts + "_matches.npy"
# matches_path = "/home/houhao/workspace/cnn-matching/matches/" + pts + "_matches.npy"
# read left image
image1 = imageio.imread(target)
image2 = imageio.imread(src)

match_data = (np.load(matches_path, allow_pickle=True)).item()
# save matched data
locations_1_to_use = match_data["keypoint1"]  # dst
locations_2_to_use = match_data["keypoint2"]  # src
inlier_idxs = match_data["matches"]


# Visualize correspondences, and save to file.
# 1 绘制匹配连线
plt.rcParams["savefig.dpi"] = 100  # 图片像素
plt.rcParams["figure.dpi"] = 100  # 分辨率
plt.rcParams["figure.figsize"] = (4.0, 3.0)  # 设置figure_size尺寸
_, ax = plt.subplots()
plotmatch.plot_matches(
    ax,
    image1,
    image2,
    locations_1_to_use,
    locations_2_to_use,
    # np.column_stack((inlier_idxs, inlier_idxs)),
    inlier_idxs,
    plot_matche_points=False,
    matchline=True,
    matchlinewidth=3.0,
    matches_color='r'
)
ax.axis("off")
ax.set_title("")
plt.show()
