import pdb
import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import plotmatch
from skimage import measure
from skimage import transform
import json


def json_load(filename):
    with open(filename, "r") as f:
        data = json.loads(f.read())
    return data


def plot_helper(image1, image2, locations_1_to_use, locations_2_to_use, matches):
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
        matches,
        plot_matche_points=False,
        matchline=True,
        matchlinewidth=0.3,
    )
    ax.axis("off")
    ax.set_title("")
    plt.show()


def plot_mma(err):
    plt_lim = [1, 15]
    plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)

    plt.subplot()
    plt.plot(plt_rng, [err[thr] for thr in plt_rng], color="red", linewidth=3, label=pts)
    plt.title("Mean reproject error given threshold")
    plt.xlim(plt_lim)
    plt.xticks(plt_rng)
    plt.ylabel("MMA")
    plt.ylim([0, 1])
    plt.grid()
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.legend()
    plt.show()


def evaluate_helper(kps1, kps2, matches, homography):
    lim = [1, 15]
    rng = np.arange(lim[0], lim[1] + 1)
    err = {thr: 0 for thr in rng}

    pos_a = kps1[matches[:, 0], :2]
    pos_a_h = np.concatenate([pos_a, np.ones([matches.shape[0], 1])], axis=1)
    pos_b_proj_h = np.transpose(np.dot(np.linalg.inv(homography), np.transpose(pos_a_h)))
    pos_b_proj = pos_b_proj_h[:, :2] / pos_b_proj_h[:, 2:]

    pos_b = kps2[matches[:, 1], :2]

    dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))
    # dist = np.linalg.norm(pos_b - pos_b_proj)
    if dist.shape[0] == 0:
        dist = np.array([float("inf")])
    print("The reprojection error is {}".format(dist))
    print("The average reprojection error is {}".format(np.mean(dist)))
    for thr in rng:
        err[thr] += np.mean(dist <= thr)
    return err


def evaluate(target, src, matches_npy, homography):
    image1 = imageio.imread(target)
    image2 = imageio.imread(src)

    data = (np.load(matches_npy, allow_pickle=True)).item()
    print(data["matches"].shape[0])

    kps_left = data["keypoint1"]
    kps_right = data["keypoint2"]
    matches = data["matches"]
    error = evaluate_helper(kps_left, kps_right, matches, homography)
    print("The MMA is {}".format(error))
    # plot_mma(error)
    if 1:
        plot_helper(image1, image2, kps_left, kps_right, matches)


pts = "pt6"
target = "/home/houhao/workspace/bim_view_o3d/ros_data/validation2/rs_images/" + pts + ".jpg"
src = (
    "/home/houhao/workspace/bim_view_o3d/ros_data/validation2/BIM_perspective_revised/images_light_on/"
    + pts
    + "_color.png"
)

cnn_matches = "/home/houhao/workspace/cnn-matching/matches_light_off/" + pts + "_matches.npy"
sift_matches = "/home/houhao/workspace/cnn-matching/sift_ransac/" + pts + "_matches.npy"


Homograph_path = "/home/houhao/workspace/cnn-matching/homography.json"
homographies = json_load(Homograph_path)


homography = np.asarray(homographies[pts])

print("D2-Net matching performance")
cnn_ev = evaluate(target, src, cnn_matches, homography)
print("-" * 100)
print("SIFT matching performance")
sift_ev = evaluate(target, src, sift_matches, homography)
