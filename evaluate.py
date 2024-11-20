import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import plotmatch
from skimage import measure
from skimage import transform
from lib.cnn_feature import cnn_feature_extract

pts = "pt6"
target = "/home/houhao/workspace/bim_view_o3d/ros_data/validation2/rs_images/" + pts + ".jpg"
src = (
    "/home/houhao/workspace/bim_view_o3d/ros_data/validation2/BIM_perspective_revised/images_light_on/"
    + pts
    + "_color.png"
)
matches_npy = "/home/houhao/workspace/cnn-matching/matches_light_off/" + pts + "_matches.npy"


def distance_between_matches(locations_1_to_use, locations_2_to_use, matches, n=None):
    pts1 = np.float32([locations_1_to_use[m[0]] for m in matches])
    pts2 = np.float32([locations_2_to_use[m[1]] for m in matches])

    # Convert x, y coordinates into complex numbers
    # so that the distances are much easier to compute
    z1 = np.array([[complex(c[0], c[1]) for c in pts1]])
    z2 = np.array([[complex(c[0], c[1]) for c in pts2]])

    # Computes the intradistances between keypoints for each image
    KP_dist1 = abs(z1.T - z1)
    KP_dist2 = abs(z2.T - z2)

    # Distance between featured matched keypoints
    FM_dist = abs(z2 - z1)
    if n is None:
        print("average distance between all good matches {}".format(np.mean(FM_dist)))
    else:
        FM_dist = sorted(FM_dist.flatten())
        print("sorted")
        print("average distance between all good matches {}".format(np.mean(FM_dist[:n])))


def distance_fv(kps1, kps2):
    dist_list = []
    for i in range(kps1.shape[0]):
        dist = np.inf
        Sik = kps1[i]
        for j in range(kps2.shape[0]):
            Tjk = kps2[j]
            dist_tmp = np.linalg.norm(Sik - Tjk)
            if dist_tmp < dist:
                dist = dist_tmp
        dist_list.append(dist)
    final_dist = np.mean(dist_list)
    print("The average descriptor distance between {}".format(final_dist))


def convert_to_kp_obj(kps):
    obj_list = []
    for i in range(kps.shape[0]):
        obj_list.append(cv2.KeyPoint(int(kps[i][0]), int(kps[i][1]), 1))
    return obj_list


def sift(src, target):
    img1 = cv2.imread(src, cv2.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv2.imread(target, cv2.IMREAD_GRAYSCALE)  # trainImage

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kps_left, des_left = sift.detectAndCompute(img1, None)
    kps_right, des_right = sift.detectAndCompute(img2, None)

    # Flann特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=40)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_left, des_right, k=2)

    goodMatch = []
    locations_1_to_use = []
    locations_2_to_use = []

    # 匹配对筛选
    min_dist = 1000
    max_dist = 0
    disdif_avg = 0
    # 统计平均距离差
    for m, n in matches:
        disdif_avg += n.distance - m.distance
    disdif_avg = disdif_avg / len(matches)

    for m, n in matches:
        # 自适应阈值
        if n.distance > m.distance + disdif_avg:
            goodMatch.append(m)
            # p2 = cv2.KeyPoint(kps_right[m.trainIdx][0], kps_right[m.trainIdx][1], 1)
            # p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
            p2 = kps_right[m.trainIdx]
            p1 = kps_left[m.queryIdx]

            locations_1_to_use.append([p1.pt[0], p1.pt[1]])
            locations_2_to_use.append([p2.pt[0], p2.pt[1]])

    print("match num is %d" % len(goodMatch))

    locations_1_to_use = np.array(locations_1_to_use)
    locations_2_to_use = np.array(locations_2_to_use)

    _RESIDUAL_THRESHOLD = 30
    model_ransac, inliers = measure.ransac(
        (locations_1_to_use, locations_2_to_use),
        transform.AffineTransform,
        min_samples=3,
        residual_threshold=_RESIDUAL_THRESHOLD,
        max_trials=1000,
    )

    print("Found %d inliers" % sum(inliers))
    inlier_idxs = np.nonzero(inliers)[0]
    final_matches = np.column_stack((inlier_idxs, inlier_idxs))

    data = {}
    data["keypoint1"] = locations_1_to_use  # dst
    data["keypoint2"] = locations_2_to_use  # src
    data["matches"] = np.column_stack((inlier_idxs, inlier_idxs))
    np.save("./sift/" + pts + "_matches.npy", data)

    # match_dist_list = []
    # for i in range(final_matches.shape[0]):
    #     idx1 = final_matches[i, 0]
    #     idx2 = final_matches[i, 1]
    #     des1 = des_left[idx1]
    #     des2 = des_right[idx2]
    #     match_dist_list.append(np.linalg.norm(des2 - des1))
    # print("The matched descriptor distance for sift is {}".format(np.mean(match_dist_list)))

    # goodMatch = sorted(goodMatch, key=lambda x: x.distance)
    # for gm in range(len(goodMatch)):
    #     m = goodMatch[gm]
    #     des2 = des_right[m.trainIdx]
    #     des1 = des_left[m.queryIdx]
    #     match_dist_list.append(np.linalg.norm(des2 - des1))
    # print("The matched descriptor distance between {}".format(np.mean(match_dist_list)))
    # print("The percentage of matched SIFT features is {}".format((len(goodMatch) / des_right.shape[0]) * 100))

    # distance_fv(des_left, des_right)

    # Visualize correspondences, and save to file.
    # 1 绘制匹配连线
    """ plt.rcParams["savefig.dpi"] = 100  # 图片像素
    plt.rcParams["figure.dpi"] = 100  # 分辨率
    plt.rcParams["figure.figsize"] = (4.0, 3.0)  # 设置figure_size尺寸
    _, ax = plt.subplots()
    plotmatch.plot_matches(
        ax,
        img1,
        img2,
        locations_1_to_use,
        locations_2_to_use,
        np.column_stack((inlier_idxs, inlier_idxs)),
        plot_matche_points=False,
        matchline=True,
        matchlinewidth=0.3,
    )
    ax.axis("off")
    ax.set_title("")
    plt.show() """


def cnn_matching(src, target):
    # read left image
    image1 = imageio.imread(target)
    image2 = imageio.imread(src)

    kps_left, sco_left, des_left = cnn_feature_extract(image1, nfeatures=-1)
    kps_right, sco_right, des_right = cnn_feature_extract(image2, nfeatures=-1)
    kps_left_obj = convert_to_kp_obj(kps_left)
    kps_right_obj = convert_to_kp_obj(kps_right)

    # Flann特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=40)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_left, des_right, k=2)

    goodMatch = []
    locations_1_to_use = []
    locations_2_to_use = []

    # 匹配对筛选
    min_dist = 1000
    max_dist = 0
    disdif_avg = 0
    # 统计平均距离差
    for m, n in matches:
        disdif_avg += n.distance - m.distance
    disdif_avg = disdif_avg / len(matches)

    for m, n in matches:
        # 自适应阈值
        if n.distance > m.distance + disdif_avg:
            goodMatch.append(m)
            p2 = cv2.KeyPoint(kps_right[m.trainIdx][0], kps_right[m.trainIdx][1], 1)
            p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
            locations_1_to_use.append([p1.pt[0], p1.pt[1]])
            locations_2_to_use.append([p2.pt[0], p2.pt[1]])

    print("match num is %d" % len(goodMatch))
    locations_1_to_use = np.array(locations_1_to_use)
    locations_2_to_use = np.array(locations_2_to_use)

    _RESIDUAL_THRESHOLD = 30
    model_ransac, inliers = measure.ransac(
        (locations_1_to_use, locations_2_to_use),
        transform.AffineTransform,
        min_samples=3,
        residual_threshold=_RESIDUAL_THRESHOLD,
        max_trials=1000,
    )

    print("Found %d inliers" % sum(inliers))
    inlier_idxs = np.nonzero(inliers)[0]
    final_matches = np.column_stack((inlier_idxs, inlier_idxs))

    # match_dist_list = []
    # for i in range(final_matches.shape[0]):
    #     idx1 = final_matches[i, 0]
    #     idx2 = final_matches[i, 1]
    #     des1 = des_left[idx1]
    #     des2 = des_right[idx2]
    #     match_dist_list.append(np.linalg.norm(des2 - des1))
    # print("The matched descriptor distance for cnn matching is {}".format(np.mean(match_dist_list)))

    # plot
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
        np.column_stack((inlier_idxs, inlier_idxs)),
        plot_matche_points=False,
        matchline=True,
        matchlinewidth=0.3,
    )
    ax.axis("off")
    ax.set_title("")
    plt.show()


def cnn_match(src, target, matches_npy, n):

    image1 = imageio.imread(target)
    image2 = imageio.imread(src)

    data = (np.load(matches_npy, allow_pickle=True)).item()

    kps_dst = data["keypoint1"]
    kps_src = data["keypoint2"]
    matches = data["matches"]
    print("cnn matches number is {}".format(len(matches)))
    distance_between_matches(kps_dst, kps_src, matches, n)


sift(src, target)
# cnn_match(src, target, matches_npy, n)
# cnn_matching(src, target)
