import cv2
import numpy as np


class ptsSelect:
    """docstring for ptsSelect."""

    def __init__(self, img_file):
        super(ptsSelect, self).__init__()
        self.img = cv2.imread(img_file, -1)
        self.saved_pts = []

    def mousePoints(self, event, x, y, flags, params):
        # Left button mouse click event opencv
        if event == cv2.EVENT_LBUTTONDOWN:
            pts = [x, y]
            # cv2.circle(self.img,(x,y),100,(255,0,0),-1)
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            self.saved_pts.append([x, y])
            print(x, " ", y)
            cv2.circle(self.img, (x, y), 2, (0, 0, 255), -1)
            cv2.imshow("image", self.img)

    def process(self):
        # displaying the image
        cv2.imshow("image", self.img)

        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback("image", self.mousePoints)

        # Refreshing window all time
        cv2.waitKey(0)

        # close the window
        cv2.destroyAllWindows()
        return self.saved_pts


def homography(src, dst, pts_src, pts_dst):
    im_src = cv2.imread(src, -1)

    im_dst = cv2.imread(dst, -1)

    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

    alpha = 0.5
    img_merge = np.uint8(im_dst * alpha + im_out * (1 - alpha))
    # cv2.imwrite('tmp.png', img_merge)
    # Display images
    cv2.imshow("tmp.png", img_merge)
    print(h)

    cv2.waitKey(0)


if __name__ == "__main__":
    pts = "pt6"

    # Read source image.
    src_file = (
        "/home/houhao/workspace/bim_view_o3d/ros_data/validation2/BIM_perspective_revised/images_light_on/"
        + pts
        + "_color.png"
    )
    # src_file = "../ros_data/hallway_carto/model_map/E1_08_floor.jpg"
    dst_file = "/home/houhao/workspace/bim_view_o3d/ros_data/validation2/rs_images/" + pts + ".jpg"

    ptsobj1 = ptsSelect(src_file)
    pts_src = ptsobj1.process()
    print(pts_src)

    ptsobj2 = ptsSelect(dst_file)
    print("start2")
    pts_dst = ptsobj2.process()
    print(pts_dst)

    # pts_src = [[74, 334], [109, 328], [111, 0], [130, 284], [256, 291], [132, 5]]
    # pts_dst = [[1, 338], [12, 332], [6, 2], [48, 298], [183, 296], [42, 7]]

    homography(src_file, dst_file, np.asarray(pts_src), np.asarray(pts_dst))
    # Read destination image.
    # im_dst = cv2.imread('book1.jpg')
