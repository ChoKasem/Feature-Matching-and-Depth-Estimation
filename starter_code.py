import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import csv
import os

class FrameCalib:
    """Frame Calibration

    Fields:
        p0-p3: (3, 4) Camera P matrices. Contains extrinsic and intrinsic parameters.
        r0_rect: (3, 3) Rectification matrix
        velo_to_cam: (3, 4) Transformation matrix from velodyne to cam coordinate
            Point_Camera = P_cam * R0_rect * Tr_velo_to_cam * Point_Velodyne
        """

    def __init__(self):
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.r0_rect = []
        self.velo_to_cam = []


def read_frame_calib(calib_file_path):
    """Reads the calibration file for a sample

    Args:
        calib_file_path: calibration file path

    Returns:
        frame_calib: FrameCalib frame calibration
    """

    data_file = open(calib_file_path, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all = []

    for i in range(4):
        p = data[i]
        p = p[1:]
        p = [float(p[i]) for i in range(len(p))]
        p = np.reshape(p, (3, 4))
        p_all.append(p)

    frame_calib = FrameCalib()
    frame_calib.p0 = p_all[0]
    frame_calib.p1 = p_all[1]
    frame_calib.p2 = p_all[2]
    frame_calib.p3 = p_all[3]

    # Read in rectification matrix
    tr_rect = data[4]
    tr_rect = tr_rect[1:]
    tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
    frame_calib.r0_rect = np.reshape(tr_rect, (3, 3))

    # Read in velodyne to cam matrix
    tr_v2c = data[5]
    tr_v2c = tr_v2c[1:]
    tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
    frame_calib.velo_to_cam = np.reshape(tr_v2c, (3, 4))

    return frame_calib


class StereoCalib:
    """Stereo Calibration

    Fields:
        baseline: distance between the two camera centers
        f: focal length
        k: (3, 3) intrinsic calibration matrix
        p: (3, 4) camera projection matrix
        center_u: camera origin u coordinate
        center_v: camera origin v coordinate
        """

    def __init__(self):
        self.baseline = 0.0
        self.f = 0.0
        self.k = []
        self.center_u = 0.0
        self.center_v = 0.0


def krt_from_p(p, fsign=1):
    """Factorize the projection matrix P as P=K*[R;t]
    and enforce the sign of the focal length to be fsign.


    Keyword Arguments:
    ------------------
    p : 3x4 list
        Camera Matrix.

    fsign : int
            Sign of the focal length.


    Returns:
    --------
    k : 3x3 list
        Intrinsic calibration matrix.

    r : 3x3 list
        Extrinsic rotation matrix.

    t : 1x3 list
        Extrinsic translation.
    """
    s = p[0:3, 3]
    q = np.linalg.inv(p[0:3, 0:3])
    u, b = np.linalg.qr(q)
    sgn = np.sign(b[2, 2])
    b = b * sgn
    s = s * sgn

    # If the focal length has wrong sign, change it
    # and change rotation matrix accordingly.
    if fsign * b[0, 0] < 0:
        e = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    if fsign * b[2, 2] < 0:
        e = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    # If u is not a rotation matrix, fix it by flipping the sign.
    if np.linalg.det(u) < 0:
        u = -u
        s = -s

    r = np.matrix.transpose(u)
    t = np.matmul(b, s)
    k = np.linalg.inv(b)
    k = k / k[2, 2]

    # Sanity checks to ensure factorization is correct
    if np.linalg.det(r) < 0:
        print('Warning: R is not a rotation matrix.')

    if k[2, 2] < 0:
        print('Warning: K has a wrong sign.')

    return k, r, t


def get_stereo_calibration(left_cam_mat, right_cam_mat):
    """Extract parameters required to transform disparity image to 3D point
    cloud.

    Keyword Arguments:
    ------------------
    left_cam_mat : 3x4 list
                   Left Camera Matrix.

    right_cam_mat : 3x4 list
                   Right Camera Matrix.


    Returns:
    --------
    stereo_calibration_info : Instance of StereoCalibrationData class
                              Placeholder for stereo calibration parameters.
    """

    stereo_calib = StereoCalib()
    k_left, r_left, t_left = krt_from_p(left_cam_mat)
    _, _, t_right = krt_from_p(right_cam_mat)

    stereo_calib.baseline = abs(t_left[0] - t_right[0])
    stereo_calib.f = k_left[0, 0]
    stereo_calib.k = k_left
    stereo_calib.center_u = k_left[0, 2]
    stereo_calib.center_v = k_left[1, 2]

    return stereo_calib


## Input
left_image_dir = os.path.abspath('./test/left')
right_image_dir = os.path.abspath('./test/right')
calib_dir = os.path.abspath('./test/calib')
sample_list = ['000001', '000002', '000003', '000004','000005', '000006', '000007', '000008', '000009', '000010']
test_list = ['000011','000012','000013','000014','000015']
depth_map_dir = os.path.abspath('./training/gt_depth_map')
## Output
output_file = open("P3_result.txt", "a")
output_file.truncate(0)


## Main

RMSE_sum = 0

for sample_name in test_list:
    left_image_path = left_image_dir +'/' + sample_name + '.png'
    right_image_path = right_image_dir +'/' + sample_name + '.png'
    # depth_map_path = depth_map_dir + '/' + sample_name + '.png'

    img_left = cv.imread(left_image_path, 0)
    img_right = cv.imread(right_image_path, 0)
    # depth_image = cv.imread(depth_map_path, cv.IMREAD_GRAYSCALE).T

    # TODO: Initialize a feature detector

    # Initialize ORB detector
    orb = cv.ORB_create(nfeatures = 1000)
    # find the keypoints with ORB
    kp_left = orb.detect(img_left,None)
    kp_right = orb.detect(img_right,None)
    # compute the descriptors with ORB
    kp_left, des_left = orb.compute(img_left, kp_left)
    kp_right, des_right = orb.compute(img_right, kp_right)
    # print((des_left[1]))
    # input()
    # TODO: Perform feature matching

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
    # match descriptors
    matches = bf.match(des_left,des_right)
    # match in order of distance
    matches = sorted(matches, key = lambda x:x.distance)

    # TODO: Perform outlier rejection

    # create source and destination point from the matches
    src_pts = np.float32([kp_left[m.queryIdx].pt for m in matches ]).reshape(-1,2)
    dst_pts = np.float32([kp_right[m.trainIdx].pt for m in matches ]).reshape(-1,2)
    # run RANSAC from the src and dst points to find trasnform matrix and the mask telling inliers or outliers
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 1.0, maxIters=5000, confidence=.925)

    # Read calibration
    frame_calib = read_frame_calib(calib_dir + '/' +  sample_name + '.txt')
    stereo_calib = get_stereo_calibration(frame_calib.p2, frame_calib.p3)

    # Find disparity and depth
    pixel_u_list = [] # x pixel on left image
    pixel_v_list = [] # y pixel on left image
    disparity_list = []
    depth_list = []

    # measuring depth estimation performance with RMSE
    n = 0
    square_sum = 0.

    for i, match in enumerate(matches):

        # apply mask (ie outlier rejection)
        if mask[i] == 0:
            continue

        left_u, left_v = kp_left[match.queryIdx].pt
        left_u, left_v = int(round(left_u)), int(round(left_v))
        right_u, right_v = kp_right[match.trainIdx].pt
        right_u, right_v = int(round(right_u)), int(round(right_v))

        # enforce epipolar threshold
        # epipolar_threshold = 0
        # if abs(left_v - right_v) > epipolar_threshold:
        #     continue

        disparity = left_u - right_u
        if disparity == 0:
            continue
        pixel_u_list.append(left_u)
        pixel_v_list.append(left_v)
        disparity_list.append(disparity)
        depth = stereo_calib.f * stereo_calib.baseline / disparity
        depth_list.append(depth)

        # measuring accuracy
        # if depth_image[left_u][left_v] == 0:
        #     continue
        # diff = depth - depth_image[left_u][left_v]
        # square_sum += diff**2
        # print(square_sum)
    # print("Image", sample_name)
    # RMSE = np.sqrt(square_sum/n)
    # print(format(RMSE, ".4f"))
    # RMSE_sum += RMSE
    # print(n)

    # Output
    for u, v, disp, depth in zip(pixel_u_list, pixel_v_list, disparity_list, depth_list):
        line = "{} {:.2f} {:.2f} {:.2f} {:.2f}".format(sample_name, u, v, disp, depth)
        output_file.write(line + '\n')
    
    # Draw matches
    img = cv.drawMatches(img_left, kp_left, img_right, kp_right, matches, None, matchesMask=mask,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img)
    plt.show()
# print(format(RMSE_sum,".4f"))
output_file.close()

