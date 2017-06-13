import cv2
import numpy as np
import matplotlib.image as matimg
import matplotlib.pyplot as plt
import glob
import pickle
import os.path as path

pixel_2_meter_x = 0
cal_result_fname = 'cal_result.p'

# points is selected from 'test_images/straight_lines1.jpg'
src = np.array([(277, 670), (568, 470), (717, 470), (1030, 670)], np.float32)
dst = np.array([(340, 720), (340, 0), (940, 0), (940, 720)], np.float32)
M = cv2.getPerspectiveTransform(src, dst)
revM = cv2.getPerspectiveTransform(dst, src)


def calibrate_camera(debug=False):
    # detect if there is already save a calibration result
    if path.exists(cal_result_fname):
        with open(cal_result_fname, 'rb') as file:
            dict = pickle.load(file)
        return dict['mat'], dict['dist']

    image_files = glob.glob('camera_cal/calibration*.jpg')

    objects = np.zeros((6 * 9, 3), np.float32)
    objects[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    obj_points = []
    img_points = []

    finded_count = 0
    for image_file in image_files:
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6))
        if ret:
            finded_count += 1
            obj_points.append(objects)
            img_points.append(corners)

    if debug:
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(5000)

    img = plt.imread('camera_cal/calibration1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ret, mat, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, gray.shape[::], None, None)
    if ret:
        with open(cal_result_fname, 'wb') as file:
            pickle.dump({'mat': mat, "dist": dist}, file)
        return mat, dist
    else:
        raise ValueError('fail to get distortion coefficience')


def abs_sobel_thresh(gray, kernel_size=3, orient='x', thresh_min=0, thresh_max=255):
    if orient == 'x':
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    elif orient == 'y':
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # 先去绝对值，然后伸缩为0->255
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_x = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))
    binary_output = np.zeros_like(abs_sobel_x)
    binary_output[(abs_sobel_x > thresh_min) & (abs_sobel_x < thresh_max)] = 1
    return binary_output


def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    binary_output = np.zeros_like(magnitude)
    binary_output[(magnitude > mag_thresh[0]) & (magnitude < mag_thresh[1])] = 1
    return binary_output


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # 不指定ksize，内核大小不生效
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_orientataion = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    # abs_orientataion = np.abs(orientation)
    binary_output = np.zeros_like(abs_orientataion)
    binary_output[(abs_orientataion >= thresh[0]) & (abs_orientataion <= thresh[1])] = 1
    return binary_output


def saturation_threshold(img, thresh=(100, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]
    binary_output = np.zeros_like(S)
    binary_output[(S >= thresh[0]) & (S <= thresh[1])] = 1
    return binary_output


def color_gradient_filter(img, debug=False):
    # filter horizontal line
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sx_binary = abs_sobel_thresh(gray, kernel_size=5, orient='x', thresh_min=30, thresh_max=100)
    plt.imsave('output_images/sx_bianry.png', sx_binary, cmap='gray')
    sy_binary = abs_sobel_thresh(gray, kernel_size=5, orient='y', thresh_min=30, thresh_max=100)
    plt.imsave('output_images/sy_bianry.png', sy_binary, cmap='gray')

    mags_binary = mag_thresh(gray, sobel_kernel=15, mag_thresh=(30, 100))
    plt.imsave('output_images/mags_bianry.png', mags_binary, cmap='gray')
    directs_binary = dir_threshold(gray, sobel_kernel=9, thresh=(0.7, 1.4))
    plt.imsave('output_images/directs_bianry.png', directs_binary, cmap='gray')

    s_binary = saturation_threshold(img, thresh=(120, 250))
    plt.imsave('output_images/s_bianry.png', s_binary, cmap='gray')

    if debug:
        figure, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes[0, 0].set_title('Sobel x')
        axes[0, 0].imshow(sx_binary, cmap='gray')
        axes[0, 1].set_title('Sobel y')
        axes[0, 1].imshow(sy_binary, cmap='gray')
        axes[1, 0].set_title('Sobel magnitude')
        axes[1, 0].imshow(mags_binary, cmap='gray')
        axes[1, 1].set_title('Sobel direction')
        axes[1, 1].imshow(directs_binary, cmap='gray')
        axes[2, 0].set_title('Saturation')
        axes[2, 0].imshow(s_binary, cmap='gray')
        plt.subplots_adjust(left=0., right=1., top=0.95, bottom=0.)

    combined = np.zeros_like(gray)

    combined[(mags_binary == 1) & (directs_binary == 1) & (sx_binary == 1) & (sy_binary == 1) | (s_binary == 1)] = 1
    return combined


def find_lane_line(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    height, width = binary_warped.shape

    window_count = 10
    window_height = np.int(height / window_count)

    nonzero = binary_warped.nonzero()

    nonzero_x_pos = nonzero[1]
    nonzero_y_pos = nonzero[0]
    left_lane_x = []
    left_lane_y = []
    right_lane_x = []
    right_lane_y = []
    margin = 100

    min_count = 50

    middle = np.int(width / 2)
    left_base_index = np.argmax(histogram[0:middle])
    right_base_index = np.argmax(histogram[middle:width]) + middle

    corrected_left_index = left_base_index
    corrected_right_index = right_base_index

    # scan from bottom to up
    for i in range(window_count):
        window_bottom = height - (i + 1) * window_height
        window_top = height - i * window_height
        left_min_index = corrected_left_index - margin
        left_max_index = corrected_left_index + margin
        right_min_index = corrected_right_index - margin
        right_max_index = corrected_right_index + margin
        #
        # cv2.rectangle(out_img, (left_min_index, window_bottom), (left_max_index, window_top), color=(255, 0, 0),
        #               thickness=2)
        # cv2.rectangle(out_img, (right_min_index, window_bottom), (right_max_index, window_top), color=(255, 0, 0),
        #               thickness=2)

        left_nonzero_pos_of_pos = ((nonzero_y_pos >= window_bottom) & (nonzero_y_pos <= window_top) \
                                   & (left_min_index <= nonzero_x_pos) \
                                   & (nonzero_x_pos <= left_max_index)).nonzero()[0]

        left_nonzero_x_pos = nonzero_x_pos[left_nonzero_pos_of_pos]
        left_nonzero_y_pos = nonzero_y_pos[left_nonzero_pos_of_pos]
        right_nonzero_pos_of_pos = ((nonzero_y_pos >= window_bottom) & (nonzero_y_pos <= window_top) \
                                    & (right_min_index <= nonzero_x_pos) & \
                                    (nonzero_x_pos <= right_max_index)).nonzero()[0]
        right_nonzero_x_pos = nonzero_x_pos[right_nonzero_pos_of_pos]
        right_nonzero_y_pos = nonzero_y_pos[right_nonzero_pos_of_pos]

        left_lane_x.append(left_nonzero_x_pos)
        left_lane_y.append(left_nonzero_y_pos)

        right_lane_x.append(right_nonzero_x_pos)
        right_lane_y.append(right_nonzero_y_pos)

        if (len(left_nonzero_pos_of_pos) >= min_count):
            corrected_left_index = np.int(np.mean((left_nonzero_x_pos)))
        if (len(right_nonzero_pos_of_pos) >= min_count):
            corrected_right_index = np.int(np.mean((right_nonzero_x_pos)))

    left_lane_x = np.concatenate(left_lane_x)
    left_lane_y = np.concatenate(left_lane_y)
    right_lane_x = np.concatenate(right_lane_x)
    right_lane_y = np.concatenate(right_lane_y)

    # fit y -> x,not x -> y ,because fitted plot's y's range is determined,we can use
    # y's' range get x directly
    left_coee = np.polyfit(left_lane_y, left_lane_x, 2)
    right_coee = np.polyfit(right_lane_y, right_lane_x, 2)

    ploty = np.linspace(0, height - 1, height)
    left_plotx = left_coee[0] * ploty ** 2 + left_coee[1] * ploty + left_coee[2]
    right_plotx = right_coee[0] * ploty ** 2 + right_coee[1] * ploty + right_coee[2]
    left_line = Line(left_plotx, ploty, left_coee)
    right_line = Line(right_plotx, ploty, right_coee)
    # declare global
    global pixel_2_meter_x
    pixel_2_meter_x = 3.7 / np.mean(right_plotx - left_plotx)

    return left_line, right_line


class Line(object):
    def __init__(self, x, y, coee):
        self.x = x
        self.y = y
        self.coee = coee
        self.base_cur_r = self.r_curvature(coee, max(y))

    def r_curvature(self, coee, y):
        radiu_in_pixel = (1 + (2 * coee[0] * y + coee[1]) ** 2) ** (3 / 2) / abs(2 * coee[0])

        return radiu_in_pixel * pixel_2_meter_x


def draw_lane_zone_back(src, warped, revM, left, right):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left.x, left.y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right.x, right.y])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    newwarp = cv2.warpPerspective(color_warp, revM, (src.shape[1], src.shape[0]))
    return cv2.addWeighted(src, 1, newwarp, 0.3, 0)


def mark_lane_zone(src):
    camera_mat, dist_coeff = calibrate_camera()
    undist = cv2.undistort(src, camera_mat, dist_coeff)
    binary_filtered = color_gradient_filter(undist)
    warped = cv2.warpPerspective(binary_filtered, M, dsize=src.shape[:2][::-1])
    left, right = find_lane_line(warped)
    marked = draw_lane_zone_back(src, warped, revM, left, right)
    return marked


if __name__ == '__main__':
    fnames = glob.glob('test_images/test*.jpg')
    srcs = []

    for f in fnames:
        src = plt.imread(f)
        marked = mark_lane_zone(src)
        srcs.append(src)
        srcs.append(marked)

    size = len(srcs)

    _, axes = plt.subplots(3, 4, figsize=(24, 15))
    for i in range(3):
        axes[i, 0].imshow(srcs[i * 4])
        axes[i, 1].imshow(srcs[i * 4 + 1])
        axes[i, 2].imshow(srcs[i * 4 + 2])
        axes[i, 3].imshow(srcs[i * 4 + 3])

    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.)
    plt.savefig('output_images/marked_img.png')
    plt.show()
