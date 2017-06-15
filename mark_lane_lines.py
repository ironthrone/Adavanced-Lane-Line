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
perspective_src = np.array([(277, 670), (568, 470), (717, 470), (1030, 670)], np.float32)
perspective_dst = np.array([(340, 720), (340, 0), (940, 0), (940, 720)], np.float32)
M = cv2.getPerspectiveTransform(perspective_src, perspective_dst)
revM = cv2.getPerspectiveTransform(perspective_dst, perspective_src)


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


def lightness_threshold(img, thresh=(100, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    L = hls[:, :, 1]
    binary_output = np.zeros_like(L)
    binary_output[(L >= thresh[0]) & (L <= thresh[1])] = 1
    return binary_output


def color_gradient_filter(img, debug=False):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # filter horizontal line
    sx_binary = abs_sobel_thresh(gray, kernel_size=5, orient='x', thresh_min=20, thresh_max=255)

    directs_binary = dir_threshold(gray, sobel_kernel=15, thresh=(0.4, 1.2))
    l_binary = lightness_threshold(img, (100, 255))

    s_binary = saturation_threshold(img, thresh=(170, 255))

    # add debug ,or io will slow the whole process of marking video
    if debug:
        plt.imsave('output_images/sx_bianry.png', sx_binary, cmap='gray')
        plt.imsave('output_images/directs_bianry.png', directs_binary, cmap='gray')
        plt.imsave('output_images/s_bianry.png', s_binary, cmap='gray')

    combined = np.zeros_like(gray)

    combined[(directs_binary == 1) & ((sx_binary == 1)) | ((s_binary == 1) & (l_binary == 1))] = 1
    return combined


meter_per_pixel_x = 3.7 / 800
meter_per_pixel_y = 30 / 650


class Finder(object):
    def __init__(self):
        self.last_left_line = None
        self.last_right_line = None
        self.frame_count = 0
        self.complete_scan_window = 0
        self.hist_correct = []
        self.binary_warped = None
        self.correct_success = 0
        self.fail_correct = 0
        self.correct_data_count = 5

    def find_lane_line(self, binary_warped):
        self.binary_warped = binary_warped
        self.frame_count += 1
        height, width = binary_warped.shape

        # not 0 pixel postion
        nonzero = binary_warped.nonzero()
        nonzero_x_pos = nonzero[1] #x positon list
        nonzero_y_pos = nonzero[0] # y position list

        left_lane_x = []
        left_lane_y = []
        right_lane_x = []
        right_lane_y = []

        margin = 100

        if not self.is_usable(self.last_left_line, self.last_right_line):
            # slide window
            self.complete_scan_window += 1
            window_count = 10
            window_height = np.int(height / window_count)
            min_count = 50

            histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
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
        else:
            # use last usable line
            left_line_for_nonzero_y = self.last_left_line.coeff[0] * nonzero_y_pos ** 2 + self.last_left_line.coeff[
                                                                                              1] * nonzero_y_pos + \
                                      self.last_left_line.coeff[2]
            left_nonzero_pos_of_pos = ((nonzero_x_pos >= (left_line_for_nonzero_y - margin)) &
                                       (nonzero_x_pos <= (left_line_for_nonzero_y + margin))).nonzero()[0]
            right_line_for_nonzero_y = self.last_right_line.coeff[0] * nonzero_y_pos ** 2 + self.last_right_line.coeff[
                                                                                                1] * nonzero_y_pos + \
                                       self.last_right_line.coeff[2]
            right_nonzero_pos_of_pos = ((nonzero_x_pos >= (right_line_for_nonzero_y - margin)) &
                                        (nonzero_x_pos <= (right_line_for_nonzero_y + margin))).nonzero()[0]
            left_lane_x = nonzero_x_pos[left_nonzero_pos_of_pos]
            left_lane_y = nonzero_y_pos[left_nonzero_pos_of_pos]
            right_lane_x = nonzero_x_pos[right_nonzero_pos_of_pos]
            right_lane_y = nonzero_y_pos[right_nonzero_pos_of_pos]

        # fit y -> x,not x -> y ,because fitted plot's y's range is determined,we can use
        # y's' range get x directly
        left_coeff = np.polyfit(left_lane_y, left_lane_x, 2)
        right_coeff = np.polyfit(right_lane_y, right_lane_x, 2)

        # real world polynomial coeff
        left_coeff_real = np.polyfit(left_lane_y * meter_per_pixel_y, left_lane_x * meter_per_pixel_x, 2)
        right_coeff_real = np.polyfit(right_lane_y * meter_per_pixel_y, right_lane_x * meter_per_pixel_x, 2)

        # calculate polynomial x position list
        ploty = np.linspace(0, height - 1, height)
        left_plotx = np.array(left_coeff[0] * ploty ** 2 + left_coeff[1] * ploty + left_coeff[2], np.int32)
        right_plotx = np.array(right_coeff[0] * ploty ** 2 + right_coeff[1] * ploty + right_coeff[2], np.int32)

        left_line = Line(left_plotx, ploty, left_lane_x, left_lane_y, left_coeff, left_coeff_real)
        right_line = Line(right_plotx, ploty, right_lane_x, right_lane_y, right_coeff, right_coeff_real)


        # check is usable,if not use cache data to determin new line
        if self.is_usable(left_line, right_line):
            self._add_to_hist_lines(left_line, right_line)
        else:
            hist_left_lane_x = []
            hist_right_lane_x = []
            hist_left_lane_y = []
            hist_right_lane_y = []

            for data in self.hist_correct:
                if self.frame_count - data[0] > self.correct_data_count:
                    self.hist_correct.remove(data)
                else:
                    hist_left_lane_x.append(data[1].allx)
                    hist_right_lane_x.append(data[2].allx)
                    hist_left_lane_y.append(data[1].ally)
                    hist_right_lane_y.append(data[2].ally)

            assert len(self.hist_correct) <= self.correct_data_count
            random_bin = len(hist_left_lane_x)
            if len(hist_left_lane_x) > 0:
                hist_left_lane_x = np.array(np.hstack(hist_left_lane_x),np.int32).flatten()
                hist_left_lane_y = np.array(np.hstack(hist_left_lane_y),np.int32).flatten()
                hist_right_lane_x = np.array(np.hstack(hist_right_lane_x),np.int32).flatten()
                hist_right_lane_y = np.array(np.hstack(hist_right_lane_y),np.int32).flatten()


                left_coeff = np.polyfit(hist_left_lane_y, hist_left_lane_x, 2)
                right_coeff = np.polyfit(hist_right_lane_y, hist_right_lane_x, 2)

                ploty = np.linspace(0, height - 1, height)
                left_plotx = np.array(left_coeff[0] * ploty ** 2 + left_coeff[1] * ploty + left_coeff[2], np.int32)
                right_plotx = np.array(right_coeff[0] * ploty ** 2 + right_coeff[1] * ploty + right_coeff[2], np.int32)

                left_coeff_real = np.polyfit(hist_left_lane_y * meter_per_pixel_y, hist_left_lane_x * meter_per_pixel_x, 2)
                right_coeff_real = np.polyfit(hist_right_lane_y * meter_per_pixel_y, hist_right_lane_x * meter_per_pixel_x, 2)

                random_left_count = hist_left_lane_x.shape[0]//random_bin
                left_line = Line(left_plotx, ploty, np.random.choice(hist_left_lane_x,(random_left_count),False)
                                 , np.random.choice(hist_left_lane_y,(random_left_count),False), left_coeff, left_coeff_real)
                random_right_count = hist_right_lane_x.shape[0]//random_bin
                right_line = Line(right_plotx, ploty,
                                  np.random.choice(hist_right_lane_x,(random_right_count),False),
                                  np.random.choice(hist_right_lane_y,(random_right_count),False), right_coeff, right_coeff_real)

                if self.is_usable(left_line, right_line):
                    self._add_to_hist_lines(left_line, right_line)
                    self.correct_success += 1
                else:
                    self.fail_correct += 1


        self.offset = ((left_plotx[-1] + right_plotx[-1]) / 2 - width / 2) * meter_per_pixel_x

        self.last_left_line = left_line
        self.last_right_line = right_line
        return left_line, right_line

    def _add_to_hist_lines(self, left_line, right_line):
        self.hist_correct.append([self.frame_count, left_line, right_line])
        if len(self.hist_correct) > self.correct_data_count:
            self.hist_correct = self.hist_correct[-self.correct_data_count:-1]

    def mark_on_bird_view(self):
        '''
        :return: bird view image with marked lane line zone
        '''
        left, right = self.last_left_line, self.last_right_line
        binary_warped = self.binary_warped

        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        weight_img = np.zeros_like(out_img)

        margin = 100
        left_window_left = np.transpose(np.vstack((left.x - margin, left.y)))
        left_window_right = np.transpose(np.vstack((left.x + margin, left.y)))
        # cv2.fillPoly() need strange pts shape (1,n,2),and dtype need to be np.int32,or checkVector() error is throwed
        left_window_border = np.array([(np.vstack((left_window_left, np.flipud(left_window_right))))], np.int32)

        right_window_left = np.transpose(np.vstack((right.x - margin, right.y)))
        right_window_right = np.transpose(np.vstack((right.x + margin, right.y)))
        right_window_border = np.array([np.vstack((right_window_left, np.flipud(right_window_right)))], np.int32)

        cv2.fillPoly(weight_img, left_window_border, (0, 255, 0))
        cv2.fillPoly(weight_img, right_window_border, (0, 255, 0))

        out_img[left.ally, left.allx] = [255, 0, 0]
        out_img[right.ally, right.allx] = [0, 0, 255]

        # blend two picture,not cover cmopletely
        return cv2.addWeighted(out_img, 1, weight_img, 0.3, 0)

    def prespect_back(self, src, revM):
        '''
        prespect back to original ,and add radius ,offset text on iamge
        :param src:
        :param revM:
        :return:
        '''
        left, right = self.last_left_line, self.last_right_line
        warped = self.binary_warped
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left.x, left.y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right.x, right.y])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        newwarp = cv2.warpPerspective(color_warp, revM, (src.shape[1], src.shape[0]))
        cv2.putText(newwarp, 'Left curvature radius is {}'.format(left.base_cur_r), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2)
        cv2.putText(newwarp, 'Right curvature radius is {}'.format(right.base_cur_r), (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2)

        cv2.putText(newwarp, 'Offset is {}'.format(self.offset), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255),
                    2)
        return cv2.addWeighted(src, 1, newwarp, 0.3, 0)

    def is_usable(self, left, right):
        if not left or not right:
            return False
        radius_right = (0.1 < left.base_cur_r / right.base_cur_r < 10)
        interval = right.x - left.x
        interval_range = np.max(interval) - np.min(interval)
        interval_mae = np.mean(np.abs(interval - np.mean(interval)))
        # print('interval range is:{},interval mae is: {}'.format(interval_range,interval_mae))
        
        interval_right = interval_mae < 15 and interval_range < 50
        return radius_right and interval_right


class Line(object):
    def __init__(self, x, y, allx, ally, coeff, coeff_real, usable=True):
        self.x = x
        self.y = y
        self.coeff = coeff
        self.coeff_real = coeff_real
        self.usable = usable
        self.allx = allx
        self.ally = ally
        self.base_cur_r = self.r_curvature(coeff_real, max(y) * meter_per_pixel_y)

    def r_curvature(self, coeff, y):
        radius = (1 + (2 * coeff[0] * y + coeff[1]) ** 2) ** (3 / 2) / abs(2 * coeff[0])

        return radius


finder = Finder()


def mark_lane_zone(src):
    '''
    complete pipeline for process an image
    :param src:
    :return:
    '''
    camera_mat, dist_coeff = calibrate_camera()
    undist = cv2.undistort(src, camera_mat, dist_coeff)
    binary_filtered = color_gradient_filter(undist)
    warped = cv2.warpPerspective(binary_filtered, M, dsize=src.shape[:2][::-1])
    finder.find_lane_line(warped)
    marked_bird = finder.mark_on_bird_view()
    marked = finder.prespect_back(src, revM)

    # combine marked, filtered,warped,warp and marked image to one image
    filtered_3 = np.dstack((binary_filtered, binary_filtered, binary_filtered)) * 255
    warped_3 = np.dstack((warped, warped, warped)) * 255
    row1 = np.hstack((marked, filtered_3))
    row2 = np.hstack((warped_3, marked_bird))
    result = np.vstack((row1, row2))
    result = cv2.resize(result,(0,0),fx=0.5,fy=0.5)
    return result


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
