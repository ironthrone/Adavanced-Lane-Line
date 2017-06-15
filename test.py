import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from mark_lane_lines import *



def test_undistortion(src,dst):
    img = plt.imread(src)
    mat, dist = calibrate_camera()
    undist = cv2.undistort(img, mat, dist)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title('Src image')
    ax1.imshow(img)
    ax2.set_title('Undistorted image')
    ax2.imshow(undist)
    plt.savefig(dst)
    plt.show()


def test_find_lane():
    '''
    test code for find lane line
    '''
    binary_warped = plt.imread('examples/warped-example.jpg') // 255
    left, right = finder.find_lane_line(binary_warped)

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
    result = cv2.addWeighted(out_img, 1, weight_img, 0.3, 0)

    plt.imshow(result)
    plt.plot(left.x, left.y, color='y')
    plt.plot(right.x, right.y, color='y')
    plt.imsave('output_images/marked_lane_lines.png', result)
    plt.show()



def test_bird_view():
    '''
    test code for perspective
    '''
    # vertex shape is (1,<vertex count>,2)
    src = plt.imread('test_images/straight_lines1.jpg')
    src_vertex = np.int32(perspective_src[np.newaxis])
    mat, dist_coeff = calibrate_camera()
    undist = cv2.undistort(src, mat, dist_coeff)
    warped = cv2.warpPerspective(undist, M, dsize=src.shape[:2][::-1])


    cv2.polylines(src, src_vertex, True, color=(255, 0, 0), thickness=10)
    cv2.polylines(undist, src_vertex, True, color=(255, 0, 0), thickness=10)
    dst_vertex = np.int32(perspective_dst[np.newaxis])
    cv2.polylines(warped, dst_vertex, True, color=(255, 0, 0), thickness=10)

    _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 12))

    ax1.set_title('origin image')
    ax1.imshow(src)
    ax2.set_title('undist image')
    ax2.imshow(undist)
    ax3.set_title('warped image')
    ax3.imshow(warped)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.)
    plt.savefig('output_images/undistort_warped.png')
    plt.show()  # zu se

def test_color_gradient_filter():
    '''
    test code for filter
    '''
    img = plt.imread('test_images/test5.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sx_binary = abs_sobel_thresh(gray, kernel_size=5, orient='x', thresh_min=20, thresh_max=255)

    sy_binary = abs_sobel_thresh(gray, kernel_size=3, orient='y', thresh_min=20, thresh_max=255)

    mags_binary = mag_thresh(gray, sobel_kernel=5, mag_thresh=(30, 150))
    directs_binary = dir_threshold(gray, sobel_kernel=15, thresh=(0.4, 1.2))
    l_binary = lightness_threshold(img,(120,255))

    s_binary = saturation_threshold(img, thresh=(170, 255))

    # add debug ,or io will slow the whole process of marking video
    plt.imsave('output_images/gray.png', gray, cmap='gray')
    plt.imsave('output_images/sx_bianry.png', sx_binary, cmap='gray')
    plt.imsave('output_images/sy_bianry.png', sy_binary, cmap='gray')
    plt.imsave('output_images/directs_bianry.png', directs_binary, cmap='gray')
    plt.imsave('output_images/mags_bianry.png', mags_binary, cmap='gray')
    plt.imsave('output_images/s_bianry.png', s_binary, cmap='gray')
    plt.imsave('output_images/l_bianry.png', l_binary, cmap='gray')

    figure, axes = plt.subplots(5, 2, figsize=(12, 12))
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

    combined[(directs_binary == 1)& ((sx_binary == 1) ) | ((s_binary == 1) & (l_binary == 1))] = 1
    plt.imsave('output_images/combined_bianry.png', combined, cmap='gray')

    sx_sy = np.zeros_like(gray)

    axes[2,1].imshow(combined,cmap='gray')
    sx_sy[((sx_binary == 1) & (sy_binary == 1)) ] = 1
    axes[3,0].imshow(sx_sy,cmap='gray')
    axes[3,1].imshow(gray,cmap='gray')
    axes[4,0].imshow(l_binary,cmap='gray')
    plt.show()


# test_undistortion('camera_cal/calibration3.jpg','output_images/undistorted_chessboard.png')
# test_undistortion('test_images/test2.jpg','output_images/undistorted_test2.png')
test_bird_view()
# test_find_lane()
# test_color_gradient_filter()

# src = plt.imread('test_images/straight_lines1.jpg')
# plt.imshow(src)
# plt.show()
