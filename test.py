import numpy as np
import cv2
import matplotlib.pyplot as plt
import math




'''
test code for find lane line
'''

# binary_warped = plt.imread('examples/warped-example.jpg') // 255


# weight_img = np.zeros_like(out_img)
#
# left_window_left = np.transpose(np.vstack((left_plotx - margin, ploty)))
# left_window_right = np.transpose(np.vstack((left_plotx + margin, ploty)))
# # cv2.fillPoly() need strange pts shape (1,n,2),and dtype need to be np.int32,or checkVector() error is throwed
# left_window_border = np.array([(np.vstack((left_window_left, np.flipud(left_window_right))))], np.int32)
#
# right_window_left = np.transpose(np.vstack((right_plotx - margin, ploty)))
# right_window_right = np.transpose(np.vstack((right_plotx + margin, ploty)))
# right_window_border = np.array([np.vstack((right_window_left, np.flipud(right_window_right)))], np.int32)
#
# cv2.fillPoly(weight_img, left_window_border, (0, 255, 0))
# cv2.fillPoly(weight_img, right_window_border, (0, 255, 0))
#
# out_img[left_lane_y, left_lane_x] = [255, 0, 0]
# out_img[right_lane_y, right_lane_x] = [0, 0, 255]
#
# # blend two picture,not cover cmopletely
# result = cv2.addWeighted(out_img, 1, weight_img, 0.3, 0)
#
# plt.imshow(result)
# plt.plot(left_plotx, ploty, color='y')
# plt.plot(right_plotx, ploty, color='y')
# plt.imsave('output_images/marked_lane_lines.png', result)
# plt.show()


'''
test code for perspective
'''

# def undistorion(img):
#     undist = cv2.undistort(img,mat,distCoeffs=dist)
#     return undist

# mat,dist = calibrate()
# undist = cv2.undistort(img,mat,distCoeffs=dist)
# if debug:
#     _,(ax1,ax2) = plt.subplots(1,2,figsize=(12,9))
#     ax1.imshow(img)
#     ax2.imshow(undist)
#     plt.show()

# vertex shape is (1,<vertex count>,2)
# src_vertex = np.int32(src[np.newaxis])
# cv2.polylines(undist, src_vertex, True, color=(255, 0, 0), thickness=10)
# dst_vertex = np.int32(dst[np.newaxis])
# cv2.polylines(warped, dst_vertex, True, color=(255, 0, 0), thickness=10)
#
# _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 12))
#
# ax1.set_title('origin image')
# ax1.imshow(img)
# ax2.set_title('undist image')
# ax2.imshow(undist)
# ax3.set_title('warped image')
# ax3.imshow(warped)
# plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.)
# plt.savefig('output_images/undistort_warped.png')
# plt.show()  # zu se

# img = plt.imread('test_images/straight_lines1.jpg')
# mat, dist = calibrate()
# undist = cv2.undistort(img, mat, dist)


'''
test code for sobel
'''
# img = matimg.imread('sobel_train.png')
#
# # filter horizontal line
# x_sobel = abs_sobel_thresh(img=img, kernel_size=5,orient='x', thresh_min=30,thresh_max=100)
#
# y_sobel = abs_sobel_thresh(img=img, kernel_size=5,orient='y', thresh_min=30,thresh_max=100)
#
# mag_sobel = mag_thresh(img,sobel_kernel=15,mag_thresh=(30,100))
# direct_sobel = dir_threshold(img,sobel_kernel=15,thresh=(0.7,1.4))
#
#
# combined = np.zeros_like(img)
# combined[(mag_sobel==1) & (direct_sobel==1) | ((x_sobel == 1) &(y_sobel == 1))] = 1
# # pyplot.subplot(2,2,1)
# pyplot.imshow(combined)
# pyplot.show()