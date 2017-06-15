
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undist_chessboard]: ./output_images/undistorted_chessboard.png 
[undist_test2]: ./output_images/undistorted_test2.png 
[gray]: ./output_images/gray.png 
[sx_binary]: ./output_images/sx_bianry.png 
[dir_binary]: ./output_images/directs_bianry.png 
[s_binary]: ./output_images/s_bianry.png
[l_binary]: ./output_images/l_bianry.png

[combined]: ./output_images/combined_bianry.png
[bird_view]: ./output_images/undistort_warped.png
[marked]: ./output_images/marked_img.png

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code is in mark_lane_lines.py ,function calibrate_camera(),and the test code is in test.py, test_undistortion()

To get the camera matrix and undistortion coeff,i need a object position in real world and image position in image. I use chessboard as pattern, the image come from project's cal_camera directory,after observed these chessboard image,i decide use 9x6 as the coners.
The object points is produced by hand,assume the chessboard paper is on a wall, z = 0, so i just set x and y, and the chessbord iamge come from the same real world chessboard papter.In code it is `objects`
`obj_points` and `img_points` save all the object points and image  points from all  picture
Then i use `cv2.findChessboardCorners()` find image points for every image, if get the correct result ,add `objects` and `corners` to `obj_points` and `img_points`

Finally, use `cv2.calibrateCamera()` ,i got the camere matrix and distortion coeff, add a `cv2.undistort()` we can get the undistorted image

once the result pass the test,i save the matrix and coeff into disk,avoid repetate calibrate

![][undist]


### Pipeline (single images)
The complete pipeline is located in mark_lane_lines.py mark_lane_zone()

#### 1. Provide an example of a distortion-corrected image.


![][undist_test2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code is in mark_lane_lines.py color_gradient_filter().I use color with gradient together.Main code is here
```
  sx_binary = abs_sobel_thresh(gray, kernel_size=5, orient='x', thresh_min=20, thresh_max=255)

    directs_binary = dir_threshold(gray, sobel_kernel=15, thresh=(0.4, 1.2))
    l_binary = lightness_threshold(img, (100, 255))

    s_binary = saturation_threshold(img, thresh=(170, 255))

```
First i use a x sobel to filter horizontal line
Then i create a sobel direction filter 
![][dir_binary]
Then create a saturation filter ,it helps to get the  `pure color` , lane line is belong to, and especially yellow lane line.Yellow lane line can not be detected well in gradient filter,they have a low gradient compared to ambient road color
![alt text][gray]
x sobel do not recgonize the yellow lane line
![][sx_binary]
saturation detect it 
![][s_binary]
Then i create a lightness filter. It helps me to remove the dark pixel,like the shdow of tree,road edge
![][l_binary]

Then i combine them togher
```
    combined[(directs_binary == 1) & ((sx_binary == 1)) | ((s_binary == 1) & (l_binary == 1))] = 1
```
The final result is this
![][combined]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code is located in mark_lane_lines.py  line13~line16.I collect src point from `output_iamges/straight_lines1.jpg` by hand ,and create the dst points


This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 277, 670      | 340, 720        | 
| 568, 470      | 340, 0      |
| 717, 470     | 940, 0      |
| 1030, 670      | 940, 720        |

after call `cv2.getPrespectiveTransform()`,i got the Matrix

The test code is in test.py test_bird_view().I got a reasonable result
![][bird_view] 

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code is located in mark_lane_lines.py class Finder.find_lane_line().
I use slide window to find the lane line points ,detected lane line points position  are saved into 
```
        left_lane_x = []
        left_lane_y = []
        right_lane_x = []
        right_lane_y = []
```
then i use `np.polyfit()` get a corret polynomail coeff.

After finding first lane liens,i set these lines to FInder,and next time ,i just check last lines is usable.if they are usable, i use these line as the middle line,so i just search around area of these lines to get a resonable result

Error can always happen ,especially when the road is mixed with shadow of tree, edge of road,disappeared lane liens,so i need a method to detect if the detected lines is usable, and if it is not usable ,what i should do 
My check usable code is in Class Finder.is_usable() ,i compare left and right lane line's radius , they should on the same magnitude. I set the mean absolute error of intervel between left and right line need be less 15,and range of the intervel max and min value should be amoung 50. If they not meet these ,they are not usable

Then i save a usable lines list ,the max length is 5. if i checked  unusable lines,i select latest 5 frame's lane lines from this list,and use these lines's lane line point to fit a lane line. The code is here ,and located in Finder.find_lane_lines()
```

if self.is_usable(left_line, right_line):
    self._add_to_hist_lines(left_line, right_line)
else:
    hist_left_lane_x = []
    hist_right_lane_x = []
    hist_left_lane_y = []
    hist_right_lane_y = []

    for data in self.hist_correct:
        if self.frame_count - data[0] > correct_data_count:
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
```

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code to calculate radius is located in Finder find_lane_line() and class Line.I assumed the horizontal and vertical direction's pixel-to-meter  is 3.7/800,30/650
```
meter_per_pixel_x = 3.7 / 800
meter_per_pixel_y = 30 / 650
```
Then i fit a real world lane line polynomial
```python
left_coeff_real = np.polyfit(left_lane_y * meter_per_pixel_y, left_lane_x * meter_per_pixel_x, 2)
right_coeff_real = np.polyfit(right_lane_y * meter_per_pixel_y, right_lane_x * meter_per_pixel_x, 2)
```
Then with the  formula i can get the radius,these code is located in class Line r_curvature()
```python
def r_curvature(self, coeff, y):
    radius = (1 + (2 * coeff[0] * y + coeff[1]) ** 2) ** (3 / 2) / abs(2 * coeff[0])

    return radius
```
It is easy to calculate the offset to middle of lane lines.Assume  the middle of image is the center positon of car,from middle line of detected lane lines substract middle of image,  get the offset in pixel ,then multiply `meter_per_pixel_x`,the radius in meter is got
The code is located in the bottom of Finder.find_lane_line()
```
     self.offset = ((left_plotx[-1] + right_plotx[-1]) / 2 - width / 2) * meter_per_pixel_x
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The Code is located in mark_lane_lines.py class Finder prespect_back()
i just use a revert perspective matrix to switch back
In the final part of mark_lane_lines.py ,there is my code. where i chained all the process together ,and do test on all picture from test_images folder,this is the result:

![][marked]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_marked.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

When develop pipline for image,everything is fine.After process the project video,i got many error detection.Then i deal with it from two part: optimizing color gradient filter , add unusable check and correct
For part one ,i removed y sobel filter, it is use to filter vertical line , this project do not need this function. I observed the location where error happend ,there are many shadow or clear road egde,so i add a lightness filter,use the lightness filter result filter saturaton result's shadow,egde
For part two.i already explained it in the "find lane line" zone.The check and cached lines do much favor.My final video has shown it is improved a lot

But there is still a lot problem.My pipeline does not work on the challenge video , there are many disturb line ,  my color gradient filter is not work well with it .There ara still jitter on the project video,i need to smooth the process.I should extract some image from the challenge video and try to do a good work on these image ,then use it to mark video


