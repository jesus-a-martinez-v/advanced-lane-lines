# Advanced Lane Finding

This project was developed as part of the Computer Vision module of the amazing Self-Driving Car Engineer Nanodegree
program offered by
 [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

This README is structured in a Q&A fashion, where each section is comprised of several questions or issues that we had to
tackle in order to meet the minimum requirements stated in [this rubric](https://review.udacity.com/#!/rubrics/571/view).

## Writeup / README

#### I. _"Provide a Writeup / README that includes all the rubric points and how you addressed each one."_

This is the README. Keep reading to find out how we applied several cool computer vision techniques to solve the problem at hand ;)

## Camera Calibration

#### I. _"Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image."_

To calibrate the camera, we used a series of chess board images with dimensions 9x6 (this is, 9 corners per row, and 6 corners per column, where a corner is a point in common for two black squares and two white squares). We used the following function:

```
def calibrate_camera(directory_path='./camera_cal', chessboard_shape=(9, 6), save_location='./cal_pickle.p'):
    """
    Takes a series of check board images and uses them to find the appropriate calibration parameters.
    :param directory_path: Directory locations where the calibration images lie.
    :param chessboard_shape: Dimensions (in squares) of the chess boards.
    :param save_location: Path of the file where the calibration parameters will be stored (pickled)
    :return:
    """
    # Let's prepare object points, like (0, 0, 0)... (8, 5, 0)
    columns, rows = chessboard_shape
    op = np.zeros((rows * columns, 3), np.float32)
    op[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

    # Lists to store object points and image points
    obj_points = []  # Points in 3D, real world images.
    img_points = []  # Points in 2D images.

    image_name_pattern = directory_path + "/calibration*.jpg"
    for img_index, img_name in enumerate(glob.glob(image_name_pattern)):
        # Load the image and transform it into gray scale
        img = cv2.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Try to find the corners in the image
        ret, corners = cv2.findChessboardCorners(gray, chessboard_shape, None)

        # If the corners were found, append the object and image points to result
        # and print the corners in the original image.
        if ret:
            obj_points.append(op)
            img_points.append(corners)

            cv2.drawChessboardCorners(img, chessboard_shape, corners, ret)
            cv2.imwrite(directory_path + '/corners' + str(img_index) + ".jpg", img)

    # We load the first image just to determine its dimensions.
    img = cv2.imread(directory_path + '/calibration2.jpg')
    image_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)

    cv2.imwrite(directory_path + '/calibration2_corrected.jpg', cv2.undistort(img, mtx, dist, None, mtx))

    if save_location:
        pickle.dump({'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}, open(save_location, 'wb'))

    return mtx, dist, rvecs, tvecs
```

The overall process is as follows:

* Compute the real world object points (stored in `op`). These are points in (X, Y, Z) space, where Z = 0 because each chess board is a flat image and all share the same dimensions, so we replicate such points for each image.
* For each calibration image in the directory `camera_cal` we load the image, turn it into gray scale and try to find its corners.
* If we succeed, then draw these corners over the original image and save it. Then we append the `corners` to the `img_points` collection, an a new copy of `op` to the `obj_points` collection.
* We use the `img_points` and `obj_points` to obtain the calibration matrix and the distortion coefficients, using `cv2.calibrateCamera()`
* Finally, we save these parameters in a pickled format, and return them.

Here's an original, distorted, chess board image:
![alt-tag](https://github.com/jesus-a-martinez-v/advanced-lane-lines/blob/master/camera_cal/calibration2.jpg)

Here's the same image with corners drawn on it:
![alt-tag](https://github.com/jesus-a-martinez-v/advanced-lane-lines/blob/master/camera_cal/corners0.jpg)

Finally, here's the undistorted image:
![alt-tag](https://github.com/jesus-a-martinez-v/advanced-lane-lines/blob/master/camera_cal/calibration2_corrected.jpg)

## Pipeline (test images)

#### I. _"Provide an example of a distortion-corrected image."_

Using the camera matrix and the distortion coefficients calculated in the last step, we applied the `cv2.undistort()` function to each image in the `/test_images` directory.

Here's an original test image:
![alt-tag](https://github.com/jesus-a-martinez-v/advanced-lane-lines/blob/master/test_images/test4.jpg)

And here's the undistorted version:
![alt-tag](https://github.com/jesus-a-martinez-v/advanced-lane-lines/blob/master/output_images/undistorted_test4.jpg)

#### II. _"Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result."_

We used two thresholding methods:

* Gradient thresholding. For each image we calculated the gradients in the X and Y directions. For that matter we used the following function:

```
def abs_sobel_threshold(img, orientation='x', sobel_kernel=3, threshold=(0, 255)):
    """
    Applies the sobel operation to the input image to find the gradients in a particular orientation (X or Y).
    :param img: Image whose gradients will be calculated.
    :param orientation: Orientation of the sobel. X for horizontal, Y for vertical.
    :param sobel_kernel: Kernel size. Must be an odd number greater than 3. The greater the number, the smoother the result.
    :param threshold: Pixels will be considered in the resulting mask if they are within these boundaries.
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Heads up! Change to RGB if using mpimg.imread

    if orientation.lower() == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)

    # Apply threshold
    lower, upper = threshold
    selector = (scaled_sobel >= lower) & (scaled_sobel <= upper)
    binary_output[selector] = 1

    return binary_output
```
* Color thresholding. For each image we used the H channel (from the HLS color space) and the V channel (from the HSV color space) to pick the lane lines. For that matter we used the following function:
```
def color_threshold(img, s_threshold=(0, 255), v_threshold=(0, 255)):
    """
    Thresholds an image using the H channel (of the HLS color space) and the V channel (of the HSV color space).
    :param img: Input image.
    :param s_threshold: Range of accepted values for the H channel.
    :param v_threshold: Range of accepted values fir the V channel.
    :return: A color mask with the same dimensions as the input image but with only one channel.
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s = hls[:, :, 2]
    s_binary = np.zeros_like(s)
    lower_s, upper_s = s_threshold
    s_binary[(s >= lower_s) & (s <= upper_s)] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    v_binary = np.zeros_like(v)
    lower_v, upper_v = v_threshold
    v_binary[(v >= lower_v) & (v <= upper_v)] = 1

    binary_output = np.zeros_like(s)
    binary_output[(s_binary == 1) & (v_binary == 1)] = 1
    return binary_output
```

In our pipeline, we first create a black canvas with one channel and the same width and height than our original image. Then we compute three different masks:

* Gradient in the X orientation.
* Gradient in the Y orientation.
* Color mask.

Finally, we turn white those pixels that are captured either by the color mask or by **both** gradient masks. Here's the code:

```
    processed_image = np.zeros_like(input_image[:, :, 0])  # Just a black canvas
    x_gradient_mask = abs_sobel_threshold(input_image, orientation='x', threshold=(12, 255), sobel_kernel=3)
    y_gradient_mask = abs_sobel_threshold(input_image, orientation='y', threshold=(25, 255), sobel_kernel=3)
    color_binary_mask = color_threshold(input_image, s_threshold=(100, 255), v_threshold=(50, 255))
    selection = (x_gradient_mask == 1) & (y_gradient_mask == 1) | (color_binary_mask == 1)
    processed_image[selection] = 255
```

The threshold ranges for each mask were the result of many trial and error iterations.

Here's an original image:
![alt-tag](https://github.com/jesus-a-martinez-v/advanced-lane-lines/blob/master/output_images/undistorted_test4.jpg)

And here's the same image after the thresholding operation:
![alt-tag](https://github.com/jesus-a-martinez-v/advanced-lane-lines/blob/master/output_images/raw_test4.jpg)

#### III. _"Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image."_

Given that our goal is to accurately identify the lane lines, as well as their curvature and the position of the car in the lane, we need a more detailed view
of these lines. What's wrong with the original perspective of the images? Well, as the "depth" increases (this is, the farther points in the image from the camera perspective) we lose valuable information about the lane lines shape. For example, in the image below we can see how the left lane looks straighter than the right lane:
![alt tag](https://github.com/jesus-a-martinez-v/advanced-lane-lines/blob/master/output_images/undistorted_test5.jpg)

But after we transform our point of view to a top perspective, we can see with more detail that both lane lines are curving in about the same degree:
![alt tag](https://github.com/jesus-a-martinez-v/advanced-lane-lines/blob/master/output_images/warped_test5.jpg)

What steps did we take to perform this perspective transform? Here's the outline:

* First, we selected four points that form a trapezoid form in the original image: `[[ 589.  446.] [ 691.  446.] [ 973.  677.] [ 307.  677.]]`
* Second, we selected four destination points that would warp the trapezoidal from into a rectangular one, focusing on a birds-eye-view of the lanes: `[[ 320.    0.] [ 960.    0.] [ 960.  720.] [ 320.  720.]]` 
* Using these two sets of points, we calculated both the perspective transform matrix and its inverse using the `cv2.getPerspectiveTransform()` function.

Here's the function that performs this task:
```
def calculate_perspective_transform_parameters():
    """
    Calculates the parameters needed to transform the perspective an image to a birds-eye view.
    :return: 1. The transformation matrix needed to pass from the original perspective to the top perspective.
             2. The inverse of the transformation matrix, which is useful when we need to revert the perspective transformation.
             3. The source points, which describe a trapezoid in the original image.
             4. The destination points, which describe a rectangle in the transformed image.
    """
    src = np.float32([[589,  446], [691,  446], [973,  677], [307,  677]])
    dst = np.float32([[320, 0], [960, 0], [960, 720], [320, 720]])

    transform_matrix = cv2.getPerspectiveTransform(src, dst)
    inverse_transform_matrix = cv2.getPerspectiveTransform(dst, src)

    return transform_matrix, inverse_transform_matrix, src, dst
```

Then, to from one perspective to another, we use the `cv2.warpPerspective()` function. For instance: `warped = cv2.warpPerspective(processed_image, perspective_matrix, original_image_size, flags=cv2.INTER_LINEAR)`

#### IV. _"Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial"_

After reverting distortion, applying the thresholding techniques described above, and performing a bird-eye perspective transformation, we end up with an image like this:
![alt-tag](https://github.com/jesus-a-martinez-v/advanced-lane-lines/blob/master/output_images/raw_warped_test3.jpg)

The next step is to decide which pixels constitute a line. For that matter we implemented a **Sliding Windows** technique, using 1D convolutions instead of histograms. This decision was highly inspired by the amazing Q&A session held by Udacity in [this video](https://www.youtube.com/watch?v=vWY8YUayf9Q&feature=youtu.be). Given that convolutions are a pretty tricky concept to grasp, here are some useful links that might shed some light over the subject:

* [Wikipedia convolution article](https://en.wikipedia.org/wiki/Convolution)
* [NumPy convolve documentation](https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html)
* [Khan Academy lessons](https://www.khanacademy.org/math/differential-equations/laplace-transform/convolution-integral/v/introduction-to-the-convolution)

Basically, the advantage of using a convolution is that provides a smoother result by taking into account a larger area of the image.

In order to keep track of the lines found in each frame, we implemented an object to store these results, which con be found in `history_keeper.py`. All the magic happens in the `find_window_centroids()` method.

After finding the centers of the windows on each line, we fitted a second order polynomial to each line. Here's the code that does the job:

```
    ###########################################################
    # STEP 4: Find the centroids of each window in each line. #
    ###########################################################
    window_centroids = curve_centers.find_window_centroids(warped)

    # points used to find the left and right lane lines
    right_x_points = []
    left_x_points = []

    number_of_levels = len(window_centroids)
    for level in range(number_of_levels):
        left, right = window_centroids[level]  # Current centroid
        left_x_points.append(left)
        right_x_points.append(right)

    ##########################################
    # STEP 5: Fit a polynomial to each line. #
    ##########################################
    y_values = range(200, original_image_height)
    res_y_vals = np.arange(original_image_height - (curve_centers.window_height / 2), 0, -curve_centers.window_height)

    left_polynomial_coefficients = np.polyfit(res_y_vals, left_x_points, 2)
    left_xs = evaluate_polynomial(y_values, left_polynomial_coefficients)

    right_polynomial_coefficients = np.polyfit(res_y_vals, right_x_points, 2)
    right_xs = evaluate_polynomial(y_values, right_polynomial_coefficients)
```


And the definition of `evaluate_polynomial()` is:
```
def evaluate_polynomial(ys, coefficients):
    """
    Evaluates a 2nd degree polynomial over a collection of values.
    :param ys: Input values
    :param coefficients: Polynomial coefficients.
    :return: Output values (xs)
    """
    # Unpack coefficients
    a, b, c = coefficients[0], coefficients[1], coefficients[2]

    return np.array(a * np.power(ys, 2) + b * ys + c, np.int32)
```

Finally, we use these coefficients to determine the vertices of a polygon for each line (yes, a polygon because we want our lines to be solid) with this function:
```
def get_line_polygon(xs, ys, thickness=20):
    """
    Takes the X and Y values that describe the points that comprise a line, and returns the points that describe a
     polygon with the provided thickness (in pixels).
    :param xs: X coordinates.
    :param ys: Y coordinates.
    :param thickness: Width of the resulting polygon in pixels.
    :return:
    """
    all_xs = np.concatenate((xs - thickness / 2, xs[::-1] + thickness / 2), axis=0)
    all_ys = np.concatenate((ys, ys[::-1]), axis=0)

    polygon_points = np.array([(x, y) for x, y in zip(all_xs, all_ys)], np.int32)

    return polygon_points
```

Drawing each line onto the original image (after reverting the perspective transform, clearly) we get something like this:
[alt tag](https://github.com/jesus-a-martinez-v/advanced-lane-lines/blob/master/output_images/only_lines_test3.jpg)

#### V. _"Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center._

To calculate the radius of the curvature we used as a reference the left line. Given that its values are measured in pixels we must had to transform them into meters. Then we fitted a second order polynomial, so we could apply the technique showed in this [great link](http://www.intmath.com/applications-differentiation/8-radius-curvature.php).

Here's the code:
```
    ##############################################
    # STEP 6: Calculate radius of the curvature. #
    ##############################################
    meters_per_pixel_y_axis = 10 / 720
    meters_per_pixel_x_axis = 4 / 384
    res_y_vals_in_meters = np.array(res_y_vals, np.float32) * meters_per_pixel_y_axis
    left_x_in_meters = np.array(left_x_points, np.float32) * meters_per_pixel_x_axis
    curvature_radius_polynomial_coefficients = np.polyfit(res_y_vals_in_meters, left_x_in_meters, 2)

    # Unpack coefficients
    # We are using this AWESOME tutorial: http://www.intmath.com/applications-differentiation/8-radius-curvature.php
    a, b = curvature_radius_polynomial_coefficients[0], curvature_radius_polynomial_coefficients[1]
    numerator = ((1 + (2 * a * y_values[-1] * meters_per_pixel_y_axis + b) ** 2) ** 1.5)
    denominator = np.absolute(2 * a)
    curve_radius = numerator / denominator
```

Then, to calculate the position of the car respect to the center of the lane we assumed that the camera was mounted in the center of the car, so the middle point of the image would give us the car position and the middle point within the lanes would give us the center of the lane. So, to know the position of the car respect to the center of the lane we just calculated the delta between these two quantities. Here's how:
```
    lane_center = (left_xs[-1] + right_xs[-1]) / 2
    image_center = original_image_width / 2
    center_delta_in_meters = (lane_center - image_center) * meters_per_pixel_x_axis
```

#### VI. _"Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly."_

After all the processing described above, we end up with a nice annotated image like this:
![alt-tag](https://github.com/jesus-a-martinez-v/advanced-lane-lines/blob/master/output_images/annotated_test3.jpg)

## Pipeline (Video)

#### I. _"Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)"_
 
You can watch the result of processing a footage from a camera mounted on a car by clicking [here!](https://drive.google.com/file/d/0B1SO9hJRt-hgR2hQdHgydEhOalk/view?usp=sharing) :).

## Discussion

#### I. _"Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?"_

One of the disadvantages of the current pipeline is that it isn't very customizable. For instance, the trapezoid vertices and the destination points used to calculate the perspective transform parameters are hardcoded, so a different camera setting (image size, position, etc) would most likely break the code. Also, the pipeline isn't very robust against major imperfections on the road like overlapping lane lines (this happens when older lane markings are visible along recently painted ones). It also has difficulties dealing with rapidly changing lighting conditions (like in the harder_challenge_video.mp4 video).

Although computer vision provides a really powerful set of tools, I find the fine-tuning process very exhausting and I am not so sure if this could scale well to a production environment, whereas a neural network, at least to me, has fewer knobs to tweak and converges to a more robust solution faster. The downside of this latter approach, of course, is that we have no control over how the network learns, which wraps all the process with a "magic" aura, difficulting the debugging activities. On the other hand, computer vision passes all the responsibility to us, and while this situation greatly increases the complexity, knowing exactly why something works (or not) is extremely useful. Perhaps the best solution lies somewhere in between these two worlds! :)
