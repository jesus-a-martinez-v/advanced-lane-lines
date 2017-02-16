import glob
import numpy as np
import pickle
import cv2
from history_keeper import HistoryKeeper

__history_keeper = HistoryKeeper(window_width=30, window_height=60, margin=25, smooth_factor=30)


def load_calibration_parameters(location='./cal_pickle.p'):
    """
    Loads the calibration parameters (used for distortion correction) from a particular location and returns them.
    :param location: Path of the pickle file where the parameters are stored.
    :return:
    """
    with open(location, 'rb') as pickle_file:
        dist_pickle = pickle.load(pickle_file)

    return dist_pickle['mtx'], dist_pickle['dist'], dist_pickle['rvecs'], dist_pickle['tvecs']


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


def calculate_perspective_transform_parameters():
    """
    Calculates the parameters needed to transform the perspective an image to a birds-eye view.
    :return: 1. The transformation matrix needed to pass from the original perspective to the top perspective.
             2. The inverse of the transformation matrix, which is useful when we need to revert the perspective transformation.
             3. The source points, which describe a trapezoid in the original image.
             4. The destination points, which describe a rectangle in the transformed image.
    """
    # width, height = image_shape
    # print("width", width)
    # print("height", height)
    #
    # bottom_width = 0.76
    # # bottom_width = 0.8
    # # mid_width = 0.1
    # mid_width = 0.08
    # height_percentage = 0.62
    # # height_percentage = 0.58
    # bottom_trim = 0.94
    # # bottom_trim = 0.935
    #
    # src = np.float32([[width * (0.5 - mid_width / 2), height * height_percentage],
    #                   [width * (0.5 + mid_width / 2), height * height_percentage],
    #                   [bottom_width * width, height * bottom_trim],
    #                   [width - bottom_width * width, height * bottom_trim]])

    src = np.float32([[589,  446], [691,  446], [973,  677], [307,  677]])

    # src = np.float32([[585, 460], [203, 720], [1127, 720], [695, 460]])
    # dst = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])
    #
    # print("src", src)
    # offset = width * 0.25
    # # offset = width * 0.245
    # dst = np.float32([[offset, 0],
    #                   [width - offset, 0],
    #                   [width - offset, height],
    #                   [offset, height]])
    dst = np.float32([[320, 0], [960, 0], [960, 720], [320, 720]])
    #
    # print("dst", dst)

    transform_matrix = cv2.getPerspectiveTransform(src, dst)
    inverse_transform_matrix = cv2.getPerspectiveTransform(dst, src)

    return transform_matrix, inverse_transform_matrix, src, dst


def window_mask(window_width, window_height, image, center, level):
    image_width, image_height = image.shape[1], image.shape[0]

    height_lower_boundary = int(image_height - (level + 1) * window_height)
    height_upper_boundary = int(image_height - level * window_height)

    width_lower_boundary = max(0, int(center - window_width))
    width_upper_boundary = min(int(center + window_width), image_width)

    binary_output = np.zeros_like(image)
    binary_output[height_lower_boundary: height_upper_boundary,
    width_lower_boundary: width_upper_boundary] = 1
    return binary_output


def put_offset_and_radius(image, offset_from_center, radius):
    """
    Puts the offset from the center of the lane and the radius of the curvature on top of a provided image.
    :param image: Image where we'll place the text.
    :param offset_from_center: Offset (in meters) of the car from the center of the lane.
    :param radius: Radius (in meter) of the lane curvature.
    :return: Same image with the text on top of it.
    """
    cv2.putText(image, 'Radius of the Curvature (in meters) = ' + str(round(radius, 3)), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(image, 'Car is ' + str(abs(round(offset_from_center, 3))) + ' meters off of center', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return image


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


def put_lines_on_image(input_image, reverse_perspective_transform_matrix, left_lane, right_lane, inner_lane):
    """
    Overlaps the lane lines found in a top perspective image over the original image.
    :param input_image: Original, undistorted image.
    :param reverse_perspective_transform_matrix:  Matrix used to revert the perspective transform of the lanes.
    :param left_lane: Left lane polygon points.
    :param right_lane: Right lane polygon points.
    :param inner_lane: Inner lane polygon points.
    :return: Original image with the lane lines found drawn onto it.
    """
    image_size = (input_image.shape[1], input_image.shape[0])

    base_road = np.zeros_like(input_image)
    road_background = np.zeros_like(input_image)

    cv2.fillPoly(base_road, [left_lane], color=[255, 0, 0])
    cv2.fillPoly(base_road, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(base_road, [inner_lane], color=[0, 255, 0])
    cv2.fillPoly(road_background, [left_lane], color=[255, 255, 255])
    cv2.fillPoly(road_background, [right_lane], color=[255, 255, 255])

    road_warped = cv2.warpPerspective(base_road, reverse_perspective_transform_matrix, image_size,
                                      flags=cv2.INTER_LINEAR)
    road_warped_background = cv2.warpPerspective(road_background, reverse_perspective_transform_matrix, image_size,
                                                 flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(input_image, 1.0, road_warped_background, -1.0, 0.0)
    return cv2.addWeighted(base, 1, road_warped, .7, 0.0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    a, b, c, d = vertices
    vertices = np.array([[c, a, b, d]], dtype=np.int32)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


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


def process_image(input_image, curve_centers=__history_keeper):
    """
    Takes an image and returns it with the lane lines, radius curvature and center offset drawn on top of it.
    :param input_image: RGB image.
    :param curve_centers: History keeper objects used to keep the centroids of the windows found in each frame.
    :return: Original image with the lane lines, radius curvature and center offset drawn on top of it.
    """
    ########################################################
    # STEP 1: Undistort image and extract image dimensions #
    ########################################################
    camera_matrix, distortion_coefficients, _, _ = load_calibration_parameters()
    input_image = cv2.undistort(input_image, camera_matrix, distortion_coefficients, None, camera_matrix)
    original_image_size = (input_image.shape[1], input_image.shape[0])
    original_image_width, original_image_height = original_image_size

    ###############################################
    # STEP 2: Apply color & gradient thresholding #
    ###############################################
    processed_image = np.zeros_like(input_image[:, :, 0])  # Just a black canvas
    x_gradient_mask = abs_sobel_threshold(input_image, orientation='x', threshold=(12, 255), sobel_kernel=3)
    y_gradient_mask = abs_sobel_threshold(input_image, orientation='y', threshold=(25, 255), sobel_kernel=3)
    color_binary_mask = color_threshold(input_image, s_threshold=(100, 255), v_threshold=(50, 255))
    selection = (x_gradient_mask == 1) & (y_gradient_mask == 1) | (color_binary_mask == 1)
    processed_image[selection] = 255

    ########################################
    # STEP 3: Apply perspective transform. #
    ########################################
    perspective_matrix, revert_perspective_matrix, src, dst = calculate_perspective_transform_parameters()
    warped = cv2.warpPerspective(processed_image, perspective_matrix, original_image_size, flags=cv2.INTER_LINEAR)

    ###########################################################
    # STEP 4: Find the centroids of each window in each line. #
    ###########################################################
    window_centroids = curve_centers.find_window_centroids(warped)

    # points used to find the lef+t and right lane lines
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

    # Left and right lines polygons
    lest_lane_vertices = get_line_polygon(left_xs, y_values, thickness=curve_centers.window_width)
    right_lane_vertices = get_line_polygon(right_xs, y_values, thickness=curve_centers.window_width)

    # The inner lane is actually the space where the car is, so it must go from the rightmost points in the left lane
    # to the leftmost points in the right lane.
    mid_point_window_height = curve_centers.window_width / 2
    inner_lane_vertices = np.array(
        list(zip(np.concatenate((left_xs + mid_point_window_height, right_xs[::-1] - mid_point_window_height), axis=0),
                 np.concatenate((y_values, y_values[::-1]), axis=0))), np.int32)

    # Put the lines on the original image.
    result_image = put_lines_on_image(input_image, revert_perspective_matrix, lest_lane_vertices, right_lane_vertices,
                                      inner_lane_vertices)

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

    #######################################################
    # STEP 7: Calculate the offset of the car on the road #
    #######################################################
    # We assume our camera is fixed at the center of the car lanes.
    lane_center = (left_xs[-1] + right_xs[-1]) / 2
    image_center = original_image_width / 2
    center_delta_in_meters = (lane_center - image_center) * meters_per_pixel_x_axis

    result_image = put_offset_and_radius(result_image, center_delta_in_meters, curve_radius)

    # Finally return the original image with all the data of interest shown.
    return result_image
