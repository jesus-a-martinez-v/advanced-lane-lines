import numpy as np
import cv2
import glob
import pickle


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


if __name__ == '__main__':
    calibrate_camera()
