import numpy as np
import opencv as cv2
import matplotlib.pyplot as plt
import glob


def cal_calibrate_params(file_paths):
    object_points = []  # Points in three-dimensional space: 3D
    image_points = []  # Points in the image space: 2d
    # 2.1 Generate real intersection coordinates: 3D points like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)


    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    # 2.2 Detect corner coordinates of each image
    for file_path in file_paths:
        img = cv2.imread(file_path)
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Automatically detect the corners of the 4 chessboards in the chessboard (the intersection of 2 white and 2 black)
        rect, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If a corner point is detected, store it in object_points and image_points
        if rect == True:
            object_points.append(objp)
            image_points.append(corners)
        # 2.3 Get camera parameters
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


def img_undistort(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst


if __name__=="__main__":
    nx = 9
    ny = 6
    n_row=0
    n_col=1
    file_paths = glob.glob("Images/calibration*.jpg")
    ret, mtx, dist, rvecs, tvecs=cal_calibrate_params(file_paths)
    for img in file_paths:
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
        axs = axs.flatten()
        for imgs, ax in zip(img, axs):
            ax.imshow(imgs)
        plt.show()