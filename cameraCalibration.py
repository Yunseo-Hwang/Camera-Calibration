import numpy as np
import cv2 as cv
import glob

########## FIND CHESSBOARD CORNERS - object points and image points

chessboardSize = (10, 8) # number of corners in width and heigth
frameSize = (1440, 1080) 

# terminate criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# object points
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32) # (0,0,0), (0,0,0), ...
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2) # (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0), ...

# arrays holding object points and image points of images
objPoints = [] # 3d points in real-world space
imgPoints = [] # 2d points in image plane

images = glob.glob('extracted_images/' + '*.jpg')

for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    
    # If found, add object points and image points
    if ret == True:
        objPoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgPoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(1000)

cv.destroyAllWindows()


########## CALIBRATION

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, frameSize, None, None)

print("Camera Calibrated: ", ret, "\n")
# [fx 0  cx]   [1.45509271e+03 0.00000000e+00 9.33812616e+02]
# [0  fy cy]   [0.00000000e+00 1.45947634e+03 4.81809080e+02]
# [0  0  1 ]   [0.00000000e+00 0.00000000e+00 1.00000000e+00]
# fx and fy are focal length and cx and cy are optical center coordinates

print("Camera Matrix: ", cameraMatrix, "\n")
print("Distortion Parameters: ", dist, "\n")
print("Rotation Vectors: ", rvecs, "\n")
print("Translation Vectors: ", tvecs, "\n")


########## UNDISTORTION