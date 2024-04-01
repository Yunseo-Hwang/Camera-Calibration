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

img = cv.imread('extracted_images/image_0030.jpg')
h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

# undistort the image
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibrated_image_0030.jpg', dst)

# undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, cameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('remapped_image_0030.jpg', dst)

# reprojection error
mean_error = 0

for i in range(len(objPoints)):
    imgPoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgPoints[i], imgPoints2, cv.NORM_L2)/len(imgPoints2)
    mean_error += error

print("total error: {}".format(mean_error/len(objPoints)))