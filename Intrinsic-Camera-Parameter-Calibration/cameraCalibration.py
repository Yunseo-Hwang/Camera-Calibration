import numpy as np
import cv2 as cv
import glob

def generate_object_and_image_points(images, chessboard_size, checker_size):
    object_points = []
    image_points = []

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2) * checker_size
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # terminate criteria

    for image in images:
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
        
        if ret == True:
            object_points.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(corners2)

            cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
            # cv.imshow('img', img)
            # cv.waitKey(1000)

        cv.destroyAllWindows()
    return object_points, image_points

# Given parameters
chessboard_size = (10, 8)  # Number of inner corners
frame_size = (1920, 1080)  # Width and height of the image frame in pixels
checker_size = (47, 47)    # Size of each checker in mm
marker_size = (35, 35)     # Size of each marker in mm

# Given image path
images = glob.glob('extracted_images/' + '*.jpg')

# Generate object points and image points
object_points, image_points = generate_object_and_image_points(images, chessboard_size, checker_size)

print("object points: ", len(object_points), "x", len(object_points[0]), "x", len(object_points[0][0]))
print("object points: ", len(image_points), "x", len(image_points[0]), "x", len(image_points[0][0]), "\n")

########## Calibrate Camera
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(object_points, image_points, frame_size, None, None)

print("Camera Calibrated: ", ret, "\n")
# [fx 0  cx]   [1.45509271e+03 0.00000000e+00 9.33812616e+02]
# [0  fy cy]   [0.00000000e+00 1.45947634e+03 4.81809080e+02]
# [0  0  1 ]   [0.00000000e+00 0.00000000e+00 1.00000000e+00]
# fx and fy are focal length and cx and cy are optical center coordinates

print("Camera Matrix: ", cameraMatrix, "\n")
print("Distortion Parameters: ", dist, "\n")
print("Rotation Vectors: ", rvecs, "\n")
print("Translation Vectors: ", tvecs, "\n")

########### Undistort image
img = cv.imread('extracted_images/image_0030.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

# using cv.undistort()
dst = cv.undistort(img, cameraMatrix, dist, None, newcameramtx)
 
# using remapping
# mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
# dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibrated_images/calibrated_image_0030.jpg', dst)

########## Reprojection error
mean_error = 0
for i in range(len(object_points)):
    imgpoints2, _ = cv.projectPoints(object_points[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(image_points[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error
 
print("Total error: {}".format(mean_error/len(object_points)) )