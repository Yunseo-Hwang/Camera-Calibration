import cv2
import numpy as np

########## EXTRACT FEATURES
def extract_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    return corners

########## EXTRACT FEATURES
def compute_transformations(image_poses, pattern_poses, camera_matrix, dist_coeffs):
    _, rvec, tvec, _ = cv2.solvePnPRansac(pattern_poses, image_poses, camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    transformation_matrix = np.hstack((rotation_matrix, tvec))
    return transformation_matrix

########## EXTRACT FEATURES
def solve_calibration(transformation_matrices):
    A = np.zeros((3 * len(transformation_matrices), 3))
    B = np.zeros((3 * len(transformation_matrices), 1))

    for i, (T1, T2) in enumerate(zip(transformation_matrices[:-1], transformation_matrices[1:])):
        A[3*i:3*(i+1)] = np.eye(3) - T1[:3, :3]
        B[3*i:3*(i+1)] = T2[:3, 3:4] - T1[:3, 3:4]

    x = np.linalg.lstsq(A, B, rcond=None)[0]
    return x



camera_matrix = np.load('Hand-Eye-Calibration/cam_mtx.npy')
dist_coeff = np.load('Hand-Eye-Calibration/dist_coeff.npy')