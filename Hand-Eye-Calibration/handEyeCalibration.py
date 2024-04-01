import cv2
import numpy as np
import pandas as pd
import glob

########## EXTRACT FEATURES
def extract_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    return corners

########## COMPUTE HAND-EYE TRANSFORMATION
def compute_transformations(image_poses, pattern_poses, camera_matrix, dist_coeffs):
    print("Number of points in pattern poses:", pattern_poses.shape[0])
    print("Number of points in image poses:", image_poses.shape[0])
    _, rvec, tvec, _ = cv2.solvePnPRansac(pattern_poses, image_poses, camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    transformation_matrix = np.hstack((rotation_matrix, tvec))
    return transformation_matrix

########## SOLVE CALIBRATION PROBLEM
def solve_calibration(transformation_matrices):
    A = np.zeros((3 * len(transformation_matrices), 3))
    B = np.zeros((3 * len(transformation_matrices), 1))

    for i, (T1, T2) in enumerate(zip(transformation_matrices[:-1], transformation_matrices[1:])):
        A[3*i:3*(i+1)] = np.eye(3) - T1[:3, :3]
        B[3*i:3*(i+1)] = T2[:3, 3:4] - T1[:3, 3:4]

    x = np.linalg.lstsq(A, B, rcond=None)[0]
    return x

# Load images
images = [cv2.imread(image_path) for image_path in glob.glob('Intrinsic-Camera-Parameter-Calibration/extracted_images/*.jpg')]

# Load optical marker poses
df = pd.read_csv('Hand-Eye-Calibration/tf_poses_camhand.csv', header=None, names=['timestamps', 'translation_x', 'translation_y', 'translation_z', 'quaternion_x', 'quaternion_y', 'quaternion_z', 'quaternion_w'])
timestamps = df['timestamps']
translations = df[['translation_x', 'translation_y', 'translation_z']].values
quaternions = df[['quaternion_x', 'quaternion_y', 'quaternion_z', 'quaternion_w']].values

# Convert quaternions to rotation matrices
rotation_matrices = []
for quaternion in quaternions:
    q = np.array(quaternion)
    q /= np.linalg.norm(q)
    rotation_matrix = np.array([[1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
                                  [2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]],
                                  [2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], 1 - 2*q[1]**2 - 2*q[2]**2]])
    rotation_matrices.append(rotation_matrix)

# Construct pattern_poses using rotation_matrices and translations
pattern_poses = np.array([np.hstack((rotation_matrices[i], translations[i][:, np.newaxis])) for i in range(len(images))])

# Ensure pattern_poses match the number of keypoints
pattern_poses = np.array([pattern_poses[0] for _ in range(len(images))])

# Extract keypoints for all images
keypoints = [extract_keypoints(image) for image in images]

# Convert keypoints to numpy array for each image
image_poses = [np.array([keypoint.squeeze() for keypoint in keypoints[i]]) for i in range(len(images))]

# Load intrinsic camera matrix
camera_matrix = np.load('Hand-Eye-Calibration/cam_mtx.npy')

# Load distortion coefficients
dist_coeff = np.load('Hand-Eye-Calibration/dist_coeff.npy')

# Compute transformation matrices
transformation_matrices = []

for i in range(len(images)):
    print("Processing image", i+1)
    print("Pattern Poses:", pattern_poses[i])
    print("Image Poses:", image_poses[i])
    transformation_matrices.append(compute_transformations(image_poses[i], pattern_poses[i], camera_matrix, dist_coeff))

calibration_result = solve_calibration(transformation_matrices)