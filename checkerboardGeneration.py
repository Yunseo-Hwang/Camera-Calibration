import cv2
import os

# Path to input video
video_path = 'checkerboard.mp4'

# Directory to save extracted images
output_dir = 'extracted_images'
os.makedirs(output_dir, exist_ok=True)

# Parameters for checkerboard detection
checkerboard_size = (10, 8) 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Open the video file
cap = cv2.VideoCapture(video_path)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    if ret:
        # Refine corner positions
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Save extracted image
        image_name = f'image_{frame_count:04d}.jpg'
        cv2.imwrite(os.path.join(output_dir, image_name), frame)
        print(f'Saved {image_name}')

    frame_count += 1

cap.release()
cv2.destroyAllWindows()