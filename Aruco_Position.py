import cv2 as cv
from cv2 import aruco
import numpy as np

# Load camera calibration data
calib_data_path = "calib_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]

# Marker settings
MARKER_SIZE = 50  # mm (if in cm, use MARKER_SIZE = 5)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# Open camera
cap = cv.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect ArUco markers
    marker_corners, ids, _ = detector.detectMarkers(gray_frame)

    if ids is not None:
        for i in range(len(ids)):
            if ids[i][0] == 1:  # Only process ArUco ID 1
                # Define 3D marker points with center as origin
                obj_points = np.array([
                    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
                    [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
                    [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
                    [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0]
                ], dtype=np.float32)

                # Solve PnP to get position relative to camera
                _, rVec, tVec = cv.solvePnP(obj_points, marker_corners[i], cam_mat, dist_coef)
                x, y, z = tVec.flatten()
                print(f"ArUco ID 1 Position in Camera Frame: X={x:.2f} mm, Y={y:.2f} mm, Z={z:.2f} mm")

                # Draw marker outline
                corners = marker_corners[i].reshape(4, 2).astype(int)
                cv.polylines(frame, [corners], True, (0, 255, 0), 2)
                cv.putText(frame, f"ID: {ids[i][0]}", (corners[0][0], corners[0][1] - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw coordinate axes of marker
                axis_length = 30  # mm
                axis_points_3D = np.array([
                    [axis_length, 0, 0],  # X-axis (Red)
                    [0, axis_length, 0],  # Y-axis (Green)
                    [0, 0, axis_length]   # Z-axis (Blue)
                ], dtype=np.float32)

                center_3D = np.array([[0, 0, 0]], dtype=np.float32)
                center_2D, _ = cv.projectPoints(center_3D, rVec, tVec, cam_mat, dist_coef)
                axis_points_2D, _ = cv.projectPoints(axis_points_3D, rVec, tVec, cam_mat, dist_coef)
                
                center_x, center_y = int(center_2D[0][0][0]), int(center_2D[0][0][1])
                x_axis_x, x_axis_y = int(axis_points_2D[0][0][0]), int(axis_points_2D[0][0][1])
                y_axis_x, y_axis_y = int(axis_points_2D[1][0][0]), int(axis_points_2D[1][0][1])
                z_axis_x, z_axis_y = int(axis_points_2D[2][0][0]), int(axis_points_2D[2][0][1])

                cv.line(frame, (center_x, center_y), (x_axis_x, x_axis_y), (0, 0, 255), 3)
                cv.line(frame, (center_x, center_y), (y_axis_x, y_axis_y), (0, 255, 0), 3)
                cv.line(frame, (center_x, center_y), (z_axis_x, z_axis_y), (255, 0, 0), 3)

    # Draw camera coordinate frame
    camera_axis_length = 50  # mm
    camera_axes_3D = np.array([
        [camera_axis_length, 0, 0],
        [0, camera_axis_length, 0],
        [0, 0, camera_axis_length]
    ], dtype=np.float32)
    
    camera_origin_3D = np.array([[0, 0, 0]], dtype=np.float32)
    camera_origin_2D, _ = cv.projectPoints(camera_origin_3D, np.zeros((3, 1)), np.zeros((3, 1)), cam_mat, dist_coef)
    camera_axes_2D, _ = cv.projectPoints(camera_axes_3D, np.zeros((3, 1)), np.zeros((3, 1)), cam_mat, dist_coef)
    
    cam_x, cam_y = int(camera_origin_2D[0][0][0]), int(camera_origin_2D[0][0][1])
    cam_x_axis_x, cam_x_axis_y = int(camera_axes_2D[0][0][0]), int(camera_axes_2D[0][0][1])
    cam_y_axis_x, cam_y_axis_y = int(camera_axes_2D[1][0][0]), int(camera_axes_2D[1][0][1])
    cam_z_axis_x, cam_z_axis_y = int(camera_axes_2D[2][0][0]), int(camera_axes_2D[2][0][1])
    
    cv.line(frame, (cam_x, cam_y), (cam_x_axis_x, cam_x_axis_y), (0, 0, 255), 3)
    cv.line(frame, (cam_x, cam_y), (cam_y_axis_x, cam_y_axis_y), (0, 255, 0), 3)
    cv.line(frame, (cam_x, cam_y), (cam_z_axis_x, cam_z_axis_y), (255, 0, 0), 3)

    # Display frame
    cv.imshow("ArUco Detection with Axes and Camera Origin", frame)
    
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
