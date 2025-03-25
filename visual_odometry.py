import numpy as np
import cv2
import matplotlib.pyplot as plt


def estimate_motion(matches, kp1, kp2):
    if len(matches) > 5:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        M = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)[0]

        if M is not None:
            R = M[:, :2]
            t = M[:, 2]
            return R, t
    return None, None


def load_ground_truth(file_path):
    return np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=(0, 1))


def plot_trajectory(estimated_trajectory, ground_truth):
    plt.figure()
    plt.title('Camera Trajectory')
    plt.plot(ground_truth[:, 0], ground_truth[:, 1], label='Ground Truth', linestyle='dashed')

    
    estimated_trajectory_x_sym = np.copy(estimated_trajectory)
    estimated_trajectory_x_sym[:, 0] = -estimated_trajectory[:, 0]

    
    estimated_trajectory_xy_sym = np.copy(estimated_trajectory_x_sym)
    estimated_trajectory_xy_sym[:, 1] = -estimated_trajectory_x_sym[:, 1]
    plt.plot(estimated_trajectory_xy_sym[:, 0], estimated_trajectory_xy_sym[:, 1], label='XY Symmetry', linestyle='dashed')
    plt.xlabel('X position (units)')
    plt.ylabel('Y position (units)')
    plt.legend()
    plt.show()


def process_video(video_source, camera_matrix, dist_coeffs, ground_truth_path):
    cap = cv2.VideoCapture(video_source)
    orb = cv2.ORB_create(1000,
                         WTA_K=3,
                         scoreType=cv2.ORB_HARRIS_SCORE,
                         patchSize=31,
                         edgeThreshold=31,
                         nlevels=32,
                         fastThreshold=10)
    bf = cv2.BFMatcher()

    ret, old_frame = cap.read()
    if not ret:
        print("Cannot read the video or video ended!")
        return
    
    old_frame = cv2.resize(old_frame, (640, 480))
    old_frame = cv2.undistort(old_frame, camera_matrix, dist_coeffs)
    old_kp, old_des = orb.detectAndCompute(old_frame, None)

    ground_truth = load_ground_truth(ground_truth_path)
    estimated_trajectory = np.zeros((len(ground_truth), 2))
    total_translation = np.array([0.0, 0.0])

    frame_index = 0
    while True:
        ret, new_frame = cap.read()
        if not ret:
            break

        new_frame  = cv2.resize(new_frame, (640, 480))
        new_frame = cv2.undistort(new_frame, camera_matrix, dist_coeffs)
        new_kp, new_des = orb.detectAndCompute(new_frame, None)
        matches = bf.knnMatch(old_des, new_des, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.3 * n.distance:
                good_matches.append(m)

        R, t = estimate_motion(good_matches, old_kp, new_kp)
        if R is not None and t is not None:
            total_translation += t  
            estimated_trajectory[frame_index] = total_translation * 0.07
            frame_index += 1

        img_matches = cv2.drawMatches(old_frame, old_kp, new_frame, new_kp, good_matches, None, flags=2)
        cv2.imshow('Feature Matches', img_matches)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        old_frame = new_frame.copy()
        old_kp, old_des = new_kp, new_des

    cap.release()
    cv2.destroyAllWindows()
    plot_trajectory(estimated_trajectory[:frame_index], ground_truth[:frame_index])


focal_length = [1413.3, 1418.8]
principal_point = [950.0639, 543.3796]
camera_matrix = np.array([
    [focal_length[0], 0, principal_point[0]],
    [0, focal_length[1], principal_point[1]],
    [0, 0, 1]
], dtype=float)

dist_coeffs = np.array([-0.0091, 0.0666, 0, 0])

process_video('Video.MP4', camera_matrix, dist_coeffs, 'Translations.csv')

