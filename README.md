# Monocular Visual Odometry

This repository contains a Python implementation of a monocular visual odometry (VO) system. It estimates the trajectory of a camera moving through an environment using a single video as input.  The system uses feature detection and matching to track the camera's motion between frames.

## Features

* **Feature Detection and Matching:** Uses ORB (Oriented FAST and Rotated BRIEF) features for robust tracking.  Matches features between consecutive frames using a Brute-Force matcher with distance filtering.
* **Motion Estimation:** Estimates the 2D affine transformation (rotation and translation) between frames using `cv2.estimateAffinePartial2D` with RANSAC to handle outliers.
* **Trajectory Estimation:** Integrates the estimated translation over time to reconstruct the camera's path.  A scaling factor is applied to the translation to convert it to real-world units.
* **Ground Truth Comparison:** Loads ground truth trajectory data from a CSV file and plots it alongside the estimated trajectory for visual comparison.
* **Visualization:** Displays the feature matches between frames and plots the estimated and ground truth trajectories.

## Requirements

* Python 3.x
* NumPy
* OpenCV (cv2)
* Matplotlib
# Usage
* Video: Place your input video (Video.MP4 in the example code) in the same directory as the script.

* Ground Truth: Provide a CSV file (Translations.csv in the example) containing the ground truth trajectory. The file should have a header row and two columns representing the x and y coordinates of the camera's position for each frame.

* Calibration: Update the camera_matrix and dist_coeffs variables in the script with your camera's intrinsic parameters. These are essential for accurate motion estimation.

* Run: Execute the script using python your_script_name.py

* The script will process the video, estimate the trajectory, display feature matches, and plot the estimated trajectory against the ground truth.

# Output
* Trajectory Plot: A plot showing the estimated and ground truth camera trajectories.

* Feature Matches: A window displaying the feature matches between consecutive frames.

# Limitations
* Scale Ambiguity: Monocular VO suffers from scale ambiguity, meaning the estimated trajectory's scale may differ from the ground truth. A fixed scaling factor is used in this implementation, but a more sophisticated method would be required for accurate scale estimation.

* 2D Motion Estimation: This code estimates 2D affine transformations, which is a simplification of the true 3D motion. For full 3D motion estimation, a different approach (e.g., using essential matrix decomposition) would be necessary.

* No Loop Closure: This implementation does not include loop closure detection, which is crucial for long-term trajectory accuracy in SLAM systems.

* Assumed Undistorted Images: The code assumes that images are either already undistorted or that the undistortion is handled appropriately. If significant lens distortion is present, the results will be inaccurate.

# Future Improvements
* Implement 3D motion estimation using essential matrix or homography decomposition.

* Add loop closure detection and optimization.

* Implement automatic scale estimation.

* Add support for different feature detectors and matchers.

# Contributors

* Nursen Marancı

* Sena Varıcı
