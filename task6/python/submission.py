"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import helper as hlp
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d



"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    
    
    # 1. Normalize Points
    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])  # Scale transformation matrix
    pts1_h = np.column_stack((pts1, np.ones(len(pts1))))  # Convert to homogeneous
    pts2_h = np.column_stack((pts2, np.ones(len(pts2))))
     # Normalize points
    pts1_norm = (T @ pts1_h.T).T 
    pts2_norm = (T @ pts2_h.T).T

    # 2. Construct Matrix A
    x1, y1 = pts1_norm[:, 0], pts1_norm[:, 1]
    x2, y2 = pts2_norm[:, 0], pts2_norm[:, 1]
    A = np.column_stack((x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, np.ones(len(pts1))))

    # 3. Compute F using SVD
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)  # Take last column as solution

    # 4. Enforce Rank-2 Constraint
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0  # Force last singular value to zero
    F_rank2 = U @ np.diag(S) @ Vt

    # 5. Refine F 
    # F_rank2 = hlp.refineF(F_rank2, pts1_norm, pts2_norm)
    # 6. Unscale F
    F_final = T.T @ F_rank2 @ T

    return F_final


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    window_size = 5
    # Convert images to grayscale if they are in color
    if len(im1.shape) == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) == 3:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    half_w = window_size // 2
    pts2 = []

    for x, y in pts1:
        # Convert to homogeneous coordinates
        pt1_h = np.array([x, y, 1]).reshape(3, 1)

        # Compute the epipolar line in the second image
        epipolar_line = F @ pt1_h
        a, b, c = epipolar_line.flatten()

        # Generate candidate points along the epipolar line
        candidates = []
        height, width = im2.shape

        for x2 in range(max(half_w, int(x - 50)), min(width - half_w, int(x + 50))):
            y2 = int(-(a * x2 + c) / b) if b != 0 else int(y)
            
            if half_w <= y2 < height - half_w:  # Ensure within bounds
                candidates.append((x2, y2))

        # Extract reference patch from im1
        x, y = int(round(x)), int(round(y))
        patch1 = im1[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1]


        best_match = None
        min_distance = float("inf")

        for x2, y2 in candidates:
            x2, y2 = int(round(x2)), int(round(y2))
            patch2 = im2[y2 - half_w:y2 + half_w + 1, x2 - half_w:x2 + half_w + 1]


            # Compute SSD (Sum of Squared Differences or Euclidean distances)
            distance = np.sum((patch1.astype(float) - patch2.astype(float))**2)

            if distance < min_distance:
                min_distance = distance
                best_match = (x2, y2)

        if best_match:
            pts2.append(best_match)
        else:
            pts2.append((x, y))  # in case no match is found

    return np.array(pts2)
"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):

    
    E = K2.T @ F @ K1 

    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
   # Convert points to homogeneous coordinates (3xN)
    N = len(pts1)  # Number of points
    pts3d_homogeneous = np.zeros((N, 4))  # Store 3D points in homogeneous coordinates

    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        # Build the system of equations (Ax = 0)
        A = np.vstack([
            y1 * P1[2, :] - P1[1, :],
            P1[0, :] - x1 * P1[2, :],
            y2 * P2[2, :] - P2[1, :],
            P2[0, :] - x2 * P2[2, :]
        ])

        # Solve using SVD
        U, S, Vt = np.linalg.svd(A)
        X = Vt[-1]  # Last column of V gives the solution

        # Normalize homogeneous coordinates
        pts3d_homogeneous[i] = X / X[-1]

    # Convert to Euclidean coordinates (drop homogeneous component)
    pts3d = pts3d_homogeneous[:, :3]

    return pts3d
"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    #1
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)
    #2
    r1 = (c1 - c2) / np.linalg.norm(c1 - c2)  # New x-axis (baseline direction)
    r1 = r1.flatten() 
    r2 = np.cross(R1[2, :], r1)  # New y-axis (orthogonal to x and old z)
    r2 = r2 / np.linalg.norm(r2)  # Normalize
    r3 = np.cross(r2, r1)  # New z-axis (orthogonal to x and y)
    Re = np.vstack((r1, r2, r3))  # Final rotation matrix

    R1p = Re
    R2p = Re
    #3
    # Set new intrinsic parameters
    K1p = K2p = K2 
    #4
    # Compute new translation vectors
    t1p = -Re @ c1
    t2p = -Re @ c2
    #5
    M1 = (K1p @ R1p) @ np.linalg.inv(K1 @ R1)
    M2 = (K2p @ R2p) @ np.linalg.inv(K2 @ R2)
    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p



"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    h, w = im1.shape  # Image dimensions
    dispM = np.zeros((h, w), dtype=np.float32)  # Initialize disparity map
    w_half = win_size // 2 

    mask = np.ones((win_size, win_size))
    min_ssd = np.full((h, w), np.inf)  # Store min SSD values
    for d in range(max_disp + 1):
        im2_shifted = np.zeros_like(im2)  # Initialize shifted image
        
        if d > 0:  # Only shift if d > 0
            im2_shifted[:, d:] = im2[:, :-d]  # Shift right image to the left by d pixels
        else:
            im2_shifted = im2.copy()  # No shift needed for d = 0

        # Compute squared differences
        ssd = (im1 - im2_shifted) ** 2

        # Apply convolution for window sum
        ssd_sum = convolve2d(ssd, mask, mode='same', boundary='fill', fillvalue=0)

        # Update disparity where SSD is minimum
        update_mask = ssd_sum < min_ssd
        min_ssd[update_mask] = ssd_sum[update_mask]
        dispM[update_mask] = d  # Assign disparity value

    return dispM




"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
     # Compute optical centers c1 and c2
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)

    # Compute baseline (distance between camera centers)
    b = np.linalg.norm(c1 - c2)

    # Get focal length from K1 
    f = K1[1, 1]

    # Compute depth map
    depthM = np.zeros_like(dispM, dtype=np.float32)
    valid_disp = dispM > 0  # Ensure we don't divide by zero
    depthM[valid_disp] = (b * f) / dispM[valid_disp]

    # Ensure depthM is exactly 0 where dispM is 0
    depthM[dispM == 0] = 0

    return depthM


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
