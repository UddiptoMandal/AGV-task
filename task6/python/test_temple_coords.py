import numpy as np
import helper as hlp
# import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

# 1. Load the two temple images and the points from data/some_corresp.npz

data = np.load("data/some_corresp.npz")
# print(data.files)
pts1 = data["pts1"].astype(np.float64) # Corresponding points
pts2 = data["pts2"].astype(np.float64)

# Load the images
img1 = cv2.imread("data/im1.png")  
img2 = cv2.imread("data/im2.png")
# Convert images to RGB (since OpenCV loads in BGR format)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#loading intrinsic matrices
data_intrinsics = np.load('data/intrinsics.npz')
K1 = data_intrinsics['K1'].astype(np.float64)
K2 = data_intrinsics['K2'].astype(np.float64)  

# 2. Run eight_point to compute F
H1, W1 = img1.shape[:2]
M = max(H1, W1) 
F = sub.eight_point(pts1, pts2, M)


print("Fundamental Matrix F:\n", F)
# hlp.displayEpipolarF(img1, img2, F)


# 3. Load points in image 1 from data/temple_coords.npz
# data = np.load("data/temple_coords.npz")
# pts1 = data["pts1"].astype(np.float64) 



# 4. Run epipolar_correspondences to get points in image 2
pts2_estimated = sub.epipolar_correspondences(img1, img2, F, pts1)

# hlp.epipolarMatchGUI(img1, img2, F)



# 5. Compute the camera projection matrix P1
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))  


# 6. Use camera2 to get 4 camera projection matrices P2
E = sub.essential_matrix(F, K1, K2)
print("Computed Essential Matrix E:\n", E)

extrinsics = hlp.camera2(E) #4 possible values of P2
# print(extrinsics)



# 7. Run triangulate using the projection matrices

for i in range(4):  # Loop through all 4 possible extrinsics
    P2_test = K2 @ extrinsics[:, :, i]  # Get the i-th 3x4 matrix

    pts3d_test = sub.triangulate(P1, pts1, P2_test, pts2)
    
    
    # 8. Figure out the correct P2
    # Ensure most 3D points are in front of the camera (Z > 0)
    if np.sum(pts3d_test[:, 2] > 0) > len(pts3d_test) // 2:
        P2 = P2_test  
        R2 = extrinsics[:, :3, i]  # Extract Rotation matrix
        t2 = extrinsics[:, 3, i].reshape(-1, 1)  # Extract Translation vector
        break
# Compute 3D point
pts3d = sub.triangulate(P1, pts1, P2, pts2)
# print("Triangulated 3D Points:\n", pts3d)


def reprojection_error(P, pts2d, pts3d):

    # Convert 3D points to homogeneous form
    pts3d_hom = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))

    # Project 3D points onto 2D
    pts2d_proj = (P @ pts3d_hom.T).T
    pts2d_proj /= pts2d_proj[:, 2].reshape(-1, 1)  # Normalize homogeneous coordinates
    pts2d_proj = pts2d_proj[:, :2]

    # Compute mean Euclidean error
    error = np.linalg.norm(pts2d - pts2d_proj, axis=1).mean()
    return error

# Compute reprojection error for image 1
error = reprojection_error(P1, pts1, pts3d)
print(f"Reprojection Error: {error:.4f}")




# 9. Scatter plot the correct 3D points


def plot_3D_correspondences(pts3d, img1, pts1):
    fig = plt.figure(figsize=(24, 6))

    # Define three different 3D viewing angles
    angles = [
        (90, 90), 
        (20, 30), 
        (60, 120)  
    ]

    # First plot: 2D Correspondences on Image 1
    ax0 = fig.add_subplot(1, 4, 1)
    ax0.imshow(img1, cmap='gray')  # Show the first image
    ax0.scatter(pts1[:, 0], pts1[:, 1], c='r', marker='o')  # Overlay points
    ax0.set_title('Selected Points in Image 1')
    ax0.axis('off')

    # Three different 3D views
    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 4, i + 2, projection='3d')
        ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], c='b', marker='o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if i == 0:
            ax.set_title('Front View (Temple Structure)')
        else:
            ax.set_title(f'View {i+1}: Elev={elev}, Azim={azim}')

        ax.view_init(elev=elev, azim=azim)  # Set custom angle

    plt.show()

plot_3D_correspondences(pts3d, img1, pts1)

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
# Assume first camera is canonical
R1 = np.eye(3)  
t1 = np.zeros((3, 1))

np.savez('data/extrinsics.npz', R1 = R1, R2 = R2, t1 = t1, t2 = t2)

print("Extrinsic parameters saved successfully!")

