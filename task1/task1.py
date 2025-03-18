import cv2
import numpy as np
# sobel for calculating derivatives
Gx = [[-5, -4, 0, 4, 5], [-8, -10, 0, 10, 8], [-10, -20, 0, 20, 10], [-8, -10, 0, 10, 8], [-5, -4, 0, 4, 5]]
Gy = [[5, 8, 10, 8, 5], [4, 10, 20, 10, 4], [0, 0, 0, 0, 0], [-4, -10, -20, -10, -4], [-5, -8, -10, -8, -5]]

cap = cv2.VideoCapture("OPTICAL_FLOW_clip.mp4")
#parameters for shi tomasi feature detection
feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2)
#initializing prev frame
ret, prev = cap.read()
if not ret:
    print("Failed to read video")
    cap.release()
    exit()
# grayscale the image
prevgr = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
points_and_vectors = [] # Store points and their vectors
#loop through all frams
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        framegr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #locating corner points in the frame
        pointsold = cv2.goodFeaturesToTrack(prevgr, mask=None, **feature_params)

        if pointsold is not None:

            for point in pointsold:
                x, y = int(point[0][0]), int(point[0][1])
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

                if y < 2 or x < 2 or y > framegr.shape[0] - 3 or x > framegr. shape[1] - 3:
                    continue
                #slicing frames into 5x5 kernels
                arr1 = prevgr[y - 2:y + 3, x - 2:x + 3].astype(np.float32)
                arr2 = framegr[y - 2:y + 3, x - 2:x + 3].astype(np.float32)
                #initializing derivatives
                Ix = np.zeros((5, 5), dtype=np.float32)
                Iy = np.zeros((5, 5), dtype=np.float32)
                #finding Ix and Iy by convolving with sobel operator
                for i in range(5):
                    for j in range(5):
                        Ix[i, j] = sum(arr1[k, l] * Gx[i][j] for k in range(5) for l in range(5)) / 40.0
                        Iy[i, j] = sum(arr1[k, l] * Gy[i][j] for k in range(5) for l in range(5)) / 40.0

                It = arr1 - arr2

                Ix_flat = Ix.flatten()
                Iy_flat = Iy.flatten()
                It_flat = It.flatten()
                # AxU = B
                A = np.array([Ix_flat, Iy_flat]).T
                B = np.array(It_flat)
                # print('A', A)
                # print('B', B)
                #exception handling to handle case where A.T times A is singular
                try:
                    U = (np.linalg.inv((A.T)@A)@(A.T))@B
                except:
                    continue
                    
                # 
                #times 75 to view the lines better 
                # I am unable to normalize it to the level of the inbuild function in opencv. I have tried
                # many different values other than 75, and i have tried using different methods to compute derivatives
                # such as sobel 3x3 and also by using definition and averaging values for a 2x2 kernel
                # but this method gave the best results compared to the other ones I tried.
                dx, dy = int(U[0]*75), int(U[1]*75)
                # print(dx, dy)
                points_and_vectors.append([(x,y), (x+(dx), y+(dy))])
                
                
                
        #drawing all the lines as a mask on the frames
        for item in points_and_vectors:
            cv2.line(frame, item[0], item[1], (0, 255, 0), 3)

        cv2.imshow("final", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        prevgr = framegr
    else:
        break
# print(points_and_vectors)
cap.release()
cv2.destroyAllWindows()
