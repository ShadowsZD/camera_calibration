import numpy as np
import cv2
import glob
import argparse

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

parser = argparse.ArgumentParser()

parser.add_argument('input_corners_height', type=int, help='number of inside corners for black squares on height')
parser.add_argument('input_corners_width', type=int, help='number of inside corners for black squares on width')

args = parser.parse_args()

input_corners_height = args.input_corners_height
input_corners_width = args.input_corners_width

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#inside_corners_height
objp = np.zeros((input_corners_height*input_corners_width,3), np.float32)
objp[:,:2] = np.mgrid[0:input_corners_height,0:input_corners_width].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (input_corners_height,input_corners_width),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (input_corners_height,input_corners_width), corners2,ret)
        #cv2.imshow('img',img)
        #cv2.waitKey(500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

for fname in images:
    img = cv2.imread(fname)
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite(f'calibresult_{fname}.png',dst)

mean_error = 0
tot_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print ("total error: ", mean_error/len(objpoints))
