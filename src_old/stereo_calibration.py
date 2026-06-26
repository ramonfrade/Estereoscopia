import numpy as np
import cv2 as cv
import glob
import json


######### FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS ##########

chessboardSize = (7, 7)
frameSize = (640, 480)



# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0) ......, (6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)


# Arrays to store object points and image points from all the images
objpoints = []
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.


imagesLeft = sorted(glob.glob('/Dados/Fernando/semana_sfs/estereoscopia/images/stereoLeft/*.png'))
imagesRight = sorted(glob.glob('/Dados/Fernando/semana_sfs/estereoscopia/images/stereoRight/*.png'))

for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if retL and retR:

        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners (opcional — requer DISPLAY)
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('Img Left', imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('Img Right', imgR)
        cv.waitKey(1000)


cv.destroyAllWindows()
print(f"Pares válidos: {len(objpoints)}")
if len(objpoints) == 0:
    raise SystemExit("Nenhum canto detectado. Verifique chessboardSize e as imagens.")

######### CALIBRATION ###########
# Calibrate the camera
retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))



########## Stereo Vision Calibration ##########

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic matrixes so that only Rot, Trans, Emat and Fmat are 
# Hence intrinsic paramaters are the same

criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential Matrix and Fundamental Matrix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

#print(newCameraMatrixL)
#print(newCameraMatrixR)

########## Stereo Rectification #################################################

rectifyScale= 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("Saving parameters!")
cv_file = cv.FileStorage("stereoMap.xml", cv.FILE_STORAGE_WRITE)

cv_file.write("stereoMapL_x", stereoMapL[0])
cv_file.write("stereoMapL_y", stereoMapL[1])
cv_file.write("stereoMapR_x", stereoMapR[0])
cv_file.write("stereoMapR_y", stereoMapR[1])

cv_file.release()

calibration_path = "/Dados/Fernando/semana_sfs/estereoscopia/calibration.json"
calibration_data = {
    "Q": Q.tolist(),
    "frameSize": [int(grayL.shape[1]), int(grayL.shape[0])],
    "baseline_units": float(np.linalg.norm(trans)),
    "physical_baseline_cm": 7.0,
    "chessboardSize": list(chessboardSize),
    "cameras": {
        "calibLeft": 0,
        "calibRight": 2,
        "physicalLeft": 2,
        "physicalRight": 0,
    },
}
with open(calibration_path, "w", encoding="utf-8") as f:
    json.dump(calibration_data, f, indent=2)

# print(f"Calibração salva em {stereo_map_path} e {calibration_path}")
print(f"Calibração salva em {calibration_path}")
print("Stereo calibration complete")
