import os
import face_recognition
import cv2
import numpy as np
import pathlib

size = (640, 480)
vvw = cv2.VideoWriter('slideshow.avi', cv2.VideoWriter_fourcc('X','V','I','D'), 24, size)

py = pathlib.Path().glob("photos/*.jpg")
for file in py:
    print(str(file))
    frame = cv2.imread("./" + str(file))
    resized_frame = cv2.resize(frame, size, interpolation = cv2.INTER_CUBIC)
    for x in range(1, 15):
        vvw.write(resized_frame)
vvw.release()


def get_face_image(path):

    # load image and find face locations.
    image = face_recognition.load_image_file("IMG_0919.jpg")
    face_locations = face_recognition.face_locations(image, model="hog")

    # detect 68-landmarks from image. This includes left eye, right eye, lips, eye brows, nose and chins
    face_landmarks = face_recognition.face_landmarks(image)

    '''
    Let's find and angle of the face. First calculate 
    the center of left and right eye by using eye landmarks.
    '''
    leftEyePts = face_landmarks[0]['left_eye']
    rightEyePts = face_landmarks[0]['right_eye']

    leftEyeCenter = np.array(leftEyePts).mean(axis=0).astype("int")
    rightEyeCenter = np.array(rightEyePts).mean(axis=0).astype("int")

    leftEyeCenter = (leftEyeCenter[0],leftEyeCenter[1])
    rightEyeCenter = (rightEyeCenter[0],rightEyeCenter[1])

    # draw the circle at centers and line connecting to them
    cv2.circle(image, leftEyeCenter, 2, (255, 0, 0), 10)
    cv2.circle(image, rightEyeCenter, 2, (255, 0, 0), 10)
    cv2.line(image, leftEyeCenter, rightEyeCenter, (255,0,0), 10)

    # find and angle of line by using slop of the line.
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # to get the face at the center of the image,
    # set desired left eye location. Right eye location 
    # will be found out by using left eye location.
    # this location is in percentage.
    desiredLeftEye=(0.35, 0.35)
    #Set the croped image(face) size after rotaion.
    desiredFaceWidth = 128
    desiredFaceHeight = 128

    desiredRightEyeX = 1.0 - desiredLeftEye[0]
    
    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
        (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    (y2,x2,y1,x1) = face_locations[0] 
            
    output = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC)

    image = cv2.resize(image, (300, 400), interpolation = cv2.INTER_CUBIC)

    return image