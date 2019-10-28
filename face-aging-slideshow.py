import os
import face_recognition
import cv2
import numpy as np
import pathlib
import json
import concurrent.futures


def convert_heic_photos():
    files = pathlib.Path().glob(photos_path + "/*.heic")



def delete_old_faces():
    files = pathlib.Path().glob(faces_path + "/*.jpg")
    for f in files:
        os.remove(f)


def find_and_store_faces():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        photo_files = pathlib.Path().glob(photos_path + "/IMG_*.*")
        executor.map(save_face_image, photo_files)
    

def get_photo_date_taken(path):
    json_path = path + ".json"
    if os.path.exists(json_path) == False:
        return None

    with open(json_path, 'r') as f:
        dict = json.load(f)
        date_taken = dict["photoTakenTime"]["timestamp"]
    return date_taken


def save_face_image(file):
    if file.suffix.lower() != ".jpg":
        return None

    print("Scanning " + str(file))

    cache_file_path = cache_path + "/" + file.name + ".json"
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r') as f:
            dict = json.load(f)
            face_locations = dict["face_locations"]
            face_landmarks = dict["face_landmarks"]
        if len(face_landmarks) == 1:
            image = face_recognition.load_image_file(str(file))
        else:
            return None
    else:
        # load image and find face locations
        image = face_recognition.load_image_file(str(file))
        face_locations = face_recognition.face_locations(image, model="hog")

        # detect 68-landmarks from image. This includes left eye, right eye, lips, eye brows, nose and chins
        face_landmarks = face_recognition.face_landmarks(image)

        data = {}
        data["face_locations"] = face_locations
        data["face_landmarks"] = face_landmarks
        with open(cache_file_path, 'w') as outfile:
            json.dump(data, outfile)

        # skip if there isn't one face
        if len(face_landmarks) != 1:
            return None

    '''
    Let's find the angle of the face. First calculate 
    the center of left and right eye by using eye landmarks.
    '''
    leftEyePts = face_landmarks[0]['left_eye']
    rightEyePts = face_landmarks[0]['right_eye']

    leftEyeCenter = np.array(leftEyePts).mean(axis=0).astype("int")
    rightEyeCenter = np.array(rightEyePts).mean(axis=0).astype("int")

    leftEyeCenter = (leftEyeCenter[0],leftEyeCenter[1])
    rightEyeCenter = (rightEyeCenter[0],rightEyeCenter[1])

    # draw the circle at centers and line connecting to them
    # cv2.circle(image, leftEyeCenter, 2, (255, 0, 0), 10)
    # cv2.circle(image, rightEyeCenter, 2, (255, 0, 0), 10)
    # cv2.line(image, leftEyeCenter, rightEyeCenter, (255, 0, 0), 10)

    # find the angle of line by using slop of the line.
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # to get the face at the center of the image,
    # set desired left eye location. Right eye location 
    # will be found out by using left eye location.
    # this location is in percentage.
    desiredLeftEye = (0.35, 0.5)
    #Set the cropped image(face) size after rotaion.
    desiredFaceWidth = 256
    desiredFaceHeight = 456

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
    tY = desiredFaceHeight * desiredLeftEye[1] # + 40

    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    (y2,x2,y1,x1) = face_locations[0]

    output = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC)

    # image = cv2.resize(output, size, interpolation = cv2.INTER_CUBIC)

    date_taken = get_photo_date_taken(str(file))
    if date_taken:
        face_path = faces_path + "/" + date_taken + " - " + file.stem + ".jpg"
        cv2.imwrite(face_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))


def write_movie():
    vvw = cv2.VideoWriter(faces_path + "/slideshow.mp4", cv2.VideoWriter_fourcc(*'avc1'), 30, size)

    py = pathlib.Path().glob(faces_path + "/*.jpg")
    for file in py:
        frame = cv2.imread("./" + str(file))

        for x in range(1, 15):
            vvw.write(frame)

    vvw.release()



size = (256, 456)
days_per_min = 365
photos_path = "photos"
cache_path = "cache"
faces_path = "faces"

delete_old_faces()
find_and_store_faces()
write_movie()

