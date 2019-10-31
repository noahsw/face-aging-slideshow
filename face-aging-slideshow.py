import os
import face_recognition
import cv2
import numpy as np
import pathlib
import json
import concurrent.futures
import json_tricks
import sys


def convert_heic_photos():
    files = pathlib.Path().glob(photos_path + "/*.heic")



def delete_old_faces():
    files = pathlib.Path().glob(faces_path + "/*.jpg")
    for f in files:
        os.remove(f)


def get_known_face_encodings(known_face_paths):
    known_encodings = []
    for known_face_path in known_face_paths:
        image = face_recognition.load_image_file(known_face_path)
        known_encodings.append(face_recognition.face_encodings(image)[0])
    print(known_encodings)
    return known_encodings


def find_and_store_faces():
    photo_files = pathlib.Path().glob(photos_path + "/IMG_*.*")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(save_face_image, photo_files)
    

def get_photo_date_taken(path):
    json_path = path + ".json"
    if os.path.exists(json_path) == False:
        # Google does some funky numbering when there are dupes
        # original: IMG_0906(1).JPG
        # expected: IMG_0906(1).JPG.json
        # actual: IMG_0906.JPG(1).json
        i = 1
        for i in range(1, 10):
            if json_path.find("(" + str(i) + ")"):
                json_path = json_path.replace("(" + str(i) + ")", "").replace(".json", "(" + str(i) + ").json")
                break
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
            dict = json_tricks.loads(f.read())
        face_locations = dict["face_locations"]
        face_landmarks = dict["face_landmarks"]
        face_encodings = dict["face_encodings"]
        if len(face_landmarks) > 0:
            image = face_recognition.load_image_file(str(file))
        else:
            print("No faces in " + str(file) + " (cached)")
            return None
    else:
        # load image and find face locations
        try:
            image = face_recognition.load_image_file(str(file))
        except:
            print("Error opening " + str(file))
            return None

        face_locations = face_recognition.face_locations(image, model="hog")

        # detect 68-landmarks from image. This includes left eye, right eye, lips, eye brows, nose and chins
        face_landmarks = face_recognition.face_landmarks(image)

        face_encodings = face_recognition.face_encodings(image, face_locations)

        data = {}
        data["face_locations"] = face_locations
        data["face_landmarks"] = face_landmarks
        data["face_encodings"] = face_encodings
        with open(cache_file_path, 'w') as outfile:
            outfile.write(json_tricks.dumps(data))

    if len(face_encodings) == 0:
        print("No faces found in " + str(file))
        return None

    face_index = -1
    for index in range(len(face_encodings)):
        unknown_face_encoding = face_encodings[index]
        # Now we can see the two face encodings are of the same person with `compare_faces`!
        results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding, tolerance=0.3)
        # if an unknown face matches against any known face, return that unknown face
        for result_index in range(len(results)):
            if results[result_index] == True:
                face_index = index
                break
        if face_index > -1:
            break

    if face_index == -1:
        print("No face match in " + str(file))
        return None

    '''
    Let's find the angle of the face. First calculate 
    the center of left and right eye by using eye landmarks.
    '''
    leftEyePts = face_landmarks[face_index]['left_eye']
    rightEyePts = face_landmarks[face_index]['right_eye']
    nosePts = face_landmarks[face_index]['nose_tip']
    topLipPts = face_landmarks[face_index]['top_lip']
    bottomLipPts = face_landmarks[face_index]['bottom_lip']

    leftEyeCenter = np.array(leftEyePts).mean(axis=0).astype("int")
    rightEyeCenter = np.array(rightEyePts).mean(axis=0).astype("int")
    noseCenter = np.array(nosePts).mean(axis=0).astype("int")
    topLipCenter = np.array(topLipPts).mean(axis=0).astype("int")
    bottomLipCenter = np.array(bottomLipPts).mean(axis=0).astype("int")

    # calculate where nose is relative to each eye. if centered, face is looking forward
    pose = (rightEyeCenter[0] - noseCenter[0]) / (rightEyeCenter[0] - leftEyeCenter[0])
    if (pose < 0.4 or pose > 0.6):
        print("Face not looking forward")
        return None

    # calculate lip separation as a ratio of distance between lip to nose
    smile = (bottomLipCenter[1] - topLipCenter[1]) / (bottomLipCenter[1] - noseCenter[1])
    if (smile < 0.2):
        print("Face not smiling")
        return None

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
    (desiredFaceWidth, desiredFaceHeight) = size

    desiredRightEyeX = 1.0 - desiredLeftEye[0]
    
    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    if scale > 1:
        print("Face is too small")
        return None

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

    output = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC)

    date_taken = get_photo_date_taken(str(file))
    if date_taken:
        face_path = faces_path + "/" + date_taken + " - " + file.stem + ".jpg"
        cv2.imwrite(face_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        print("Saved face from " + str(file))
        return face_path
    else:
        print("No date taken for " + str(file))


def write_movie():
    movie_path = faces_path + "/slideshow.mp4"
    if os.path.exists(movie_path):
        os.remove(movie_path)

    vvw = cv2.VideoWriter(movie_path, cv2.VideoWriter_fourcc(*'avc1'), 30, size)

    frames_per_photo = get_frames_per_photo()

    py = pathlib.Path().glob(faces_path + "/*.jpg")
    for file in sorted(py):
        frame = cv2.imread("./" + str(file))

        for x in range(1, frames_per_photo):
            vvw.write(frame)

    vvw.release()


def remove_photo_clusters():
    previous_timestamp = 0
    remove_count = 0

    py = pathlib.Path().glob(faces_path + "/*.jpg")
    for file in sorted(py):
        timestamp = int(file.name.split(" - ")[0])
        # delete if previous one was taken within a minute of this one
        if timestamp - previous_timestamp < 60:
            file.unlink()
            remove_count += 1
        previous_timestamp = timestamp

    print("Photo clusters removed: " + str(remove_count))


def get_frames_per_photo():
    earliest_timestamp = sys.maxsize
    latest_timestamp = 0
    photo_count = 0

    py = pathlib.Path().glob(faces_path + "/*.jpg")
    for file in py:
        photo_count += 1
        timestamp = int(file.name.split(" - ")[0])
        if timestamp < earliest_timestamp:
            earliest_timestamp = timestamp
        if timestamp > latest_timestamp:
            latest_timestamp = timestamp

    timestamp_diff_in_days = 1.0 * (latest_timestamp - earliest_timestamp) / 60 / 60 / 24
    movie_frames_per_second = 30
    days_per_photo = timestamp_diff_in_days / photo_count
    photos_per_sec = days_per_min / days_per_photo / 60

    return int(movie_frames_per_second / photos_per_sec)



size = (256, 356)
days_per_min = 180
photos_path = "photos"
cache_path = "cache"
faces_path = "faces"

delete_old_faces()
known_face_encodings = get_known_face_encodings(["photos/IMG_4751.jpg"])
known_face_encodings = []
find_and_store_faces()
write_movie()

