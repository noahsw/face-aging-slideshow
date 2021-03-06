import os
import face_recognition
import cv2
import numpy as np
import pathlib
import json
import concurrent.futures
import json_tricks
import sys
import subprocess
import datetime



def convert_heic_photos():
    print("Converting HEIC photos...")

    heic_files = sorted(pathlib.Path().glob(photos_path + "/*.*"))

    #for file in heic_files:
    #    convert_heic_to_jpeg(file)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(convert_heic_to_jpeg, heic_files)
    

def convert_heic_to_jpeg(file):
    if file.suffix.lower() != ".heic":
        return None

    # we have to do some renaming shenanigans 
    # because there might already be a jpg with the same name

    # IMG_0103.HEIC --> IMG_0103.converted.jpg
    # IMG_0103.HEIC.json --> IMG_0103.converted.jpg.json

    # IMG_0103(1).HEIC --> IMG_0103(1).converted.jpg
    # IMG_0103.HEIC(1).json --> IMG_0103.HEIC(1).converted.jpg.json

    heic_path = photos_path + "/" + file.name
    jpeg_path = photos_path + "/" + file.stem + ".converted.jpg"
    if os.path.exists(jpeg_path) == False:
        cmd = ['magick', 'convert', heic_path, jpeg_path]
        subprocess.call(cmd, shell=False)
        print("Converted " + heic_path + " to " + jpeg_path)

        json_path = get_json_path(heic_path)
        if json_path:
            new_json_path = json_path.replace(".json", ".converted.jpg.json")
            new_json_path = new_json_path.replace(".HEIC", "", 1)
            os.rename(json_path, new_json_path)
            print("Renamed " + json_path + " to " + new_json_path)



def delete_old_faces():
    files = pathlib.Path().glob(faces_path + "/*.jpg")
    for f in files:
        os.remove(f)


def calculate_face_encoding(face_path):
    print("Calculating face encoding for " + face_path)
    image = face_recognition.load_image_file(face_path)
    encodings = face_recognition.face_encodings(image)
    if len(encodings):
        return encodings[0]
    else:
        return np.array([])


def get_known_face_encodings(known_face_paths):
    print("Calculating face encodings for primary face...")

    known_encodings = []

    for face_path in known_face_paths:
        encoding = calculate_face_encoding(face_path)
        if encoding.size == 0:
            print("No encodings found in " + face_path)
        else:
            known_encodings.append(encoding)
    '''

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for known_face_path, face_encoding in zip(known_face_paths, executor.map(calculate_face_encoding, known_face_paths)):
            if face_encoding.size == 0:
                print("No encodings found in " + known_face_path)
            else:
                known_encodings.append(face_encoding)
    '''
    
    print("Known encodings: " + str(len(known_encodings)))

    return known_encodings


def find_and_store_faces():
    photo_files = sorted(list(pathlib.Path().glob(photos_path + "/*.*")))

    print("Count of photos to scan: " + str(len(photo_files)))

    results = []
    
    for file in photo_files:
        result = save_face_image(file)
        results.append(result)
    '''

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(save_face_image, photo_files):
            results.append(result)
    '''
    



def get_json_path(photo_path):
    json_path = photo_path + ".json"
    return json_path
    

def get_photo_date_taken(path):
    json_path = get_json_path(path)
    if json_path == False:
        return None

    try:
        with open(json_path, 'r') as f:
            dict = json.load(f)
            date_taken = dict["timestamp"]
        return date_taken
    except:
        print("Could not determine date taken for " + path)
        return None


def save_face_image(file):
    if file.suffix.lower() != ".jpg":
        return None

    print("Scanning " + str(file))

    # Ignore Google Photos collages so we don't get duplicates
    if "-COLLAGE" in file.name:
        print("Ignoring collage in " + str(file))
        return None

    cache_file_path = photos_path + "/" + file.name + ".json"
    if os.path.exists(movie_path) == False:
        print("Cache file doesn't exist at " + cache_file_path)
        return None

    dict = {}
    with open(cache_file_path, 'r') as f:
        dict = json_tricks.loads(f.read())
    if "face_locations" in dict:
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
        face_landmarks = face_recognition.face_landmarks(image, face_locations)

        face_encodings = face_recognition.face_encodings(image, face_locations)

        dict['face_locations'] = face_locations
        dict['face_landmarks'] = face_landmarks
        dict['face_encodings'] = face_encodings
        with open(cache_file_path, 'w') as outfile:
            outfile.write(json_tricks.dumps(dict))

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

    leftEyeCenter = np.array(leftEyePts).mean(axis=0).astype("int")
    rightEyeCenter = np.array(rightEyePts).mean(axis=0).astype("int")

    leftEyeCenter = (leftEyeCenter[0], leftEyeCenter[1])
    rightEyeCenter = (rightEyeCenter[0], rightEyeCenter[1])

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
    tY = desiredFaceHeight * desiredLeftEye[1]

    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    (w, h) = (desiredFaceWidth, desiredFaceHeight)

    transformed_image = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC)

    # recalculate landmarks on transformed image
    face_landmarks = face_recognition.face_landmarks(transformed_image)
    if len(face_landmarks) == 0:
        print("No faces found after transformation in " + str(file))
        return None

    leftEyePts = face_landmarks[0]['left_eye']
    rightEyePts = face_landmarks[0]['right_eye']
    nosePts = face_landmarks[0]['nose_tip']
    topLipPts = face_landmarks[0]['top_lip']
    bottomLipPts = face_landmarks[0]['bottom_lip']

    leftEyeCenter = np.array(leftEyePts).mean(axis=0).astype("int")
    rightEyeCenter = np.array(rightEyePts).mean(axis=0).astype("int")
    noseCenter = np.array(nosePts).mean(axis=0).astype("int")
    topLipCenter = np.array(topLipPts).mean(axis=0).astype("int")
    bottomLipCenter = np.array(bottomLipPts).mean(axis=0).astype("int")

    leftEyeCenter = (leftEyeCenter[0], leftEyeCenter[1])
    rightEyeCenter = (rightEyeCenter[0], rightEyeCenter[1])
    noseCenter = (noseCenter[0], noseCenter[1])
    topLipCenter = (topLipCenter[0], topLipCenter[1])
    bottomLipCenter = (bottomLipCenter[0], bottomLipCenter[1])

    # calculate where mouth is relative to each eye. if centered, face is looking forward
    pose = (rightEyeCenter[0] - bottomLipCenter[0]) / (rightEyeCenter[0] - leftEyeCenter[0])
    if (pose < 0.45 or pose > 0.55):
        print("Face not looking forward in " + str(file))
        return None

    # calculate lip separation as a ratio of distance between lip to nose
    smile = (bottomLipCenter[1] - topLipCenter[1]) / (bottomLipCenter[1] - noseCenter[1])
    if (smile < 0.2):
        print("Face not smiling in " + str(file))
        return None



    date_taken = get_photo_date_taken(str(file))
    if date_taken:
        face_path = faces_path + "/" + date_taken + " - " + file.stem + ".jpg"
        cv2.imwrite(face_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])

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
    print("Frames per photo: " + str(frames_per_photo))

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
        date_time_str = file.name.split(" - ")[0]
        date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%dT%H:%M:%SZ')
        timestamp = int(date_time_obj.strftime("%s"))

        # delete if previous one was taken within 12 hours of this one
        if timestamp - previous_timestamp < 60 * 60 * 12:
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

        date_time_str = file.name.split(" - ")[0]
        date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%dT%H:%M:%SZ')
        timestamp = int(date_time_obj.strftime("%s"))
        if timestamp < earliest_timestamp:
            earliest_timestamp = timestamp
        if timestamp > latest_timestamp:
            latest_timestamp = timestamp

    if photo_count == 0:
        return 0

    timestamp_diff_in_days = 1.0 * (latest_timestamp - earliest_timestamp) / 60 / 60 / 24
    movie_frames_per_second = 30
    days_per_photo = timestamp_diff_in_days / photo_count
    photos_per_sec = days_per_min / days_per_photo / 60

    return int(movie_frames_per_second / photos_per_sec)



size = (384, 512)
days_per_min = 800

album_path = "Leo - ABNe77DQpmdI6ed16wtM004qX3oYZNHmI4EWXua2UWMeUeUtBQauaV3NEGiOqQy5ZSIo5dlRL1Rd"
photos_path = "photos/" + album_path
faces_path = "faces/" + album_path

if not os.path.exists(faces_path):
    os.makedirs(faces_path)
        

'''
convert_heic_photos()
delete_old_faces()
known_face_encodings = get_known_face_encodings([
    # photos_path + "/ABNe77CP-kRcC8sNt_94OLA4OfVKLEH6A6ZyXR4nRq7VlDo6kFBqGZ3gYievZZ0hN_RAOYG-pDc-danwY9Ccgg-IxocaHx0_5w-img-106bw.jpg",
    photos_path + "/ABNe77Bv6bdthudnb76bCeVZCKuTZOgnzvpVWqiZhnXSewWh56h5PZOBlbHQg10KbdTNnSJ2HiqkU04jl-Rtk5vGgC8ujLMIAA-DSC_8323.jpg", # 9 month
    photos_path + "/ABNe77DJKqiHHDzl7h2uXKOobTDbC1nE4vA77DKVbjKvAhyQv0M9RoIkg-nrV9jaDv3AfuN-i7Nb7JiQ-0tLDQxwz_EjyqCSZQ-img-135.jpg",
    photos_path + "/ABNe77Bb7E6g2klyY4h9D2jpjid7pT31NaMiPLw6gQczoHRHxpZLESvtUCOvSoQFrDRgTcad3JQAPVgk8w0Rb9YzOb_znZMV_g-img-110.jpg", # 1 year
    photos_path + "/ABNe77Cq3YsFP5oVHJCooVXKCKcVkRBTCVlpDI43c4angZUMJ9YAe3VuCinlJZZJMvhpINd4JsE3jTcF-IOX1k_UlDTYdBgwtQ-IMG_9419.jpg",
    photos_path + "/ABNe77A9vTKlbvG-y99K39KG5lkBh8fklFUUXMWXzjHx1v7EpkH0qTY_aBnrSHMbNXll6nx3YXrXYlUq_ubHBTwwt4Rz_6w6MQ-img-117.jpg",
    photos_path + "/ABNe77Dx_AS8o7S3tGszmD_Im9ZNK2vUNYPPZJp5L7bcGofpkE246-6wysSRM-aetf0zyULW5FMdUX_77Irna83jhFW1SZfurg-IMG_9375.converted.jpg"
    ])
find_and_store_faces()
'''
remove_photo_clusters()
write_movie()

