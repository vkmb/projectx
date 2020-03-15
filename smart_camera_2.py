# ----------------------------------------------------------------
# [^∆^] --> Created on 8:10 PM Nov 7 2019
# [•∆•] --> Author : Kumm & Google
# [^∆^] --> Last edited on 8:10 PM Nov 7 2019
# [•∆•] --> current Functions
# [^∆^] ------> RTSP - No Auth
# [•∆•] ------> Face Recognition
# [^∆^] ------> Local Face Repo
# [•∆•] ------> Offline Survillence
# [^∆^] ------> Integrate with firebase
# [•∆•] ------> NOT DRY JUST KISS
# ----------------------------------------------------------------
import os
import gi
import sys
import copy
import time
import pickle
import signal
import socket
import platform
import threading
import numpy as np
from cv2 import cv2
import configparser
import face_recognition
from datetime import datetime, timedelta
from google.cloud import firestore, storage
from google.api_core import datetime_helpers
from sync_it_up import sync_it_to_local


gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib

Gst.init()
configPath = "config.ini"
thread_counter, accuracy = 0, 0.65
model_options = {"Good":"hog", "Best":"cnn"}
model_mode = "Good"

# collection name = '$user_uid'+ config.ini[FAD]
face_data_template = {
    "editedOn": None,  # None === Timestamp
    "faceEncoding": None,  # blob
    "firstSeen": None,  # None === Timestamp
    "firstSeenThisInteraction": None,  # None === Timestamp
    "imageUri": "",  # firebase storage url
    "label": "",  # User set name
    "lastSeen": None,  # None === Timestamp
    "seenCount": 0,  # Number of times the person has visited totally
    "seenFrames": 0,  # Frame count
    "userId": "",  # unique id of document in collection
}

# collection name = '$user_uid'+ config.ini[TFD]
train_data_template = {
    "editedOn": None,  # None === Timestamp
    "faceEncoding": None,  # blob
    "firstSeen": None,  # None === Timestamp
    "firstSeenThisInteraction": None,  # None === Timestamp
    "imageUri": "",  # firebase storage url
    "label": "",  # User set name
    "lastSeen": None,  # None === Timestamp
    "seenCount": 0,  # Number of times the person has visited totally
    "seenFrames": 0,  # Frame count
    "userId": "",  # unique id of document in collection
}

# collection name = '$user_uid'+ config.ini[LOG]
log_document_template = {
    "imageUri": "",
    "location": "",
    "timestamp": None,
    "peopleDetected": [],
}

# collection name = '$user_uid'+ config.ini[CAM]
cam_document_template = {
    "url": "",
    "label": "",
    "addedOn": None,
    "editedOn": None,
    "lastSeen": None,
    "statusOn": False,
    "uid": "",
    "mode" : "good"
}


# ----------------------------------------------------------------
# Program Quit & Cleanup
def wrap_it_up(p1, p2):
    try:

        save_known_faces()

        global mainloop, db, bucket, facedata_collection, \
            facedata_notifier, trainface_notifier, camera_doc
        
        camera_doc.update({"statusOn":False, "lastSeen": datetime.utcnow()})

        facedata_notifier.unsubscribe()
        trainface_notifier.unsubscribe()
        
        del db
        del bucket
        del facedata_collection
        del facedata_notifier
        del trainface_notifier
        
        mainloop.quit()
    except:
        # mainloop was not created
        pass
    print(
        "\n\n[^∆^] --> It's time to go...\n[•∆•] --> Hope I was a good program\n[^∆^] --> You caused a bug and that's why we are leaving\n"
    )
    quit()


signal.signal(signal.SIGINT, wrap_it_up)
# ----------------------------------------------------------------


def running_on_jetson_nano():
    # To make the same code work on a laptop or on a Jetson Nano, we'll detect when we are running on the Nano
    # so that we can access the camera correctly in that case.
    # On a normal Intel laptop, platform.machine() will be "x86_64" instead of "aarch64"
    return platform.machine() == "aarch64"


def get_jetson_gstreamer_source(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    """
    Return an OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano
    """
    return (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), "
        + f"width=(int){capture_width}, height=(int){capture_height}, "
        + f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        + f"nvvidconv flip-method={flip_method} ! "
        + f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        + "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


def survillence():
    global camera, offline
    while offline:
        currentTime = datetime_helpers.utcnow()
        _, frame = camera.read()
        if _:
            threading.Thread(
                target=face_recog, args=(frame, currentTime), daemon=True
            ).start()


def face_recog(frame, currentTime=None):

    global known_face_encodings, known_face_metadata, thread_counter, debug, \
        log_document_template, db, bucket, devId, frequency, parser, intruder_collection, \
            model_options, model_mode

    temp_log_document = copy.deepcopy(log_document_template)

    permFileName = (
        devId + "_" + str(currentTime).replace(" ", "_").replace(":", "_") + ".jpg"
    )

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Find all the face locations and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame,  model=model_options[model_mode])
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    # Loop through each detected face and see if it is one we have seen before
    # If so, we'll give it a label that we'll draw on top of the video.
    face_labels = []
    doc_id_path = ""
    for face_location, face_encoding in zip(face_locations, face_encodings):
        # See if this face is in our list of known faces.
        metadata = lookup_known_face(face_encoding)
        # If we found the face, label the face with some useful information.
        if metadata is not None:
            face_label = metadata["label"]
            doc_id_path =  "/" + parser["CLOUD_CONFIG"].get("FAD") + "/" + metadata["userId"]

        # If this is a brand new face, add it to our list of known faces
        else:
            face_label = "New visitor!"
            # Grab the image of the the face from the current frame of video
            top, right, bottom, left = face_location
            face_image = small_frame[top:bottom, left:right]
            face_image = cv2.resize(face_image, (150, 150))
            # Add the new face to our known face data
            doc_id_path = (
                "/"
                + parser["CLOUD_CONFIG"].get("FAD")
                + "/"
                + register_new_face(face_encoding, face_image)
            )

        face_labels.append(face_label)
       
        temp_log_document["peopleDetected"].append(doc_id_path)
    # Draw a box around each face and label each face
    for (top, right, bottom, left), face_label in zip(face_locations, face_labels):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
        )
        cv2.putText(
            frame,
            face_label,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            1,
        )

    thread_counter += 1
    if len(temp_log_document["peopleDetected"]) and frequency//2 < thread_counter < frequency:
        thread_counter = frequency
        
    if thread_counter % frequency == 0:
        if debug:
            print()
            print("Total threads completed :", thread_counter)
        thread_counter = 0
        if logging:
            cv2.imwrite(permFileName, frame)
            blob = bucket.blob(permFileName)
            blob.upload_from_filename(permFileName)
            
            tempIntruderLogDocument = intruder_collection.document(devId + " " + str(currentTime))
            temp_log_document["timestamp"] = datetime_helpers.utcnow()
            temp_log_document["imageUri"] = permFileName
            temp_log_document["location"] =  "/"+ parser["CLOUD_CONFIG"].get("CAM") + "/" + devId
            tempIntruderLogDocument.create(temp_log_document)
            del temp_log_document
            del tempIntruderLogDocument
            os.remove(permFileName)

    return frame


def register_new_face(face_encoding, face_image):

    global db, bucket, facedata_collection
    document = facedata_collection.document()

    """
    Add a new person to our list of known faces
    """
    # Add the face encoding to the list of known faces
    known_face_encodings.append(face_encoding)
    # Add a matching dictionary entry to our metadata list.
    # We can use this to keep track of how many times a person has visited, when we last saw them, etc.

    permFileName = document.id + "_faceImage.jpg"
    cv2.imwrite(permFileName, face_image)
    blob = bucket.blob(permFileName)
    blob.upload_from_filename(permFileName)

    temp_facedata_document = copy.deepcopy(face_data_template)
    temp_facedata_document["editedOn"] = datetime_helpers.utcnow()
    temp_facedata_document["firstSeen"] = datetime_helpers.utcnow()
    temp_facedata_document["faceEncoding"] = face_encoding.tostring()
    temp_facedata_document["firstSeenThisInteraction"] = datetime_helpers.utcnow()
    temp_facedata_document["imageUri"] = permFileName
    temp_facedata_document["label"] = "Unknown"
    temp_facedata_document["lastSeen"] = datetime_helpers.utcnow()
    temp_facedata_document["seenCount"] = 1
    temp_facedata_document["seenFrames"] = 1
    temp_facedata_document["userId"] = document.id

    known_face_metadata.append(temp_facedata_document)
    document.create(temp_facedata_document)

    # os.remove(permFileName)

    save_known_faces()

    if debug:
        print("New Face added")

    return document.id


def save_known_faces():
    with open("known_faces.dat", "wb") as face_data_file:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, face_data_file)
        print("Known faces backed up to disk.")


def load_known_faces():

    global known_face_encodings, known_face_metadata
    try:
        with open("known_faces.dat", "rb") as face_data_file:
            known_face_encodings, known_face_metadata = pickle.load(face_data_file)
            print("Known faces loaded from disk.")
    except FileNotFoundError:
        sync_it_to_local()
        print(
            "No previous face data found - starting with a blank known face list.\nTry to sync from cloud with sync_it_up module\n"
        )
        pass


# def camera_meta_changed(document_snapshot, changes, read_time):
#     global camera_doc, model_mode, model_options
#     for change in changes:
#         if change.type.name == "MODIFIED":
#             temp_doc = change.document.to_dict()
#             ip = socket.gethostbyname(socket.gethostname())
#             if not temp_doc["url"].contains(ip):
#                 camera_doc.update({"url":"rtsp://"+ip+":8554/video"})
#             if temp_doc["mode"] != model_mode:
#               model_mode = temp_doc["mode"]


def face_data_changed(collection_snapshot, changes, read_time):

    global db, bucket, known_face_metadata, debug, known_face_encodings

    for change in changes:
        if change.type.name == "MODIFIED":
            temp_doc = change.document.to_dict()
            for doc in known_face_metadata:
                if doc["userId"] == temp_doc["userId"]:
                    doc["label"] = temp_doc["label"]
                    doc["editedOn"] = temp_doc["editedOn"]
                    if debug:
                        print(
                            "Metadata of id {} has been updated".format(
                                temp_doc["userId"]
                            )
                        )

        if change.type.name == "ADDED":
            temp_doc = change.document.to_dict()
            doc_id_list = [doc["userId"] for doc in known_face_metadata]
            if change.document.id in doc_id_list:
                continue
            if len(temp_doc.keys()) < 7:
                continue
            known_face_metadata.append(temp_doc)
            known_face_encodings.append(np.frombuffer(temp_doc["faceEncoding"]))
            if debug:
                print("{} face id added".format(temp_doc["userId"]))

        if change.type.name == "REMOVED":
            temp_doc = change.document.to_dict()
            for index in range(len(known_face_metadata)):
                if (
                    known_face_metadata[index]["faceEncoding"]
                    == temp_doc["faceEncoding"]
                ):
                    known_face_metadata.pop(index)
                    known_face_encodings.pop(index)
            if debug:
                print("{} face id deleted".format(temp_doc["userId"]))

    save_known_faces()


def new_face_added(collection_snapshot, changes, read_time):
    global bucket, facedata_collection, trainface_collection, trainface_notifier, debug, \
        model_options, model_mode
    for change in changes:
        if change.type.name == "ADDED":
            error = "unknown"
            temp_doc = change.document.to_dict()
            try:
                error = "Dict key"
                temp_file_name = temp_doc["imageUri"] + ".jpg"
                # wait for file to be available in storage
                blob = bucket.get_blob(temp_doc["imageUri"])
                while blob == None:
                    blob = bucket.get_blob(temp_doc["imageUri"])
                    continue
                else:
                    if debug:
                        print("Loaded from resource")
                error = "Image not found "
                blob.download_to_filename(temp_file_name)
                frame = cv2.imread(temp_file_name)
                # cv2 specifics
                rgb_small_frame = frame[:, :, ::-1]
                face_locations = face_recognition.face_locations(rgb_small_frame, model=model_options[model_mode])
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations
                )
                # Check if only once face is present
                if len(face_locations) is 1 and len(face_encodings) is 1:
                    error = "Dict creation "
                    facedata_document_template = {
                        "editedOn": datetime_helpers.utcnow(),
                        "faceEncoding": face_encodings[0].tostring(),
                        "firstSeen": datetime_helpers.utcnow(),
                        "firstSeenThisInteraction": datetime_helpers.utcnow(),
                        "imageUri": temp_doc["imageUri"],
                        "label": temp_doc["label"],
                        "lastSeen": datetime_helpers.utcnow(),
                        "seenCount": 0,
                        "seenFrames": 0,
                        "userId": change.document.id,
                    }
                    error = "Document exsists : Cloud document creation "
                    facedata_collection.add(
                        facedata_document_template, temp_doc["userId"]
                    )
                    if debug:
                        print("1 face added")

                else:
                    blob.delete()

                trainface_collection.document(change.document.id).delete()
                # os.remove(temp_file_name)
                print("Done")
            except:
                print("{} error while processing {}".format(error, change.document.id))
                print(change.document.to_dict())
                continue
        else:
            pass

    print("Last updated at {}\n".format(read_time))


def lookup_known_face(face_encoding):
    global facedata_collection, accuracy
    """
    See if this is a face we already have in our face list
    """
    metadata = None

    # If our known face list is empty, just return nothing since we can't possibly have seen this face.
    if len(known_face_encodings) == 0:
        return metadata

    # Calculate the face distance between the unknown face and every face on in our known face list
    # This will return a floating point number between 0.0 and 1.0 for each known face. The smaller the number,
    # the more similar that face was to the unknown face.
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
    best_match_index = np.argmin(face_distances)

    # If the face with the lowest distance had a distance under 0.6, we consider it a face match.
    # 0.6 comes from how the face recognition model was trained. It was trained to make sure pictures
    # of the same person always were less than 0.6 away from each other.
    # Here, we are loosening the threshold a little bit to 0.65 because it is unlikely that two very similar
    # people will come up to the door at the same time.
    if face_distances[best_match_index] < accuracy:
        # If we have a match, look up the metadata we've saved for it (like the first time we saw it, etc)
        metadata = known_face_metadata[best_match_index]

        # Update the metadata for the face so we can keep track of how recently we have seen this face.
        metadata["lastSeen"] = datetime_helpers.utcnow()
        metadata["seenCount"] += 1

        # We'll also keep a total "seen count" that tracks how many times this person has come to the door.
        # But we can say that if we have seen this person within the last 5 minutes, it is still the same
        # visit, not a new visit. But if they go away for awhile and come back, that is a new visit.
        if datetime_helpers.utcnow() - metadata["firstSeenThisInteraction"].replace(
            tzinfo=None
        ) > timedelta(minutes=2):
            metadata["firstSeenThisInteraction"] = datetime_helpers.utcnow()
            metadata["seenCount"] += 1

            tempDoc = facedata_collection.document(metadata["userId"])
            tempDoc.update(
                {
                    "seenCount": metadata["seenCount"],
                    "lastSeen": metadata["lastSeen"],
                    "seenFrames": metadata["seenFrames"],
                    "firstSeenThisInteraction": metadata["firstSeenThisInteraction"],
                }
            )
            del tempDoc

    return metadata


# ------------------------------------------------------------------ #


def route2stream(client, loop, url):
    # ------------------------------------------------------------------ #
    """
  This is callback function which is signaled when the new client    
  is added to the stream. Works for a single client
  """
    # ------------------------------------------------------------------ #
    if debug:
        print("Streaming started")
    global offline
    offline = False


# ------------------------------------------------------------------ #


class SensorFactory(GstRtspServer.RTSPMediaFactory):
    # ------------------------------------------------------------------------------------#
    """
  Gstreamer Pipeline to stream data from camera with face recogniition via rtsp & udp
  """
    # ------------------------------------------------------------------------------------#
    global camera, debug

    def __init__(self, **properties):
        super(SensorFactory, self).__init__(**properties)
        self.cap = camera
        self.number_frames = 0
        self.fps = 15
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = (
            "appsrc name=source is-live=true block=true format=GST_FORMAT_TIME "
            "caps=video/x-raw,format=BGR,width=1280,height=720,framerate={}/1 "
            "! videoconvert ! video/x-raw,format=I420 "
            "! x264enc speed-preset=ultrafast tune=zerolatency "
            "! rtph264pay config-interval=1 name=pay0 pt=96".format(self.fps)
        )

    def on_need_data(self, src, lenght):
        global offline
        if self.cap.isOpened():
            currentTime = datetime_helpers.utcnow()
            ret, frame = self.cap.read()
            if ret:
                # Resize frame of video to 1/4 size for faster face recognition processing
                frame = face_recog(frame, currentTime)
                data = frame.tostring()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                self.number_frames += 1
                retval = src.emit("push-buffer", buf)
                if retval != Gst.FlowReturn.OK and retval == Gst.FlowReturn.FLUSHING:
                    offline = True
                    sur_thread = threading.Thread(target=survillence)
                    sur_thread.start()
                    if debug:
                        print("Going offline, streaming stopped")

    def do_create_element(self, url):
        if debug:
            print("Gstreamer Pipeline Created\nURL : ", url.get_request_uri())
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name(
            "source"
        )  # bin | pipe - series of elements with the launch string --> by definition bin is a element
        appsrc.connect("need-data", self.on_need_data)


# ------------------------------------------------------------------ #


class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(GstServer, self).__init__(**properties)
        self.factory = SensorFactory()
        self.factory.set_shared(True)
        self.get_mount_points().add_factory("/video", self.factory)
        self.attach(None)


# ------------------------------------------------------------------ #
"""
 GstRtspServer.RTSPServer Configuration :

 ip   : 0.0.0.0                              
 port : 8554                                 
 uri  : rtsp://0.0.0.0:8554/video            
 
"""
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
#                          Global Variables                          #
# ------------------------------------------------------------------ #

# To start with debug use the following command
# $ python3 $filename.py debug=true

if __name__ == "__main__":
    try:
        debug = True if (sys.argv[-1].split("="))[-1].lower() == "true" else False
        thread_counter = 0
    except:
        pass

    # read firebase config from setup dir

    parser = configparser.ConfigParser()
    if parser.read(configPath)[-1] == "config.ini":
        # Loading app configuration from config.ini
        filePath = parser["CLOUD_CONFIG"].get("SFP")
        projectName = parser["CLOUD_CONFIG"].get("PJN")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(filePath)
        devId = parser["APP_CONFIG"].get("CameraID")
        # Throws a exception if there is no internet connection
        db = firestore.Client()
        image_bucket = storage.Client()
        bucket = image_bucket.get_bucket(db.project + ".appspot.com")
        frequency = int(parser["APP_CONFIG"].get("FUP"))
        logging = True
    else:
        logging = False

    camera = None
    if running_on_jetson_nano():
        camera = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)

    else:
        camera = cv2.VideoCapture(0)

    if debug:
        if camera is not None:
            deviceSet = True
            print("Camera Initialized")
        else:
            deviceSet = False
            print("Error Initializing Camera..\nExiting")
            exit()

    known_face_encodings = []
    known_face_metadata = []
    load_known_faces()

    facedata_collection = db.collection(parser["CLOUD_CONFIG"].get("FAD"))
    trainface_collection = db.collection(parser["CLOUD_CONFIG"].get("TFD"))
    intruder_collection = db.collection(parser["CLOUD_CONFIG"].get("LOG"))
    camera_doc = db.document(parser["CLOUD_CONFIG"].get("CAM"), parser["APP_CONFIG"].get("CameraID"))
    ip = socket.gethostbyname(socket.gethostname())
    camera_doc.update({"url":"rtsp://"+ip+":8554/video", "statusOn" : True})

    model_mode = camera_doc.get().to_dict()["mode"]

    facedata_notifier = facedata_collection.on_snapshot(face_data_changed)
    trainface_notifier = trainface_collection.on_snapshot(new_face_added)
    # camera_notifier = camera_doc.on_snapshot(camera_meta_changed)

    server = GstServer()
    server.connect("client-connected", route2stream, None)

    offline = True

    sur_thread = threading.Thread(target=survillence, daemon=True)
    sur_thread.start()

    if debug:
        print("RTSP off , Survillence starting")
    mainloop = GLib.MainLoop()
    # ------------------------------------------------------------------ #
    #                          Awesome Loop                              #
    # ------------------------------------------------------------------ #
    mainloop.run()
    # ------------------------------------------------------------------ #
    #                      All the magic happens here                    #
    #                      Event driven with signals.                    #
    # ------------------------------------------------------------------ #
