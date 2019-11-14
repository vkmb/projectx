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
import cv2
import time
import pickle
import signal
import platform
import threading
import numpy as np
import configparser
import face_recognition
from datetime import datetime, timedelta
from google.cloud import firestore, storage
from google.api_core import datetime_helpers

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib

Gst.init()
configPath = "config.ini"

# ----------------------------------------------------------------
# Program Quit & Cleanup
def wrap_it_up(p1, p2):
    try:
        save_known_faces()
        global mainloop, db, bucket, facedata_collection_updater, facedata_collection
        facedata_collection_updater.unsubscribe()
        del db
        del bucket
        del facedata_collection
        del facedata_collection_updater
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

    global known_face_encodings, known_face_metadata, thread_counter, debug, db, bucket, devId, frequency
    log_document_template = {
        "ImageURL": None,
        "Labels": [],
        "Timestamp": currentTime,
        "UserID": None,
    }
    permFileName = (
        devId + "_" + str(currentTime).replace(" ", "_").replace(":", "_") + ".jpg"
    )

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Find all the face locations and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    # Loop through each detected face and see if it is one we have seen before
    # If so, we'll give it a label that we'll draw on top of the video.
    face_labels = []
    for face_location, face_encoding in zip(face_locations, face_encodings):
        # See if this face is in our list of known faces.
        metadata = lookup_known_face(face_encoding)
        # If we found the face, label the face with some useful information.
        if metadata is not None:
            time_at_door = datetime_helpers.utcnow() - metadata[
                "first_seen_this_interaction"
            ].replace(tzinfo=None)
            face_label = f"At door {int(time_at_door.total_seconds())}s"
            face_label = metadata["name"]
        # If this is a brand new face, add it to our list of known faces
        else:
            face_label = "New visitor!"
            # Grab the image of the the face from the current frame of video
            top, right, bottom, left = face_location
            face_image = small_frame[top:bottom, left:right]
            face_image = cv2.resize(face_image, (150, 150))
            # Add the new face to our known face data
            register_new_face(face_encoding, face_image)

        face_labels.append(face_label)
        log_document_template["Labels"].append(face_label)
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

    if thread_counter % frequency == 0:
        thread_counter = 0
        if debug:
            print("Total threads completed :", thread_counter)

        if logging:
            cv2.imwrite(permFileName, frame)
            blob = bucket.blob(permFileName)
            blob.upload_from_filename(permFileName)
            intruder_collection = db.collection(parser["CLOUD_CONFIG"].get("LOG"))
            tempIntruderLogDocument = intruder_collection.document(
                devId + " " + str(currentTime)
            )
            log_document_template["ImageURL"] = blob.media_link
            log_document_template["UserID"] = devId
            tempIntruderLogDocument.create(log_document_template)
            os.remove(permFileName)
            del log_document_template
            del tempIntruderLogDocument
            del intruder_collection

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
    known_face_metadata.append(
        {
            "first_seen": datetime_helpers.utcnow(),
            "last_seen": datetime_helpers.utcnow(),
            "first_seen_this_interaction": datetime_helpers.utcnow(),
            "seen_count": 1,
            "seen_frames": 1,
            "face_image": face_image,
            "name": "Unknown",
            "uuid": document.id,
        }
    )

    permFileName = document.id + "_faceImage.jpg"
    cv2.imwrite(permFileName, face_image)
    blob = bucket.blob(permFileName)
    blob.upload_from_filename(permFileName)

    facedata_document_template = {
        "FaceEncodingString": face_encoding.tostring(),
        "FirstSeen": datetime_helpers.utcnow(),
        "ImageURI": blob.media_link,
        "FirstSeenThisInteraction": datetime_helpers.utcnow(),
        "LastSeen": datetime_helpers.utcnow(),
        "UserSetLabels": ["Unknown"],
        "SeenCount": 1,
        "SeenFrames": 1,
    }

    document.create(facedata_document_template)
    os.remove(permFileName)
    save_known_faces()

    if debug:
        print("New Face added")


def save_known_faces():
    with open("known_faces.dat", "wb") as face_data_file:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, face_data_file)
        print("Known faces backed up to disk.")


def load_known_faces():
    global known_face_encodings, known_face_metadata, downloading

    try:
        with open("known_faces.dat", "rb") as face_data_file:
            known_face_encodings, known_face_metadata = pickle.load(face_data_file)
            print("Known faces loaded from disk.")
    except FileNotFoundError as e:
        print(
            "No previous face data found - starting with a blank known face list.\nTry to sync from cloud with sync_it_up module\n"
        )
        pass


def update_metadata(collection_snapshot, changes, read_time):

    global db, bucket, known_face_metadata, debug

    for docIndex in range(len(collection_snapshot)):
        cloudSnapshotID = collection_snapshot[docIndex].id
        cloudSnapshot = collection_snapshot[docIndex].to_dict()
        for docId in range(len(known_face_metadata)):
            if known_face_metadata[docId]["uuid"] == cloudSnapshotID:
                known_face_metadata[docId]["name"] = cloudSnapshot["UserSetLabels"][0]
                if debug:
                    print("Metadata of id {} has been updated".format(cloudSnapshotID))

    save_known_faces()


def lookup_known_face(face_encoding):
    global facedata_collection
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
    if face_distances[best_match_index] < 0.65:
        # If we have a match, look up the metadata we've saved for it (like the first time we saw it, etc)
        metadata = known_face_metadata[best_match_index]

        # Update the metadata for the face so we can keep track of how recently we have seen this face.
        metadata["last_seen"] = datetime_helpers.utcnow()
        metadata["seen_frames"] += 1

        # We'll also keep a total "seen count" that tracks how many times this person has come to the door.
        # But we can say that if we have seen this person within the last 5 minutes, it is still the same
        # visit, not a new visit. But if they go away for awhile and come back, that is a new visit.
        if datetime_helpers.utcnow() - metadata["first_seen_this_interaction"].replace(
            tzinfo=None
        ) > timedelta(minutes=2):
            metadata["first_seen_this_interaction"] = datetime_helpers.utcnow()
            metadata["seen_count"] += 1

            tempDoc = facedata_collection.document(metadata["uuid"])
            tempDoc.update(
                {
                    "SeenCount": metadata["seen_count"],
                    "LastSeen": metadata["last_seen"],
                    "SeenFrames": metadata["seen_frames"],
                    "FirstSeenThisInteraction": metadata["first_seen_this_interaction"],
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
        # Throws a exception if there is no internet connection
        devId = parser["APP_CONFIG"].get("CameraName")
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
    facedata_collection_updater = facedata_collection.on_snapshot(update_metadata)
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
