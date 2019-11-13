"""

"""

import os
import cv2
import time
import datetime
import configparser
from google.cloud import firestore, storage
from google.api_core import datetime_helpers

configPath = "config.ini"

# helper functions

parser = configparser.ConfigParser()
if parser.read(configPath)[-1] == "config.ini":
    filePath = parser["CLOUD_CONFIG"].get("SFP")
    projectName = parser["CLOUD_CONFIG"].get("PJN")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(filePath)
    db = firestore.Client()
    camera_collection = db.collection(parser["CLOUD_CONFIG"].get("CAM"))
    intruder_collection = db.collection(parser["CLOUD_CONFIG"].get("LOG"))
    facedata_collection = db.collection(parser["CLOUD_CONFIG"].get("FAD"))

    # Document Templates in firestore
    log_document_template = {
        "ImageURL": None,
        "Labels": [],
        "Timestamp": None,
        "UserID": None,
    }

    facedata_document_template = {
        "FaceEncodingURI": None,
        "FirstSeen": None,
        "ImageURI": None,
        "LastSeen": None,
        "UserSetLables": [],
    }

    # This document alone has to be fetched with the UID
    cameralist_document_template = {"CameraName": [], "CameraURL": []}

    # Bucket to store image files
    image_bucket = storage.Client()
    bucket = image_bucket.get_bucket(projectName + ".appspot.com")
    document = facedata_collection.document()
    print(document.id)

    camera = cv2.VideoCapture(0)
    _, image = camera.read()
    if _:
        currentTime = datetime_helpers.utcnow()
        # reformat it asap
        permFileName = (
            parser["APP_CONFIG"].get("CameraName")
            + "_"
            + str(currentTime).replace(" ", "_").replace(":", "_")
            + ".jpg"
        )
        cv2.imwrite(permFileName, image)
        blob = bucket.blob(permFileName)
        blob.upload_from_filename(permFileName)
        tempIntruderLogDocument = intruder_collection.document(
            parser["APP_CONFIG"].get("CameraName") + " " + str(currentTime)
        )
        # this uri requires authentication to get a public url make the blob public and get the url
        # blob.make_public()
        log_document_template["ImageURL"] = blob.media_link
        log_document_template["Labels"] = ["test", "test2"]
        log_document_template["Timestamp"] = currentTime
        log_document_template["UserID"] = parser["APP_CONFIG"].get("CameraName")
        tempIntruderLogDocument.create(log_document_template)
        os.remove(permFileName)

    camera.release()

    for doc in camera_collection.stream():
        print(u"{} => {}".format(doc.id, doc.to_dict()))

    for doc in intruder_collection.stream():
        print(u"{} => {}".format(doc.id, doc.to_dict()))

    exit(0)
