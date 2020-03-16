# ----------------------------------------------------------------
# [^∆^] --> Created on 8:10 PM Nov 12 2019
# [•∆•] --> Author : Kumm & Google
# [^∆^] --> Last edited on 8:10 PM Nov 12 2019
# [•∆•] --> current Functions
# [^∆^] ------> sync it cloud
# [•∆•] ------> sync it local
# [^∆^] ------> NOT DRY JUST KISS
# Solved issues:
# utc errors in smart_camera program execution on 13/11/2019 22:57
# ----------------------------------------------------------------

import os
import sys
import cv2
import copy
import pickle
import numpy as np
import configparser
from google.cloud import firestore, storage
from google.api_core import datetime_helpers


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


def sync_it_to_local():
    # Updates the known_faces.dat if it exsists
    # Otherwise downloads all the facedata_collection
    # and creates the know_faces.dat
    configPath = "config.ini"
    parser = configparser.ConfigParser()
    result = parser.read(configPath)
    # In a folder, each file should have a unique name, if the result is not null
    # then the file exists otherwise file is not found.
    if len(result) == 1:
        filePath = parser["CLOUD_CONFIG"].get("SFP")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(filePath)
        db = firestore.Client()
        bucket = storage.Client().get_bucket(db.project + ".appspot.com")
        facedata_collection = db.collection(parser["CLOUD_CONFIG"].get("FAD"))
        filename = "known_faces.dat"
        known_face_encodings, known_face_metadata, = [], []

        try:
            with open("known_faces.dat", "rb") as face_data_file:
                known_face_encodings, known_face_metadata = pickle.load(face_data_file)
                print("Known faces loaded from disk.")
        except FileNotFoundError as e:
            pass

        exsistinguserIds = [doc["userId"] for doc in known_face_metadata]
        count = 0

        for doc in facedata_collection.stream():
            tmpdoc = doc.to_dict()
            if doc.id in exsistinguserIds or tmpdoc == {}:
                continue
            else:
                if "faceEncoding" in tmpdoc.keys() and len(tmpdoc["faceEncoding"]) != 0:
                    face_image_blob = bucket.blob(tmpdoc["imageUri"])
                    face_image_blob.download_to_filename(
                        filename=doc.id + "_faceImage.jpg"
                    )
                    image = cv2.imread(doc.id + "_faceImage.jpg")
                    known_face_encodings.append(np.frombuffer(tmpdoc["faceEncoding"]))
                    known_face_metadata.append(tmpdoc)
                    # os.remove(doc.id + "_faceImage.jpg")
                    count += 1

        with open("known_faces.dat", "wb") as face_data_file:
            face_data = [known_face_encodings, known_face_metadata]
            pickle.dump(face_data, face_data_file)
            print("{} faces backed up to disk.".format(count))
            # known_face_metadata
    else:
        raise FileNotFoundError


def sync_it_to_cloud():
    # Updates the known_faces.dat if it exsists
    # Otherwise downloads all the facedata_collection
    # and creates the know_faces.dat
    configPath = "config.ini"
    parser = configparser.ConfigParser()
    result = parser.read(configPath)
    # In a folder, each file should have a unique name, if the result is not null
    # then the file exists otherwise file is not found.

    if len(result) == 1:
        filename = "known_faces.dat"
        filePath = parser["CLOUD_CONFIG"].get("SFP")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(filePath)
        db = firestore.Client()
        bucket = storage.Client().get_bucket(db.project + ".appspot.com")
        facedata_collection = db.collection(parser["CLOUD_CONFIG"].get("FAD"))
        known_face_encodings, known_face_metadata, = [], []
        try:
            with open("known_faces.dat", "rb") as face_data_file:
                known_face_encodings, known_face_metadata = pickle.load(face_data_file)
                print("Known faces loaded from disk.")
        except FileNotFoundError as e:
            print("No local data found to be upload")
            exit(-1)
        exsistinguserIds = [doc["userId"] for doc in known_face_metadata]
        print("Found {} faces to be uploaded".format(len(exsistinguserIds)))
        for docIndex in range(0, len(known_face_encodings)):
            document = facedata_collection.document(
                known_face_metadata[docIndex]["userId"]
            )
            if ".jpg" in known_face_metadata[docIndex]["imageUri"]:
                permFileName = known_face_metadata[docIndex]["imageUri"]
            else:
                permFileName = known_face_metadata[docIndex]["imageUri"] + ".jpg"

            known_face_metadata[docIndex]["imageUri"] = permFileName
            blob = bucket.blob(permFileName)
            blob.upload_from_filename(permFileName)
            document.set(known_face_metadata[docIndex])
    else:
        raise FileNotFoundError


if __name__ == "__main__":
    help_text = "\n1). Type 'python sync_it_up mode=cloud2local' to sync data from cloud to local storage\n2). Type 'python sync_it_up mode=local2cloud' to sync data from local storage to cloud"
    if len(sys.argv) == 2:
        if sys.argv[1] == "mode=cloud2local":
            user_input = input("Type yes to sync data from cloud to local storage : ")
            if user_input.lower() == "yes":
                sync_it_to_local()
            else:
                print("Not syncing with cloud.\nExiting..!")
                exit(0)
        elif sys.argv[1] == "mode=local2cloud":
            user_input = input("Type yes to sync data from local storage to cloud : ")
            if user_input.lower() == "yes":
                sync_it_to_cloud()
            else:
                print("Not syncing with local.\nExiting..!")
                exit(0)
        else:
            print("Invalid option" + help_text)
            exit(0)
    else:
        print("Help :-" + help_text)
        exit(0)
