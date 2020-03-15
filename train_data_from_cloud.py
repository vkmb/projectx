"""
# ----------------------------------------------------------------
# [^∆^] --> Created on 10:10 PM FEB 17 2020
# [•∆•] --> Author : Kumm
# [^∆^] --> Tried this https://github.com/PhantomInsights/firebase-python/tree/master/auth/email
# [•∆•] --> As of now the gcloud framework makes more sense
# [^∆^] --> config.ini has been updated to the requirements - Check config
# [•∆•] --> Get train data from % collection firesotre and storage and put in & collection 
# [^∆^] --> Remove data of the trained people
# [•∆•] --> Stop the other processes and do this sysnc it local and cloud 
# [^∆^] --> 
# [•∆•] --> https://googleapis.dev/python/firestore/1.6.2/client.html
# [^∆^] --> 
# [•∆•] --> 
# [^∆^] --> 
# [•∆•] --> 
# ----------------------------------------------------------------
"""

import os
import cv2
import time
import pickle
import signal
import numpy as np
import configparser
import face_recognition
from datetime import datetime, timedelta
from google.cloud import firestore, storage
from google.api_core import datetime_helpers


# collection name = '$user_uid'+ config.ini[FAD]
face_data_template = {
    
    "editedOn" : None,                 # None === Timestamp  
    "faceEncoding" : None,             # blob
    "firstSeen" : None,                # None === Timestamp
    "firstSeenThisInteraction" : None, # None === Timestamp
    "imageUri" : "",                   # firebase storage url
    "label" : "",                      # User set name
    "lastSeen" : None,                 # None === Timestamp
    "seenCount" : 0,                   # Number of times the person has visited totally
    "seenFrames" : 0,                  # Frame count 
    "userId": ""                       # unique id of document in collection
}

# collection name = '$user_uid'+ config.ini[TFD]
train_data_template = {

    "editedOn" : None,                 # None === Timestamp  
    "faceEncoding" : None,             # blob
    "firstSeen" : None,                # None === Timestamp
    "firstSeenThisInteraction" : None, # None === Timestamp
    "imageUri" : "",                   # firebase storage url
    "label" : "",                      # User set name
    "lastSeen" : None,                 # None === Timestamp
    "seenCount" : 0,                   # Number of times the person has visited totally
    "seenFrames" : 0,                  # Frame count 
    "userId": ""                       # unique id of document in collection
}





def quit():
    global trainface_notifier
    trainface_notifier.unsubscribe()


"""
@ Callback function for document changes in TFD
@ Local DB will be updated
@ Function works for change type - document addition  
@ Snapshot issues in 60 minutes : https://github.com/firebase/firebase-admin-python/issues/282

"""

def new_face_added(collection_snapshot, changes, read_time):
    global bucket, facedata_collection, trainface_collection, trainface_notifier
    for change in changes:
        if change.type.name == 'ADDED':
            error = "unknown"
            temp_doc = change.document.to_dict()
            try:
                error = "Dict key"
                temp_file_name = temp_doc["imageUri"]+".jpg"
                # wait for file to be available in storage 
                blob = bucket.get_blob(temp_doc["imageUri"])
                while  blob == None:
                    blob = bucket.get_blob(temp_doc["imageUri"])
                    continue
                else:
                    print("Loaded from resource")
                error = "Image not found "
                blob.download_to_filename(temp_file_name)
                frame = cv2.imread(temp_file_name)
                # cv2 specifics 
                rgb_small_frame = frame[:, :, ::-1]
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                # Check if only once face is present
                if len(face_locations) is 1 and len(face_encodings) is 1:
                    error = "Dict creation "
                    facedata_document_template = {
                        "editedOn" : datetime_helpers.utcnow(), 
                        "faceEncoding" : face_encodings[0].tostring(),
                        "firstSeen" : datetime_helpers.utcnow(),
                        "firstSeenThisInteraction" :  datetime_helpers.utcnow(),
                        "imageUri" : temp_doc["imageUri"],
                        "label" : temp_doc["label"],
                        "lastSeen" :  datetime_helpers.utcnow(), 
                        "seenCount" : 0,
                        "seenFrames" : 0,
                        "userId": change.document.id
                    }
                    error = "Document exsists : Cloud document creation "
                    facedata_collection.add(facedata_document_template, temp_doc["userId"])
                    print("1 face added")
                else:
                    blob.delete()
                
                trainface_collection.document(change.document.id).delete()
                os.remove(temp_file_name)
                print("Done")
            except :
                print("{} error while processing {}".format(error, change.document.id))
                print(change.document.to_dict())
                continue    
        else :
            pass

    print("Last updated at {}\n".format(read_time))

if __name__ == "__main__":
    try:
        debug = True if (sys.argv[-1].split("="))[-1].lower() == "true" else False
        thread_counter = 0
    except:
        pass

    # read firebase config from setup dir

    parser = configparser.ConfigParser()
    configPath = "config.ini"
    try:
        # Loading app configuration from config.ini
        # Requied by gcloud framework
        # Loading respective collection path from config.ini
        # Loading storage path to fetch images
        parser.read(configPath)[-1] == "config.ini"
        filePath = parser["CLOUD_CONFIG"].get("SFP")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(filePath)
        projectName = parser["CLOUD_CONFIG"].get("PJN")
        devId = parser["APP_CONFIG"].get("CameraName")

        # Check internet before doing this 
        db = firestore.Client()
        camera_collection = db.collection(parser["CLOUD_CONFIG"].get("CAM"))
        intruder_collection = db.collection(parser["CLOUD_CONFIG"].get("LOG"))
        facedata_collection = db.collection(parser["CLOUD_CONFIG"].get("FAD"))
        trainface_collection = db.collection(parser["CLOUD_CONFIG"].get("TFD"))

        image_bucket = storage.Client()
        bucket = image_bucket.get_bucket(projectName + ".appspot.com")

        trainface_notifier = trainface_collection.on_snapshot(new_face_added)
        if trainface_notifier._closed:
            trainface_notifier =  trainface_collection.on_snapshot(new_face_added)
        
    except (FileNotFoundError):
        # Throws a exception if there is no file/internet connection
    	print("Error loading config file\n Exiting")
    	exit(0)
