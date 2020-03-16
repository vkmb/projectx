"""
# ----------------------------------------------------------------
# [^∆^] --> Created on 8:10 PM Nov 12 2019
# [•∆•] --> Author : Kumm 
# [^∆^] --> Done :-
# [•∆•] ------>  httpserver
# [^∆^] ------>  config.ini file initialiser
# [•∆•] ------>  Google Service Account Json fetcher
# [•∆•] ------>  wifi hostspot creator - pyaccesspoint lib 
# [^∆^] ------>  Connect to wifi 
# [•∆•] ------>  Add CSS to registration html page
# [•∆•] ------>  default ssid ,  password = Smart Camera, 1234567890
# [^∆^] --> Todo :-
# [•∆•] ------>  Add this file to crontab show that it executes @ 
#                reboot
# [^∆^] --> Solved issues :-
# [^∆^] ------>  Auto reloading configuration
# [^∆^] --> Known issues :-
# [^∆^] ------>  Hardware support is not readily available for 
#                edge device 
# ----------------------------------------------------------------
"""
import re
import os
import cv2
import cgi
import time
import json
import socket
import datetime
import threading
import configparser
from wireless import Wireless
from datetime import datetime
from google.api_core import datetime_helpers
from http.server import BaseHTTPRequestHandler, HTTPServer
from PyAccessPoint import pyaccesspoint
from google.cloud import firestore, storage

configPath = "config.ini"
ip = None


def end_server():
    global server  # , access_point
    server.socket.close()
    # access_point.stop()
    server.shutdown()


def write_ini(data_dict, filename="config.ini"):
    try:
        global parser, current_ip
        # config file should have sender_mail[SMID], sender_pass[SMPK],
        # projectname[PJN], service_account_file_path[SFP],
        # Hotspot_ssid[H_SSID], Hotspot_pass[H_PSK]
        file = open(filename, "w+")
        parser.add_section("CLOUD_CONFIG")
        parser.set(parser.sections()[-1], "UID", data_dict["uid"])
        parser.set(parser.sections()[-1], "CAM", data_dict["uid"] + "_deviceList")
        parser.set(parser.sections()[-1], "LOG", data_dict["uid"] + "_intruderLog")
        parser.set(parser.sections()[-1], "FAD", data_dict["uid"] + "_faceData")
        parser.set(parser.sections()[-1], "TFD", data_dict["uid"] + "_trainFace")

        parser.add_section("APP_CONFIG")
        parser.set(parser.sections()[-1], "CameraInit", "False")
        parser.set(parser.sections()[-1], "SSID", data_dict["wifiName"])
        parser.set(parser.sections()[-1], "PSK", data_dict["wifiPass"])
        parser.set(parser.sections()[-1], "H_SSID", data_dict["devWifiName"])
        parser.set(parser.sections()[-1], "H_PSK", data_dict["devWifiPass"])
        parser.set(parser.sections()[-1], "CameraName", data_dict["cameraName"])
        parser.set(parser.sections()[-1], "CameraID", "")
        parser.set(
            parser.sections()[-1], "CameraURL", "rtsp://" + current_ip + ":8554/video"
        )
        parser.set(parser.sections()[-1], "mode", data_dict["mode"])
        parser.set(parser.sections()[-1], "AppConnected", "true")
        parser.set(parser.sections()[-1], "RMID", data_dict["rev_mailid"])
        parser.set(parser.sections()[-1], "FUP", data_dict["fps"])
        parser.write(file)
        file.close()
        return True
    except:
        return False


class Handler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    # GET sends back a Hello world message
    def do_GET(self):
        self._set_headers()
        self.wfile.write(json.dumps({"hello": "world", "received": "ok"}).encode())

    def do_POST(self):
        ctype, pdict = cgi.parse_header(self.headers.get("content-type"))

        # refuse to receive non-json content
        if ctype != "application/json":
            self.send_response(400)
            self.end_headers()
            return

        # read the message and convert it into a python dictionary
        length = int(self.headers.get("content-length"))
        message = json.loads(self.rfile.read(length))

        if write_ini(message):
            self._set_headers()
            messsage = {"received": "ok"}
            self.wfile.write(json.dumps(messsage).encode())
            threading.Thread(target=end_server, daemon=True).start()
            return

        self.send_response(400)
        self.end_headers()
        return


parser = configparser.ConfigParser()
parser.optionxform = str
portal = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
port_number = 80

result = parser.read(configPath)
if len(result) == 1 and parser["APP_CONFIG"].get("AppConnected") == "True":

    filePath = parser["CLOUD_CONFIG"].get("SFP")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(filePath)
    wifi_ssid = parser["APP_CONFIG"].get("SSID")
    logged_url = parser["APP_CONFIG"].get("CameraURL")
    logged_name = parser["APP_CONFIG"].get("CameraName")

    print("Configuration Loaded ...\nChecking Internet connection")

    wireless = Wireless()
    if wireless.current() != wifi_ssid:
        wireless.connect(ssid=wifi_ssid, password=parser["APP_CONFIG"].get("PSK"))
        print("Connecting to {} ".format(wifi_ssid), end=".")
        wifi_connection_init = time.time()
        while wireless.current() != wifi_ssid:
            if time.time() - wifi_connection_init:
                print(
                    "Cannot connect to {} \nSetup failed...Program Exitted".format(
                        wifi_ssid
                    ),
                    end=".",
                )
                exit()
            print(end=".")
            continue
    # delay added due to dns resolution issues
    print("Waiting for DNS resolution")
    time.sleep(10)
    # check if connected to wifi
    print("Checking cloud setup..")
    portal.connect(("8.8.8.8", 80))
    current_ip = portal.getsockname()[0]
    db = firestore.Client()
    if len(parser["APP_CONFIG"].get("CameraID").strip()) != 20:
        col = db.collection(parser["CLOUD_CONFIG"].get("CAM"))
        doc_ref = col.document()
        parser.set("APP_CONFIG", "CameraID", doc_ref.id)
        cam_document_template = {
            "url": "rtsp://" + current_ip + ":8554/video",
            "label": parser["APP_CONFIG"].get("CameraName"),
            "addedOn": datetime.now(),
            "editedOn": datetime.now(),
            "lastSeen": None,
            "statusOn": True,
            "uid": doc_ref.id,
            "mode": parser["APP_CONFIG"].get("mode"),
        }

        expected_result = doc_ref.set(cam_document_template)

        while not expected_result.IsInitialized():
            expected_result = doc_ref.set(cam_document_template)

    parser.set("APP_CONFIG", "CameraURL", "rtsp://" + current_ip + ":8554/video")
    with open("config.ini", "w") as file:
        parser.write(file)

    print("Starting smart camera")
    os.system("python3 smart_camera.py")

elif len(result) == 1 and parser["APP_CONFIG"].get("AppConnected") == "False":
    # Start wifi hotspot
    access_point = pyaccesspoint.AccessPoint(
        ssid=parser["APP_CONFIG"].get("H_SSID"),
        password=parser["APP_CONFIG"].get("H_PSK"),
    )
    access_point.start()
    print(
        "Please connect to the hotspot : \nSSID : Smart Camera\nPassword : 1234567890"
    )
    # Starting http server
    server_address = ("", port_number)
    server = HTTPServer(server_address, Handler)
    server.serve_forever()
    # Get the form data
    os.system("python3 camera_setup.py")

else:
    print("Device Corrupted")
    exit()

# execute smart_camera.py
