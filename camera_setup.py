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
# [^∆^] --> Todo :-
# [•∆•] ------>  Add this file to crontab show that it executes @ 
#                reboot
# [^∆^] --> Know issues :-
# [^∆^] ------>  Auto start not running
# ----------------------------------------------------------------
"""
import re
import os
import cv2
import time
import json
import socket
import datetime
import threading
import configparser
from wireless import Wireless
from http import server as httpserver
from PyAccessPoint import pyaccesspoint
from google.cloud import firestore, storage

configPath = 'config.ini'
html_index = """
<html>
<head>
    <title>Device Registration</title>
    <style>
    </style>
</head>
<body>
    <form action='/' method='POST' enctype='multipart/form-data'>
        Wifi Name : <input required type='text' name='ssid'><br>
        Password : <input required type='password' name='key'><br>
        Camera Name : <input required type='text' name='cam_name'><br>
        Interval of updation : <input required type='number' name='fup' min="300" max="5000" step="50" value="500"><br>
        Service Account File(JSON) : <input required type="file" name='file'><br>
        <input type='submit'>
    </form>
</body>
</html>
"""
html_end="""

<html>
<head>
    <title>Device Registration</title>
    <style>
    </style>
</head>
<body>
    Device Registered !!
</body>
</html>

"""

def end_server():
    global server, access_point
    server.socket.close()
    access_point.stop()
    server.shutdown()
    thread = threading.Thread(target=os.system('python3 camera_setup.py')).start()
    while(thread.is_alive()):
        print(end='.')
    

def write_ini(data_dict, filename='config.ini'):
    try:
    # Save it as 'config.ini'
        global parser, current_ip
        
        file = open(filename, 'w')
        parser.add_section('CLOUD_CONFIG')
        parser.set(parser.sections()[-1], "UID", 'UID')
        parser.set(parser.sections()[-1], "API", '')
        parser.set(parser.sections()[-1], "SFP", data_dict['filename'])
        parser.set(parser.sections()[-1], "CAM", 'Cameras')
        parser.set(parser.sections()[-1], "LOG", 'ImageLogs')
        parser.set(parser.sections()[-1], "FAD", 'FaceData')
        parser.add_section('APP_CONFIG')
        parser.set(parser.sections()[-1], "CameraInit", 'False')
        parser.set(parser.sections()[-1], "SSID", data_dict['ssid'])
        parser.set(parser.sections()[-1], "PSK", data_dict['key'])
        parser.set(parser.sections()[-1], "CameraName", data_dict['cam_name'])
        parser.set(parser.sections()[-1], "CameraURL", 'rtsp://'+current_ip+':8554/video')
        parser.set(parser.sections()[-1], "AppConnected", 'False')
        parser.set(parser.sections()[-1], "FUP", str(data_dict['fup']))
        parser.write(file)
        file.close()
        return True
    except :
        return False
def validate_date(data):
    data_dict = {}
    flag = False
    if len(data) > 0:
        temp = data.split('------') # removes '-------'
        for i in temp:

            temp2 = i.split('Content-Disposition: form-data;')[-1].strip()
            
            if '.json' in temp2:
                data_dict['filename'] = 'projectx.json'
                string = temp2.split('Content-Type: application/json')[-1]
                if len(string) < 0 :
                    return False
                json_string = json.loads(string, strict=False)
                file = open(data_dict['filename'], 'w')
                json.dump(json_string, file)
                file.close()
                continue

            temp2 = temp2.split('name=')
            
            if len(temp2) > 1:
                temp3 = temp2[-1].split('\r\n\r\n')
                # decode("utf8","ignore") added to remove illegal charset
                key = temp3[0].replace('"', '')
                value =  temp3[-1].strip()
                if key == 'ssid' and  2 < len(value) < 32:
                    pass
                elif key == 'key' and 2 < len(value) < 32:
                    pass
                elif key == 'cam_name' and 2 < len(value) < 32:
                    pass
                elif key == 'fup':
                    try:
                        value = int(value)
                        if 300 <=value <= 5000:
                            pass
                    except ValueError:
                        break
                else:
                    break
                data_dict[key] = value
        if len(data_dict.keys()) == 5:
            flag = write_ini(data_dict)

        return flag
    else:
        return flag


class Handler(httpserver.SimpleHTTPRequestHandler):
    global html_index, html_end

    def do_GET(self):
        self.send_response(200)
        self.wfile.write(html_index.encode())

    def do_POST(self):
        length = int(self.headers.get('Content-length', 0))
        data = self.rfile.read(length).decode()
        if validate_date(data):
            self.send_response(200)
            self.wfile.write(html_end.encode())
            threading.Thread(target=end_server, daemon=True).start()

        else:
            self.send_response(200)
            self.wfile.write(html_index.encode())


parser = configparser.ConfigParser()
parser.optionxform = str
current_ip = socket.gethostbyname(socket.gethostname())
port_number = 80

result = parser.read(configPath)
if len(result) == 1:    
    filePath = parser['CLOUD_CONFIG'].get('SFP')
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath(filePath)
    logged_url = parser['APP_CONFIG'].get('CameraURL')
    logged_name = parser['APP_CONFIG'].get('CameraName')
    logged_ip = re.findall('[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+', logged_url, re.ASCII)[-1]
    
    wireless = Wireless()
    if wireless.current() != parser['APP_CONFIG'].get('SSID'):
        wireless.connect(ssid=parser['APP_CONFIG'].get('SSID'), password=parser['APP_CONFIG'].get('PSK'))
        while len(wireless.current()) == 0:
            continue
        # delay added due to dns resolution issues
        time.sleep(10)
    # check if connected to wifi

    db = firestore.Client()
    camera_collection = db.collection(parser['CLOUD_CONFIG'].get('CAM'))
    document = db.document(parser['CLOUD_CONFIG'].get('CAM')+'/'+parser['CLOUD_CONFIG'].get('UID'))
    cam_dict = {}
    for doc in camera_collection.stream():
        cam_dict = doc.to_dict()
        break
    # update ip
    if current_ip != logged_ip:
        if logged_ip in cam_dict.keys():
            cam_dict.pop(logged_url)
            new_url = 'rtsp://'+current_ip+':8555/video' 
            cam_dict.update({new_url : logged_name})
            parser.set('APP_CONFIG', 'CameraURL', new_url)
            file = open('config.ini', 'w')
            parser.write(file)
            file.close()
    # After initialised , the configuration is not updated to the server        
    elif logged_url not in cam_dict.keys():
        document = db.document(parser['CLOUD_CONFIG'].get('CAM')+'/'+parser['CLOUD_CONFIG'].get('UID'))
        cam_dict.update({logged_url : logged_name})
    
    document.set(cam_dict)
        # update ip to cloud and config file
    
else:
    # Start wifi hotspot
    access_point = pyaccesspoint.AccessPoint(ssid='Smart Camera', password='1234567890')
    access_point.start()
    # Starting http server 
    server = httpserver.HTTPServer((current_ip, port_number), Handler)
    server.serve_forever()
    # Get the form data
    
    
# execute smart_camera.py
    


        