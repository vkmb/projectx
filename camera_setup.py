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
error = """SSID length must be greater than <br>or equal to 3 characters<br>
Password length should be greater than <br>or equal to 3 characters<br>
Json file not found<br>
Camera Name length should be greater than <br>or equal to 3 characters<br>
Update Interval range should<br>between 300 to 5000"""
html_index = """
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Device Registration</title>
    <style>
        
        body{
            background-color: #111111;
            color: #FFFFFF
        }

        label{
            font-family : inherit;
            font-size   : 90%;
            width:auto;
            padding:1px;
        }

        input{
            font-family : inherit;
            font-size   : 90%;
            width: auto;
        }

        table{
            margin:auto;
        }

        td{
            text-align:left;
            vertical-align: middle;
            width:auto;
        }

        .form{
            margin: 2em auto;
            border:1px solid #ccc;
            width:auto;
            border-radius: 10%;
            padding-top:4em;
        }

        .error{
            color: #FF0000;
            margin: auto;
            text-align: center;
            word-wrap: break-word;
        }

        #sub{
            width:5em;
            height:2em;
            padding:.2em;
            color: #00FF00;
            background-color: #000;
            margin-left: 10em;
        }

        #sub:hover,
        #sub:focus {
           background-color: #00FF00;
           color: #000;

        }
    </style>
</head>
<body>
    
    <div class="form">
    <table>
        <form action='/' method='POST' enctype='multipart/form-data'>
            <tr>
                <th colspan="2"><label>Device Registration</label></th> 
            </tr>
            <tr>
                <td></td> 
                <td></td> 
            </tr>
            <tr>
                <td></td> 
                <td></td> 
            </tr>
            <tr>
                <td><label>Wifi Name</label></td> 
                <td><input required type='text' name='ssid'></td> 
            </tr>
            <tr>
                <td><label>Password</label></td> 
                <td><input required type='password' name='key'></td> 
            </tr>
            <tr>
                <td><label>Camera Name</label></td> 
                <td><input required type='text' name='cam_name'></td> 
            </tr>
            <tr>
                <td><label>Interval of updation</label></td> 
                <td><input required type='number' name='fup' min="300" max="5000" step="50" value="500"></td> 
            </tr>
            <tr>
                <td><label>Service Account File(JSON)</label></td>
                <td><input required type="file" name='file'></td>
            </tr>
            <tr>
                <td>&nbsp</td> 
                <td>&nbsp</td> 
            </tr>
            <tr>
                <td colspan="2">
                    <div class="submission">
                    <input type='submit' id='sub' value='Register'>
                    </div>
                </td>
            </tr>
            <tr>
                <td>&nbsp</td> 
                <td>&nbsp</td> 
            </tr>
        </form>
    </table>
    </div>

"""
html_index_end = '<pre><div class="error">{0}</div></pre></body></html>'
html_end="""
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Device Registration</title>
    <style>
        
        body{
            background-color: #111111;
            color: #FFFFFF
        }

        pre{

            color: #99FF55;
            margin: 10em auto;
            text-align: center;
            word-wrap: break-word;
        }

    </style>
</head>
<body>
    <pre>
       Device Registered     
    </pre>
</body>
</html>

"""

def end_server():
    global server, access_point
    server.socket.close()
    access_point.stop()
    server.shutdown()

def write_ini(data_dict, filename='config.ini'):
    try:
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
    global error
    data_dict = {}
    flag = False
    try:
        if len(data) > 0:
            temp = data.split('------')
            for i in temp:
                temp2 = i.split('Content-Disposition: form-data;')[-1].strip()
                if '.json' in temp2:
                    data_dict['filename'] = 'projectx.json'
                    string = temp2.split('Content-Type: application/json')[-1]
                    # need to improve json validatION
                    if len(string) < 0 :
                        break
                    json_string = json.loads(string, strict=False)
                    file = open(data_dict['filename'], 'w')
                    json.dump(json_string, file)
                    file.close()
                    error.replace('Json file not found<br>', '')
                    continue
                temp2 = temp2.split('name=')
                if len(temp2) > 1:
                    temp3 = temp2[-1].split('\r\n\r\n')
                    # decode("utf8","ignore") added to remove illegal charset
                    key = temp3[0].replace('"', '')
                    value =  temp3[-1].strip()
                    if key == 'ssid' and  2 < len(value) < 32:
                        error.replace('SSID length must be greater than <br>or equal to 3 characters<br>', '')
                        pass
                    elif key == 'key' and 2 < len(value) < 32:
                        error.replace('Password length should be greater than <br>or equal to 3 characters<br>', '')
                        pass
                    elif key == 'cam_name' and 2 < len(value) < 32:
                        error.replace('Camera Name length should be greater than <br>or equal to 3 characters<br>', '')
                        pass
                    elif key == 'fup':
                        try:
                            value = int(value)
                            if 300 <=value <= 5000:
                                error.replace('Update Interval range should<br>between 300 to 5000', '')
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
    except:
        error = 'Unsupported File Error'
        return flag

class Handler(httpserver.SimpleHTTPRequestHandler):
    global html_index, html_index_end,  html_end, error

    def do_GET(self):
        self.send_response(200)
        temp = html_index + html_index_end.format('')
        self.wfile.write(temp.encode())

    def do_POST(self):
        length = int(self.headers.get('Content-length', 0))
        data = self.rfile.read(length).decode()
        if validate_date(data):
            self.send_response(200)
            self.wfile.write(html_end.encode())
            threading.Thread(target=end_server, daemon=True).start()

        else:
            self.send_response(200)
            temp = html_index + html_index_end.format(error)
            self.wfile.write(temp.encode())


parser = configparser.ConfigParser()
parser.optionxform = str
current_ip = socket.gethostbyname(socket.gethostname())
port_number = 80

result = parser.read(configPath)
if len(result) == 1:    
    filePath = parser['CLOUD_CONFIG'].get('SFP')
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath(filePath)
    wifi_ssid = parser['APP_CONFIG'].get('SSID')
    logged_url = parser['APP_CONFIG'].get('CameraURL')
    logged_name = parser['APP_CONFIG'].get('CameraName')
    logged_ip = re.findall('[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+', logged_url, re.ASCII)[-1]
    print('Configuration Loaded ...\nChecking Internet connection')
    wireless = Wireless()
    if wireless.current() != wifi_ssid:
        wireless.connect(ssid=wifi_ssid, password=parser['APP_CONFIG'].get('PSK'))
        print('Connecting to {} '.format(wifi_ssid), end='.')
        wifi_connection_init = time.time()
        while wireless.current() != wifi_ssid:
            if time.time() - wifi_connection_init:
                print('Cannot connect to {} \nSetup failed...Program Exitted'.format(wifi_ssid), end='.')
                exit()
            print(end='.')
            continue
        # delay added due to dns resolution issues
        print('Waiting for DNS resolution')
        time.sleep(10)
        # check if connected to wifi
        print('Checking cloud setup..')
        try:
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
                    new_url = 'rtsp://'+current_ip+':8554/video' 
                    cam_dict.update({new_url : logged_name})
                    parser.set('APP_CONFIG', 'CameraURL', new_url)
                    file = open('config.ini', 'w')
                    parser.write(file)
                    file.close()
                    print('Camera metadata has been updated')
            # After initialised , the configuration is not updated to the server        
            elif logged_url not in cam_dict.keys():
                document = db.document(parser['CLOUD_CONFIG'].get('CAM')+'/'+parser['CLOUD_CONFIG'].get('UID'))
                cam_dict.update({logged_url : logged_name})
                print('Camera has been initialised')
            else:
                print('No changes !!!')
            # update ip to cloud and config file
            document.set(cam_dict)
        except:
            print('Cannot connect to cloud .. logging disabled')
else:
    # Start wifi hotspot
    access_point = pyaccesspoint.AccessPoint(ssid='Smart Camera', password='1234567890')
    access_point.start()
    print('Please connect to the hotspot : \nSSID : Smart Camera\nPassword : 1234567890')
    # Starting http server 
    server = httpserver.HTTPServer((current_ip, port_number), Handler)
    server.serve_forever()
    # Get the form data
    os.system('python3 camera_setup.py')
    
    
# execute smart_camera.py
    


        