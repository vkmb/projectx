from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import cgi

class Server(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
    def do_HEAD(self):
        self._set_headers()
        
    # GET sends back a Hello world message
    def do_GET(self):
        self._set_headers()
        self.wfile.write(json.dumps({'hello': 'world', 'received': 'ok'}).encode())
        
    # POST echoes the message adding a JSON field
    def do_POST(self):
        ctype, pdict = cgi.parse_header(self.headers.get('content-type'))
        
        # refuse to receive non-json content
        if ctype != 'application/json':
            self.send_response(400)
            self.end_headers()
            return
            
        # read the message and convert it into a python dictionary
        length = int(self.headers.get('content-length'))
        message = json.loads(self.rfile.read(length))
        print(message)
        
        # add a property to the object, just to mess with data
        messsage={'received':'ok'}
        
        # send the message back
        self._set_headers()
        self.wfile.write(json.dumps(messsage).encode())
        
def run(server_class=HTTPServer, handler_class=Server, port=8008):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    
    print ('Starting httpd on port {}...'.format(port))
    httpd.serve_forever()
    
if __name__ == "__main__":
    from sys import argv
    
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()


[CLOUD_CONFIG]
UID = GSR8uoMXkPMqDRdMnWqlYVLZlBr1
SFP = projectx.json
CAM = GSR8uoMXkPMqDRdMnWqlYVLZlBr1_deviceList
LOG = GSR8uoMXkPMqDRdMnWqlYVLZlBr1_intruderLog
FAD = GSR8uoMXkPMqDRdMnWqlYVLZlBr1_faceData
TFD = GSR8uoMXkPMqDRdMnWqlYVLZlBr1_trainFace
PJN = projectx-10dcf

[APP_CONFIG]
CameraInit = False
SSID = nTalos
PSK = 12345678
CameraName = 9L2avwvW89IYPxM3Omr5 
CameraURL = rtsp://192.168.43.62:8554/video
AppConnected = False
FUP = 1000

