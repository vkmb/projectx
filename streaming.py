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
import imutils
import platform
import threading

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib

Gst.init()


def running_on_jetson_nano():
    # To make the same code work on a laptop or on a Jetson Nano, we'll detect when we are running on the Nano
    # so that we can access the camera correctly in that case.
    # On a normal Intel laptop, platform.machine() will be "x86_64" instead of "aarch64"
    return platform.machine() == "aarch64"


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
        self.fps = 2
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
        # if self.cap.isOpened():
        _, frame = self.cap.read()
            # if ret:
                # Resize frame of video to 1/4 size for faster face recognition processing
        assert _==True
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

    
    server = GstServer()
    server.connect("client-connected", route2stream, None)
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
