import cv2
import numpy as np
import datetime
import time
import os
import logic_layer
import multiprocessing as mp
import queue
from flask import Flask, render_template, Response, json, request, jsonify, redirect

# Raspberry Pi camera module (requires picamera package)
from camera_auto_release import Camera

app = Flask(__name__)

frame_queue = mp.Queue()
comand_queue = mp.Queue()
FILTER = None
TRAINING_DATA = "persons"
START_TRAIN = False
NAME = ""
TIME_DELTA = 1
COUNTER = 0
#photo train recog
VIDEO_FEED_MODE = ""
STATE = mp.Value('i', -1)
proc = None

@app.route('/')
def index():
    global proc
    proc = mp.Process(target=logic_layer.main, args=(frame_queue, comand_queue))
    proc.start()
    #print("prm", request.args.to_dict())
    return render_template('home.html')


def frame_source():
    while True:
        frame = frame_queue.get()
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/stream')
def stream():
    return Response(frame_source(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/photo', methods=['POST'])
def photo():
    answer = request.get_json()
    name = answer['name']
    dic = {'aaa': 'hi', 'bbb': 'by'}
    if name != "" and name != "admin":
        if name[0] == logic_layer.CMD_MARKER:
            name = name[1:]
        comand_queue.put(name)
    else:
        dic = {'admin': 'true'}
    return jsonify(result=dic)


@app.route('/admin', methods=['POST'])
def admin():
    os.remove("one.yaml")
    os.remove("one_aux.yaml")
    dic = {'reset': 'true'}
    return jsonify(result=dic)


@app.route('/train', methods=['POST'])
def train():
    answer = request.get_json()
    password = answer['password']
    dic = {'itisadmin': 'false'}
    if password == "yaadmin":
        dic = {'itisadmin': 'true'}
    else:
        comand_queue.put(logic_layer.CMD_MARKER)
    return jsonify(result=dic)


if __name__ == '__main__':
    #app.debug = False
    app.run(host='0.0.0.0', port=80, threaded=True)
