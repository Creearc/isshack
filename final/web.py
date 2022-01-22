import os
import threading
import time
import zmq
import numpy as np
import cv2

import pickle

from flask import Response
from flask import Flask
from flask import render_template

from flask import request, redirect, url_for
from flask import send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'tmp'

@app.route("/worker")
def index():
  return render_template("worker.html")


@app.route("/video_feed")
def video_feed():
  return Response(generate(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")

def generate():
  global socket
  while True:
    try:
      socket.send_string('img$0', zmq.NOBLOCK)
      msg = socket.recv()
    except Exception as e:
      print(e)
      continue
    
    img = pickle.loads(msg)
    if img is None:
      time.sleep(0.1)
      continue
    (flag, encodedImage) = cv2.imencode(".jpg", img)
    if not flag:
      time.sleep(0.1)
      continue
    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encodedImage) + b'\r\n')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            saved_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(saved_file)
            r = filename.split('.')
            print(r)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                      '{}_n.{}'.format(r[0], r[1]))
            
            detect_image(saved_file, image_path)
            
            out = image_path.split('/')[-1]
            return redirect(url_for('worker'))
    return '''
    <!doctype html>
    <title>upload_and_download_files</title>
    <h1>Загрузите файл</h1>
    <h3>Принимаются видео в формате mp4</h3>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Загрузить>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):  
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
  lock = threading.Lock()

  img = None

  ip = '127.0.0.1'

  context = zmq.Context()
  socket = context.socket(zmq.REQ)
  socket.connect("tcp://{}:{}".format(ip, 5001))
  
  app.run(host='0.0.0.0', port=58800, debug=False, threaded=True, use_reloader=False)
