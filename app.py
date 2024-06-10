from flask import Flask, render_template, request, Response, redirect, url_for, flash, send_from_directory
import cv2
import numpy as np
import base64
from werkzeug.utils import secure_filename
import os
import time

app = Flask(__name__)

# Konfigurasi penyimpanan
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
RESULT_IMAGE = 'result_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['RESULT_IMAGE_FOLDER'] = RESULT_IMAGE
app.secret_key = 'supersecretkey'  # Untuk flash messages

# Load Yolo model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {round(confidence, 2)}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame

def generate_frames(video_path, save_path=None):
    cap = cv2.VideoCapture(video_path)
    # Menyiapkan writer video jika save_path diberikan
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    else:
        out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_objects(frame)
        if out:
            out.write(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()
    if out:
        out.release()
    # Hapus video asli setelah selesai
    if os.path.exists(video_path) and save_path:
        os.remove(video_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_extension = filename.rsplit('.', 1)[1].lower()

            if file_extension in {'mp4'}:
                # Simpan file video
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
                file.save(filepath)
                return redirect(url_for('uploaded_video', filename=filename))
            else:
                # Simpan gambar asli ke folder uploads
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Baca gambar untuk deteksi objek
                img = cv2.imread(filepath)
                result_image = detect_objects(img)

                # Encode gambar hasil deteksi untuk ditampilkan
                _, img_encoded = cv2.imencode('.png', result_image)
                img_base64 = base64.b64encode(img_encoded).decode('utf-8')

                return render_template('result.html', img_base64=img_base64, filename=filename)
    return render_template('upload.html')

@app.route('/uploaded_video/<filename>')
def uploaded_video(filename):
    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(generate_frames(video_path, result_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/download_result_image/<filename>')
def download_result_image(filename):
    result_image_path = os.path.join(app.config['RESULT_IMAGE_FOLDER'], filename)
    if os.path.exists(result_image_path):
        return send_from_directory(app.config['RESULT_IMAGE_FOLDER'], filename)
    else:
        return "Image not found", 404


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
    app.run(debug=True)
