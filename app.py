import os
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['YOLO_FOLDER'] = 'yolo_model/'

model_path = os.path.join(app.config['YOLO_FOLDER'], 'best.onnx')  # Thay đổi đuôi tệp thành .onnx
ort_session = ort.InferenceSession(model_path)

def preprocess_image(image):
    # Việc này phụ thuộc vào cách bạn đã huấn luyện mô hình của mình, ví dụ:
    input_shape = (640, 640)  # Sử dụng kích thước input của mô hình
    image = cv2.resize(image, input_shape)
    image = image.transpose(2, 0, 1)  # Chuyển đổi từ HWC sang CHW
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32) / 255.0  # Chuẩn hóa giá trị pixel
    return image

def detect_image(image):
    input_image = preprocess_image(image)
    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_image})
    detections = postprocess_outputs(outputs)
    return detections

def postprocess_outputs(outputs):
    # Xử lý đầu ra của mô hình ONNX để vẽ bounding boxes hoặc thực hiện các thao tác cần thiết
    # Cách xử lý sẽ phụ thuộc vào cấu trúc của output của mô hình ONNX của bạn.
    return outputs  # Điều chỉnh lại tùy theo nhu cầu

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Sử dụng ONNX để detect đối tượng
        image = cv2.imread(filepath)
        detections = detect_image(image)
        cv2.imwrite(filepath, detections)
        
        return redirect(url_for('uploaded_file', filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
