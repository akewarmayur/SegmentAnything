import os
from flask import Flask, render_template, request, redirect, session
from PIL import Image
from flask_session import Session

from ultralytics import FastSAM
from ultralytics.yolo.fastsam import FastSAMPrompt
import time

DEVICE = 'cpu'

# Create a FastSAM model
model = FastSAM('FastSAM.pt')  # or FastSAM-x.pt

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No image uploaded"

    image = request.files['image']
    filename = 'uploaded_image.jpg'  # or generate a unique filename
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    session['coordinates'] = []
    return redirect('/image')


@app.route('/image')
def show_image():
    return render_template('image.html', filename='uploaded_image.jpg')


@app.route('/get_coordinates', methods=['POST'])
def get_coordinates():
    x = int(float(request.form['x']))
    y = int(float(request.form['y']))
    session['coordinates'].append([x, y])
    return ""


@app.route('/flip_image')
def flip_image():
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
    # image = Image.open(image_path)
    # flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    # flipped_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'flipped_image.jpg')
    # flipped_image.save(flipped_image_path)
    coordinates = session['coordinates']
    print(coordinates)

    everything_results = model(image_path,
                               device=DEVICE,
                               retina_masks=True,
                               imgsz=1024,
                               conf=0.4,
                               iou=0.9)
    ee = [e for e in range(len(coordinates))]
    print(ee)
    prompt_process = FastSAMPrompt(image_path, everything_results, device=DEVICE)
    # ann = prompt_process.point_prompt(points=coordinates, pointlabel=[1, 0])
    # coordinates = [[740, 61], [1040, 29]]
    ann = prompt_process.point_prompt(points=coordinates, pointlabel=[1, 0])
    prompt_process.plot(annotations=ann, output='static/output/')

    return render_template('flipped_image.html', filename='output/uploaded_image.jpg', coordinates=coordinates)


if __name__ == '__main__':
    app.run(debug=True)
