import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

UPLOAD_FOLDER = os.getcwd() + '/static/uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
img_height = 180
img_width = 180

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def predict():
    if 'file' not in request.files:
        return "No file part."

    file = request.files['file']
    if file.filename == '':
        return "No selected file."
        
    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.png')
    file.save(file_path)

    try:
        # Load the trained model
        model = load_model('model.h5')

        # Preprocess the uploaded image for prediction
        img = image.load_img(file_path, target_size=(img_height, img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values

        # Perform prediction
        prediction = model.predict(img_array)
        prediction_label = "Knock Knees" if prediction[0][0] > 0 else "Normal"

        return render_template('index.html', img_name='temp.png', status=prediction_label)

    except Exception as e:
        return "Error: {}".format(str(e))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
