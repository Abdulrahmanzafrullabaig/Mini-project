from flask import Flask, request, render_template, jsonify
import torch
from PIL import Image
import io
import numpy as np
from main import preprocess_image, load_model, make_prediction
import os

app = Flask(__name__)

# Initialize model
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get images from request
        ref_image = request.files['reference_image']
        ver_image = request.files['verification_image']
        
        # Read images
        ref_img = Image.open(io.BytesIO(ref_image.read())).convert('RGB')
        ver_img = Image.open(io.BytesIO(ver_image.read())).convert('RGB')
        
        # Check if images contain signatures
        if np.sum(np.array(ref_img) < 240) < 1000 or np.sum(np.array(ver_img) < 240) < 1000:
            return jsonify({'error': 'One or both images do not contain a signature'})
        
        # Preprocess images
        ref_tensor = preprocess_image(ref_img)
        ver_tensor = preprocess_image(ver_img)
        
        # Make prediction
        prediction = make_prediction(model, ref_tensor, ver_tensor)
        
        result = "Genuine" if prediction > 0.5 else "Forged"
        confidence = float(prediction)
        
        return jsonify({
            'result': result,
            'confidence': f"{confidence:.2%}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)