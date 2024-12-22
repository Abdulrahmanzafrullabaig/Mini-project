from flask import Flask, render_template, request, jsonify
import os
from main import predict_signatures
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Both images are required'}), 400
    
    image1 = request.files['image1']
    image2 = request.files['image2']
    
    if image1.filename == '' or image2.filename == '':
        return jsonify({'error': 'No selected files'}), 400
    
    if not (allowed_file(image1.filename) and allowed_file(image2.filename)):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save images
        filename1 = secure_filename(image1.filename)
        filename2 = secure_filename(image2.filename)
        
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        
        image1.save(filepath1)
        image2.save(filepath2)
        
        # Get prediction
        prediction, confidence = predict_signatures(filepath1, filepath2)
        
        # Clean up uploaded files
        os.remove(filepath1)
        os.remove(filepath2)
        
        return jsonify({
            'prediction': 'Genuine' if prediction else 'Forged',
            'confidence': float(confidence)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)