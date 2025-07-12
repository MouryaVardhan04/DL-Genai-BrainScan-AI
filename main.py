from flask import Flask, render_template, request, send_from_directory, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import json
from report_generator import MedicalReportGenerator

# create app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# Load the trained model
try:
    model = load_model("models/brain_tumor_model.h5")
    print("Model loaded successfully!")
except:
    print("Warning: Model not found. Using dummy predictions.")
    model = None

# Initialize report generator
report_generator = MedicalReportGenerator()

class_labels = ['pituitary', 'notumor', 'glioma', 'meningioma']

# define upload folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# helper function for prediction
def predict_image(img_path):
    image_size = 128

    # Load and preprocess the image
    img = load_img(img_path, target_size=(image_size, image_size))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    if model is not None:
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions, axis=1)[0]
    else:
        # Dummy values for testing
        predicted_class_index = 1  # 'notumor'
        confidence_score = 0.85

    # Determine the class
    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score, None
    else:
        tumor_type = class_labels[predicted_class_index]
        return f"Tumor: {tumor_type}", confidence_score, tumor_type

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']

        if file:
            # Save the uploaded file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Predict results
            result, confidence, tumor_type = predict_image(filename)
            
            # Generate medical report if tumor is detected
            medical_report = None
            if tumor_type:
                try:
                    # Convert tumor type to proper format for report generation
                    tumor_name_map = {
                        'glioma': 'Glioma',
                        'meningioma': 'Meningioma', 
                        'pituitary': 'Pituitary Adenoma'
                    }
                    tumor_name = tumor_name_map.get(tumor_type, tumor_type.title())
                    confidence_percentage = f"{confidence*100:.1f}%"
                    
                    # Generate comprehensive medical report
                    medical_report = report_generator.create_comprehensive_report(
                        patient_id="web_user",
                        tumor_name=tumor_name,
                        confidence_score=confidence_percentage
                    )
                except Exception as e:
                    print(f"Error generating medical report: {e}")
                    medical_report = None
            
            return render_template('index.html', 
                                result=result, 
                                confidence=f'{confidence*100:.2f}%', 
                                file_path=f'/uploads/{file.filename}',
                                medical_report=medical_report)
    
    return render_template('index.html', result=None)

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """API endpoint for generating medical reports"""
    try:
        data = request.get_json()
        tumor_name = data.get('tumor_name')
        confidence_score = data.get('confidence_score')
        
        if not tumor_name or not confidence_score:
            return jsonify({'error': 'Missing tumor_name or confidence_score'}), 400
        
        # Generate report
        report = report_generator.create_comprehensive_report(
            patient_id="api_user",
            tumor_name=tumor_name,
            confidence_score=confidence_score
        )
        
        return jsonify({'report': report})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def get_uploads_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# python main
if __name__ == '__main__':
    app.run(debug=True, port=5001)