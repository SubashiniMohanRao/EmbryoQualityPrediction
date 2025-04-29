import os
import sys
import json
import glob
import pandas as pd
import numpy as np
import uuid
import base64
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename

# Add project root to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

# Import evaluation and prediction modules
from src.predict_image import EmbryoPredictor
from src.db_utils import save_prediction, get_predictions_by_patient, get_unique_patients
from src.xai_utils import generate_xai_visualization
from src.train_model import get_transforms

app = Flask(__name__)
app.secret_key = 'embryo_quality_prediction_secret_key'

# Configuration class
class AppConfig:
    # Directory paths
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
    PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
    UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
    PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure all required directories exist."""
        for dir_path in [cls.MODELS_DIR, cls.RESULTS_DIR, cls.PLOTS_DIR, cls.UPLOAD_DIR, cls.PREDICTIONS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def allowed_file(cls, filename):
        """Check if file has an allowed extension."""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in cls.ALLOWED_EXTENSIONS

# Ensure all required directories exist
AppConfig.ensure_dirs()

# Routes for prediction reports
@app.route('/reports', methods=['GET'])
def prediction_reports():
    """Show prediction reports with filtering options."""
    # Get filter parameters
    patient_name = request.args.get('patient_name', '')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    
    # Get database configuration
    db_config = {
        'host': 'localhost',
        'user': 'suba',
        'password': 'Suba@123',
        'database': 'embryo_predictions'
    }
    
    # Get unique patients for the dropdown
    try:
        patients = get_unique_patients(db_config)
    except Exception as e:
        flash(f"Error retrieving patient list: {str(e)}", "danger")
        patients = []
    
    # Get predictions based on filters
    try:
        # If patient name is provided, filter by patient
        if patient_name:
            predictions = get_predictions_by_patient(patient_name, db_config)
        else:
            predictions = get_predictions_by_patient(None, db_config)
        
        # Apply date filters if provided
        if date_from or date_to:
            filtered_predictions = []
            for pred in predictions:
                pred_date = pred['timestamp'].date()
                
                # Check from date
                if date_from and pred_date < datetime.strptime(date_from, '%Y-%m-%d').date():
                    continue
                
                # Check to date
                if date_to and pred_date > datetime.strptime(date_to, '%Y-%m-%d').date():
                    continue
                
                filtered_predictions.append(pred)
            
            predictions = filtered_predictions
        
    except Exception as e:
        flash(f"Error retrieving predictions: {str(e)}", "danger")
        predictions = []
    
    return render_template('prediction_reports.html',
                          predictions=predictions,
                          patients=patients,
                          selected_patient=patient_name,
                          date_from=date_from,
                          date_to=date_to)

@app.route('/prediction/<int:prediction_id>', methods=['GET'])
def view_prediction(prediction_id):
    """View a single prediction with detailed information."""
    # Get database configuration
    db_config = {
        'host': 'localhost',
        'user': 'suba',
        'password': 'Suba@123',
        'database': 'embryo_predictions'
    }
    
    try:
        # Get all predictions and filter for the requested ID
        all_predictions = get_predictions_by_patient(None, db_config)
        prediction = next((p for p in all_predictions if p['id'] == prediction_id), None)
        
        if not prediction:
            flash(f"Prediction with ID {prediction_id} not found.", "danger")
            return redirect(url_for('prediction_reports'))
        
        # Get the image data
        image_path = prediction['image_path']
        image_data = None
        
        try:
            # Check if the image exists
            if os.path.exists(image_path):
                with open(image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                image_data = f"data:image/jpeg;base64,{encoded_image}"
            else:
                # Use a placeholder image
                image_data = url_for('static', filename='img/placeholder.jpg')
        except Exception as e:
            print(f"Error loading image: {e}")
            image_data = url_for('static', filename='img/placeholder.jpg')
        
        # Generate XAI visualization if possible
        xai_data = None
        try:
            if os.path.exists(image_path):
                # Find the latest model
                models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
                models = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith('.pth')]
                if models:
                    latest_model = max(models, key=os.path.getmtime)
                    predictor = EmbryoPredictor(latest_model)
                    _, transform = get_transforms()
                    xai_result = generate_xai_visualization(
                        model=predictor.model,
                        image_path=image_path,
                        transform=transform,
                        class_names=prediction['class_names'],
                        device=predictor.device
                    )
                    if xai_result and 'xai_image' in xai_result:
                        xai_data = xai_result['xai_image']
        except Exception as e:
            print(f"Error generating XAI visualization: {e}")
        
        return render_template('view_prediction.html',
                              prediction=prediction,
                              image_data=image_data,
                              xai_data=xai_data)
                              
    except Exception as e:
        flash(f"Error retrieving prediction details: {str(e)}", "danger")
        return redirect(url_for('prediction_reports'))

@app.route('/download/prediction/<int:prediction_id>', methods=['GET'])
def download_prediction(prediction_id):
    """Download a prediction report as PDF."""
    # This is a placeholder for PDF generation functionality
    # In a real implementation, you would generate a PDF report here
    flash("PDF download functionality will be implemented in a future update.", "info")
    return redirect(url_for('view_prediction', prediction_id=prediction_id))

# Main route for testing
@app.route('/')
def index():
    """Home page for testing reports."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
