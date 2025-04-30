import os
import sys
import json
import glob
import pandas as pd
import uuid
import base64
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, session
from datetime import datetime
import markdown
import bleach
import functools
import pymysql
import mysql.connector
from mysql.connector import Error

# Get the absolute path to the project root directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.append(PROJECT_ROOT)

# Import evaluation and prediction modules
from src.evaluate_model import ModelEvaluator, find_latest_model
from src.predict_image import EmbryoPredictor
from src.xai_utils import generate_xai_visualization
from src.train_model import get_transforms
# Import authentication module
from src.auth_utils import AuthManager

app = Flask(__name__)
app.secret_key = 'embryo_quality_prediction_app'
# Session configuration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours

# Authentication manager
auth_manager = AuthManager()

# Authentication middleware
def login_required(view_func):
    @functools.wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please sign in to access this page', 'warning')
            return redirect(url_for('login', next=request.url))
        return view_func(*args, **kwargs)
    return wrapped_view

# Configuration
class AppConfig:
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
    PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
    UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
    PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")
    
    # Allowed file extensions for image uploads
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure all required directories exist."""
        for dir_path in [cls.MODELS_DIR, cls.RESULTS_DIR, cls.PLOTS_DIR, cls.UPLOAD_DIR, cls.PREDICTIONS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
            
    @classmethod
    def allowed_file(cls, filename):
        """Check if file has an allowed extension."""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in cls.ALLOWED_EXTENSIONS

# Database connection configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'suba',
    'password': 'Suba@123',  # Use the same password as in AuthManager
    'database': 'embryo_predictions',  # Use the same database name as in AuthManager
    'cursorclass': pymysql.cursors.DictCursor
}

# Function to get database connection using mysql.connector (same as AuthManager)
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='suba',
            password='Suba@123',
            database='embryo_predictions'
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

@app.route('/')
@login_required
def index():
    """Home page showing model evaluation dashboard."""
    AppConfig.ensure_dirs()
    
    # Get list of available models
    models = []
    for model_file in glob.glob(os.path.join(AppConfig.MODELS_DIR, "*.pth")):
        model_name = os.path.basename(model_file)
        models.append({
            'name': model_name,
            'path': model_file,
            'modified': datetime.fromtimestamp(os.path.getmtime(model_file)).strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # Sort models by modification time (newest first)
    models.sort(key=lambda x: x['modified'], reverse=True)
    
    # Get list of evaluation reports
    reports = []
    for report_file in glob.glob(os.path.join(AppConfig.RESULTS_DIR, "report_*.html")):
        report_name = os.path.basename(report_file)
        reports.append({
            'name': report_name,
            'path': report_file,
            'modified': datetime.fromtimestamp(os.path.getmtime(report_file)).strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # Sort reports by modification time (newest first)
    reports.sort(key=lambda x: x['modified'], reverse=True)
    
    # Load evaluation history if available
    csv_file = os.path.join(AppConfig.RESULTS_DIR, "model_evaluations.csv")
    history = None
    if os.path.exists(csv_file):
        try:
            history = pd.read_csv(csv_file)
            # Convert to list of dicts for template
            history = history.to_dict('records')
        except Exception as e:
            flash(f"Error loading evaluation history: {e}", "danger")
    
    return render_template('index.html', 
                          models=models, 
                          reports=reports, 
                          history=history)


@app.route('/dashboard')
@login_required
def dashboard():
    """Advanced dashboard showing comprehensive model evaluation results."""
    AppConfig.ensure_dirs()
    
    # Get list of available models
    models = []
    for model_file in glob.glob(os.path.join(AppConfig.MODELS_DIR, "*.pth")):
        model_name = os.path.basename(model_file)
        models.append({
            'name': model_name,
            'path': model_file,
            'modified': datetime.fromtimestamp(os.path.getmtime(model_file)).strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # Sort models by modification time (newest first)
    models.sort(key=lambda x: x['modified'], reverse=True)
    
    # Load model results from individual CSV files
    model_data = []
    model_names = []
    accuracy_data = []
    precision_data = []
    recall_data = []
    f1_data = []
    
    # First, check if we have individual model result files
    result_files = glob.glob(os.path.join(AppConfig.RESULTS_DIR, "*_results.csv"))
    
    if result_files:
        # Process individual model result files
        for result_file in result_files:
            try:
                df = pd.read_csv(result_file)
                if not df.empty:
                    model_data.append(df.iloc[0].to_dict())
                    model_name = df.iloc[0]['model_name']
                    model_names.append(model_name)
                    accuracy_data.append(float(df.iloc[0]['accuracy']))
                    precision_data.append(float(df.iloc[0]['precision']))
                    recall_data.append(float(df.iloc[0]['recall']))
                    f1_data.append(float(df.iloc[0]['f1_score']))
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
    else:
        # Fall back to model_evaluations.csv
        csv_file = os.path.join(AppConfig.RESULTS_DIR, "model_evaluations.csv")
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                if not df.empty:
                    # Group by model_name and take the latest evaluation for each model
                    df = df.sort_values('timestamp', ascending=False)
                    df = df.drop_duplicates(subset=['model_name'])
                    
                    model_data = df.to_dict('records')
                    model_names = df['model_name'].tolist()
                    accuracy_data = df['accuracy'].tolist()
                    precision_data = df['precision'].tolist()
                    recall_data = df['recall'].tolist()
                    f1_data = df['f1_score'].tolist()
            except Exception as e:
                flash(f"Error loading evaluation history: {e}", "danger")
    
    # Sort model data by accuracy (descending)
    if model_data:
        sorted_indices = np.argsort([-m['accuracy'] for m in model_data])
        model_data = [model_data[i] for i in sorted_indices]
        model_names = [model_names[i] for i in sorted_indices]
        accuracy_data = [accuracy_data[i] for i in sorted_indices]
        precision_data = [precision_data[i] for i in sorted_indices]
        recall_data = [recall_data[i] for i in sorted_indices]
        f1_data = [f1_data[i] for i in sorted_indices]
    
    # Extract class names and prepare class-wise F1 data
    class_names = []
    class_f1_keys = []
    class_f1_data = []
    
    # If we didn't find class-specific F1 scores, try the format from model_evaluations.csv
    if not class_names:
        for key in model_data[0].keys():
            if key.startswith('f1_') and not key == 'f1_score':
                class_name = key.replace('f1_', '')
                class_names.append(class_name)
                class_f1_keys.append(key)
    
    # Prepare class-wise F1 data for charts
    for i, class_key in enumerate(class_f1_keys):
        class_f1_values = [float(m[class_key]) for m in model_data]
        class_f1_data.append(class_f1_values)
        
    # Get patient predictions from database
    patient_predictions = []
    try:
        # Connect to the database using mysql.connector (same as AuthManager)
        connection = get_db_connection()
        
        if connection:
            cursor = connection.cursor(dictionary=True)
            
            # Get the most recent 20 predictions with more detailed error reporting
            query = """
                SELECT 
                    id, image_path, patient_name, predicted_class, confidence, 
                    timestamp, created_at
                FROM 
                    predictions 
                ORDER BY 
                    timestamp DESC 
                LIMIT 20
            """
            print(f"Executing query: {query}")
            cursor.execute(query)
            patient_predictions = cursor.fetchall()
            
            # Add filename to each prediction for easier template rendering
            for pred in patient_predictions:
                if pred.get('image_path'):
                    # Extract just the filename from the full path 
                    pred['filename'] = os.path.basename(pred['image_path'])
                else:
                    pred['filename'] = ''
            
            # Check if we got any results
            print(f"Retrieved {len(patient_predictions)} prediction records")
            if patient_predictions:
                print(f"Sample record: {str(patient_predictions[0])}")
            else:
                # Try a different simpler query if no results
                cursor.execute("SELECT COUNT(*) as count FROM predictions")
                count_result = cursor.fetchone()
                print(f"Total prediction records in database: {count_result['count'] if count_result else 'unknown'}")
                
                # Try to get all columns to see structure
                cursor.execute("SHOW COLUMNS FROM predictions")
                columns = cursor.fetchall()
                print(f"Table columns: {[col['Field'] for col in columns]}")
                
                # Try a more basic query
                cursor.execute("SELECT * FROM predictions LIMIT 5")
                basic_results = cursor.fetchall()
                print(f"Basic query returned {len(basic_results)} records")
                if basic_results:
                    print(f"Sample basic record: {str(basic_results[0])}")
            
            cursor.close()
            connection.close()
        else:
            print("Failed to establish database connection")
            flash("Failed to connect to the database", "danger")
    except Exception as e:
        print(f"Error loading patient predictions: {str(e)}")
        flash(f"Error loading patient predictions: {str(e)}", "warning")
        # Don't abort, continue with empty predictions
    
    # If we don't have any predictions from the database, initialize with empty list
    if not patient_predictions:
        patient_predictions = []
        print("No patient predictions were loaded, using empty list")
    
    return render_template('dashboard.html',
                          models=models,
                          model_data=model_data,
                          model_names=model_names,
                          accuracy_data=accuracy_data,
                          precision_data=precision_data,
                          recall_data=recall_data,
                          f1_data=f1_data,
                          class_names=class_names,
                          class_f1_keys=class_f1_keys,
                          class_f1_data=class_f1_data,
                          patient_predictions=patient_predictions)


@app.route('/evaluate', methods=['POST'])
@login_required
def evaluate_model():
    """Evaluate a model and generate a report."""
    model_path = request.form.get('model_path')
    
    if not model_path:
        flash("No model selected for evaluation", "danger")
        return redirect(url_for('index'))
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(model_path)
        
        # Evaluate model
        evaluator.evaluate()
        
        # Save results
        evaluator.save_results()
        
        # Generate HTML report
        html_path = os.path.join(AppConfig.RESULTS_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        report_path = evaluator.generate_html_report(html_path)
        
        flash(f"Model evaluated successfully. Report generated at {os.path.basename(report_path)}", "success")
        
        # Redirect to the dashboard
        return redirect(url_for('dashboard'))
    
    except Exception as e:
        flash(f"Error evaluating model: {e}", "danger")
        return redirect(url_for('index'))


@app.route('/report/<path:report_path>')
@login_required
def view_report(report_path):
    """View a specific evaluation report."""
    # Check if the path is a directory or file
    full_path = os.path.join(AppConfig.RESULTS_DIR, report_path)
    
    # If it's a directory, look for the report file
    if os.path.isdir(full_path):
        # Find HTML files in the directory
        html_files = glob.glob(os.path.join(full_path, "*.html"))
        if html_files:
            full_path = html_files[0]  # Use the first HTML file found
        else:
            flash(f"No report found in {report_path}", "danger")
            return redirect(url_for('index'))
    
    # If it's not a directory and doesn't exist, report error
    if not os.path.exists(full_path):
        flash(f"Report {report_path} not found", "danger")
        return redirect(url_for('index'))
    
    # Read and return the report content with proper content type
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        return report_content, {"Content-Type": "text/html"}
    except Exception as e:
        flash(f"Error reading report: {e}", "danger")
        return redirect(url_for('index'))


@app.route('/compare', methods=['GET', 'POST'])
@login_required
def compare_models():
    """Compare multiple model evaluations."""
    AppConfig.ensure_dirs()
    
    if request.method == 'POST':
        selected_models = request.form.getlist('selected_models')
        
        if not selected_models:
            flash("No models selected for comparison", "danger")
            return redirect(url_for('compare_models'))
        
        # Load evaluation history
        csv_file = os.path.join(AppConfig.RESULTS_DIR, "model_evaluations.csv")
        if not os.path.exists(csv_file):
            flash("No evaluation history found", "danger")
            return redirect(url_for('index'))
        
        try:
            history = pd.read_csv(csv_file)
            # Filter by selected models
            comparison_data = history[history['model_name'].isin(selected_models)]
            
            if comparison_data.empty:
                flash("No evaluation data found for selected models", "danger")
                return redirect(url_for('compare_models'))
            
            # Convert to list of dicts for template
            comparison_data = comparison_data.to_dict('records')
            
            return render_template('compare.html', 
                                  comparison_data=comparison_data,
                                  selected_models=selected_models)
        
        except Exception as e:
            flash(f"Error loading comparison data: {e}", "danger")
            return redirect(url_for('index'))
    
    # GET request - show selection form
    # Load evaluation history
    csv_file = os.path.join(AppConfig.RESULTS_DIR, "model_evaluations.csv")
    models_to_compare = []
    
    if os.path.exists(csv_file):
        try:
            history = pd.read_csv(csv_file)
            # Get unique model names
            models_to_compare = history['model_name'].unique().tolist()
        except Exception as e:
            flash(f"Error loading evaluation history: {e}", "danger")
    
    return render_template('select_compare.html', models=models_to_compare)


@app.route('/api/model_metrics')
@login_required
def api_model_metrics():
    """API endpoint to get model metrics for charts."""
    # First try to load from individual model result files
    result_files = glob.glob(os.path.join(AppConfig.RESULTS_DIR, "*_results.csv"))
    
    if result_files:
        try:
            # Combine all result files
            dfs = []
            for file in result_files:
                df = pd.read_csv(file)
                if not df.empty:
                    dfs.append(df)
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # Convert timestamps to datetime for sorting
                combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
                
                # Sort by timestamp
                combined_df = combined_df.sort_values('timestamp')
                
                # Prepare data for charts
                data = {
                    'timestamps': combined_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    'models': combined_df['model_name'].tolist(),
                    'accuracy': combined_df['accuracy'].tolist(),
                    'precision': combined_df['precision'].tolist(),
                    'recall': combined_df['recall'].tolist(),
                    'f1_score': combined_df['f1_score'].tolist()
                }
                
                return jsonify(data)
        except Exception as e:
            print(f"Error processing individual result files: {e}")
    
    # Fall back to model_evaluations.csv
    csv_file = os.path.join(AppConfig.RESULTS_DIR, "model_evaluations.csv")
    if not os.path.exists(csv_file):
        return jsonify({'error': 'No evaluation data found'})
    
    try:
        df = pd.read_csv(csv_file)
        
        # Convert timestamps to datetime for sorting
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Prepare data for charts
        data = {
            'timestamps': df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'models': df['model_name'].tolist(),
            'accuracy': df['accuracy'].tolist(),
            'precision': df['precision'].tolist(),
            'recall': df['recall'].tolist(),
            'f1_score': df['f1_score'].tolist()
        }
        
        return jsonify(data)
    
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/validate', methods=['GET', 'POST'])
@login_required
def validate_image():
    """Validate embryo images using the trained model."""
    AppConfig.ensure_dirs()
    
    # Get list of available models
    models = []
    for model_file in glob.glob(os.path.join(AppConfig.MODELS_DIR, "*.pth")):
        model_name = os.path.basename(model_file)
        models.append({
            'name': model_name,
            'path': model_file,
            'modified': datetime.fromtimestamp(os.path.getmtime(model_file)).strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # Sort models by modification time (newest first)
    models.sort(key=lambda x: x['modified'], reverse=True)
    
    # Handle file upload
    if request.method == 'POST':
        # Check if model is selected
        model_path = request.form.get('model_path')
        if not model_path and models:
            model_path = models[0]['path']
            
        # Get patient name if provided
        patient_name = request.form.get('patient_name', '')
        print(f"Raw patient name from form: '{patient_name}' (type: {type(patient_name).__name__})")
        
        # Ensure patient_name is a string
        if patient_name is None:
            print("Patient name is None")
            patient_name = None
        elif isinstance(patient_name, str):
            patient_name = patient_name.strip()
            print(f"Stripped patient name: '{patient_name}' (length: {len(patient_name)})")
            
            # Only set patient_name to None if it's actually empty after stripping
            if not patient_name:
                print("Patient name is empty after stripping, setting to None")
                patient_name = None
            else:
                print(f"Using patient name: '{patient_name}'")
        else:
            # Convert to string if it's some other type
            patient_name = str(patient_name).strip()
            if not patient_name:
                print("Converted patient name is empty, setting to None")
                patient_name = None
            else:
                print(f"Using converted patient name: '{patient_name}'")
                
        print(f"Final patient name to be passed to predictor: '{patient_name}'")
        
        # Force to None if empty string (extra safety)
        if patient_name == '':
            print("Empty string detected, forcing to None")
            patient_name = None
        
        # Log the patient name for debugging
        print(f"Form submitted with patient name: '{patient_name}' (type: {type(patient_name).__name__ if patient_name is not None else 'None'})")
        
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and AppConfig.allowed_file(file.filename):
            # Generate a unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(AppConfig.UPLOAD_DIR, unique_filename)
            
            # Save the file
            file.save(file_path)
            
            try:
                # Initialize predictor with selected model
                try:
                    predictor = EmbryoPredictor(model_path)
                except Exception as e:
                    flash(f"Error initializing predictor: {e}", "danger")
                    import traceback
                    traceback.print_exc()
                    return redirect(request.url)
                
                # Make prediction with detailed error handling
                try:
                    # Pass patient_name directly to the predict method
                    result = predictor.predict(file_path, patient_name)
                    if result is None:
                        raise ValueError("Prediction returned None result")
                    
                    print(f"Prediction made with patient name: {patient_name if patient_name else 'None'}")
                except Exception as e:
                    flash(f"Error during prediction: {e}", "danger")
                    import traceback
                    traceback.print_exc()
                    return redirect(request.url)
                
                # Generate XAI visualization
                try:
                    _, transform = get_transforms()
                    xai_result = generate_xai_visualization(
                        model=predictor.model,
                        image_path=file_path,
                        transform=transform,
                        class_names=predictor.class_names,
                        device=predictor.device
                    )
                except Exception as e:
                    flash(f"Warning: Could not generate XAI visualization: {e}", "warning")
                    xai_result = None
                
                # Save prediction to database and file
                try:
                    save_result = predictor.save_prediction(result, save_to_database=True)
                    if save_result.get('database_id'):
                        print(f"Saved prediction to database with ID: {save_result['database_id']}")
                    else:
                        print("Warning: Prediction saved to file but not to database")
                except Exception as e:
                    # Non-critical error, just log it
                    print(f"Warning: Could not save prediction: {e}")
                
                # Convert image to base64 for display
                try:
                    with open(file_path, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                except Exception as e:
                    flash(f"Error encoding image: {e}", "danger")
                    return redirect(request.url)
                
                # Render result template with XAI visualization if available
                if xai_result:
                    return render_template('validation_result.html', 
                                        result=result,
                                        image_data=encoded_image,
                                        xai_data=xai_result['xai_image'],
                                        image_name=file.filename)
                else:
                    return render_template('validation_result.html', 
                                        result=result,
                                        image_data=encoded_image,
                                        image_name=file.filename)
            
            except Exception as e:
                flash(f"Error in validation process: {e}", "danger")
                import traceback
                traceback.print_exc()
                return redirect(request.url)
        else:
            flash(f"Invalid file type. Allowed types: {', '.join(AppConfig.ALLOWED_EXTENSIONS)}", "danger")
            return redirect(request.url)
    
    # GET request - show upload form
    return render_template('validate.html', models=models)


@app.route('/uploads/<path:filename>')
@login_required
def uploaded_file(filename):
    """Serve uploaded files."""
    try:
        # Extract just the base filename in case the full path was passed
        base_filename = os.path.basename(filename)
        return send_from_directory(AppConfig.UPLOAD_DIR, base_filename)
    except Exception as e:
        print(f"Error serving uploaded file '{filename}': {str(e)}")
        return f"Error: File not found or could not be served", 404


@app.route('/results/<path:filepath>')
@login_required
def serve_results(filepath):
    """Serve files from the results directory (images, plots, etc.)."""
    # Extract the directory part from the filepath
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    
    # Construct the full directory path
    full_dir_path = os.path.join(AppConfig.RESULTS_DIR, directory)
    
    return send_from_directory(full_dir_path, filename)


@app.route('/docs/<doc_name>')
@login_required
def view_docs(doc_name):
    """Render markdown documentation files as HTML."""
    # Get parent directory of PROJECT_ROOT
    PARENT_DIR = os.path.dirname(PROJECT_ROOT)
    
    # Define allowed documentation files to prevent directory traversal
    allowed_docs = {
        'README': os.path.join(PARENT_DIR, 'README.md'),
        'WORKFLOW': os.path.join(PARENT_DIR, 'WORKFLOW.md'),
        'MODEL_EVALUATION': os.path.join(PARENT_DIR, 'MODEL_EVALUATION.md')
    }
    
    # Ensure the requested doc exists and is allowed
    if doc_name not in allowed_docs:
        flash(f"Documentation '{doc_name}' not found.", "danger")
        return redirect(url_for('index'))
    
    doc_path = allowed_docs[doc_name]
    
    try:
        # Read the markdown file
        with open(doc_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=['fenced_code', 'tables', 'toc']
        )
        
        # Apply custom styling
        styled_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{doc_name} - Embryo Quality Prediction</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.3/font/bootstrap-icons.css">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    padding: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    margin-top: 1.5em;
                    margin-bottom: 0.75em;
                    color: #1a3a6c;
                }}
                h1 {{
                    padding-bottom: 0.5em;
                    border-bottom: 1px solid #eee;
                }}
                code {{
                    background-color: #f5f5f5;
                    padding: 0.2em 0.4em;
                    border-radius: 3px;
                    font-family: Consolas, Monaco, 'Andale Mono', monospace;
                }}
                pre {{
                    background-color: #f8f8f8;
                    padding: 16px;
                    border-radius: 6px;
                    overflow-x: auto;
                    border: 1px solid #e1e4e8;
                }}
                pre code {{
                    background-color: transparent;
                    padding: 0;
                    border-radius: 0;
                }}
                blockquote {{
                    border-left: 4px solid #ddd;
                    padding: 0 15px;
                    color: #777;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                table, th, td {{
                    border: 1px solid #e1e4e8;
                }}
                th, td {{
                    padding: 8px 16px;
                    text-align: left;
                }}
                th {{
                    background-color: #f8f8f8;
                }}
                tr:nth-child(even) {{
                    background-color: #f8f8f8;
                }}
                .navbar {{
                    margin-bottom: 30px;
                    background-color: #4a6fa5;
                }}
                .btn-back {{
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <nav class="navbar navbar-expand-lg navbar-dark">
                <div class="container-fluid">
                    <a class="navbar-brand" href="/">Embryo Quality Prediction</a>
                    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                        <span class="navbar-toggler-icon"></span>
                    </button>
                    <div class="collapse navbar-collapse" id="navbarNav">
                        <ul class="navbar-nav">
                            <li class="nav-item">
                                <a class="nav-link" href="/">Dashboard</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="/docs/README">Project Overview</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="/docs/WORKFLOW">Workflow Guide</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="/docs/MODEL_EVALUATION">Evaluation Guide</a>
                            </li>
                        </ul>
                    </div>
                </div>
            </nav>
            
            <a href="/" class="btn btn-outline-primary btn-back">
                <i class="bi bi-arrow-left"></i> Back to Dashboard
            </a>
            
            <div class="doc-content">
                {html_content}
            </div>
            
            <footer class="mt-5 pt-3 border-top text-muted">
                <div class="row">
                    <div class="col-md-6">
                        <p>Embryo Quality Prediction System</p>
                    </div>
                    <div class="col-md-6 text-md-end">
                        <p>Documentation rendered from Markdown</p>
                    </div>
                </div>
            </footer>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """
        
        return styled_html
        
    except FileNotFoundError:
        flash(f"Documentation file not found: {doc_path}", "danger")
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"Error loading documentation: {str(e)}", "danger")
        return redirect(url_for('index'))


@app.route('/batch_validate', methods=['GET', 'POST'])
@login_required
def batch_validate_images():
    """Validate multiple embryo images using the trained model."""
    try:
        # Ensure all required directories exist
        AppConfig.ensure_dirs()
        
        # Log request information for debugging
        if request.method == 'POST':
            print(f"\n{'='*50}")
            print("BATCH VALIDATION REQUEST")
            print(f"Content Type: {request.content_type}")
            print(f"Content Length: {request.content_length}")
            print(f"Form Data Keys: {list(request.form.keys())}")
            print(f"Files Keys: {list(request.files.keys())}")
            print(f"Files Count: {len(request.files.getlist('files[]'))}")
            print(f"{'='*50}\n")
        
        # Get list of available models
        models = []
        for model_file in glob.glob(os.path.join(AppConfig.MODELS_DIR, "*.pth")):
            model_name = os.path.basename(model_file)
            models.append({
                'name': model_name,
                'path': model_file,
                'modified': datetime.fromtimestamp(os.path.getmtime(model_file)).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Sort models by modification time (newest first)
        models.sort(key=lambda x: x['modified'], reverse=True)
        
        # Find the ResNet152 model if available
        default_model_index = None
        for i, model in enumerate(models):
            if 'resnet152_final.pth' in model['path'].lower():
                default_model_index = i
                break
        
        # Handle file upload
        if request.method == 'POST':
            # Check if model is selected
            model_path = request.form.get('model_path')
            if not model_path and models:
                # Set default model to ResNet152_final.pth if available
                default_model = next((m for m in models if 'resnet152_final.pth' in m['path'].lower()), None)
                if default_model:
                    model_path = default_model['path']
                    print(f"Using default model: {model_path}")
                else:
                    model_path = models[0]['path']
                    print(f"ResNet152_final.pth not found, using first available model: {model_path}")
                    
            # Get patient name if provided
            patient_name = request.form.get('patient_name', '')
            print(f"Raw patient name from form: '{patient_name}' (type: {type(patient_name).__name__})")
            
            # Ensure patient_name is a string
            if patient_name is None:
                print("Patient name is None")
                patient_name = None
            elif isinstance(patient_name, str):
                patient_name = patient_name.strip()
                print(f"Stripped patient name: '{patient_name}' (length: {len(patient_name)})")
                
                # Only set patient_name to None if it's actually empty after stripping
                if not patient_name:
                    print("Patient name is empty after stripping, setting to None")
                    patient_name = None
                else:
                    print(f"Using patient name: '{patient_name}'")
            else:
                # Convert to string if it's some other type
                patient_name = str(patient_name).strip()
                if not patient_name:
                    print("Converted patient name is empty, setting to None")
                    patient_name = None
                else:
                    print(f"Using converted patient name: '{patient_name}'")
                    
            print(f"Final patient name to be passed to predictor: '{patient_name}'")
            
            # Force to None if empty string (extra safety)
            if patient_name == '':
                print("Empty string detected, forcing to None")
                patient_name = None
            
            # Check if files were uploaded
            if 'files[]' not in request.files:
                error_msg = 'No files were uploaded. Please select at least one image file.'
                print(f"Error: {error_msg}")
                flash(error_msg, 'danger')
                return redirect(request.url)
            
            files = request.files.getlist('files[]')
            print(f"Files received: {len(files)}")
            
            # Validate files
            if not files or all(f.filename == '' for f in files):
                error_msg = 'No files were selected. Please choose at least one image file to upload.'
                print(f"Error: {error_msg}")
                flash(error_msg, 'danger')
                return redirect(request.url)
            
            # Initialize predictor with selected model
            try:
                print(f"Initializing predictor with model: {model_path}")
                predictor = EmbryoPredictor(model_path)
            except Exception as e:
                error_msg = f"Error initializing predictor: {str(e)}"
                print(f"Error: {error_msg}")
                flash(error_msg, "danger")
                import traceback
                traceback.print_exc()
                return redirect(request.url)
            
            results = []
            error_messages = []
            
            # Create a list of patient names (same patient name for all images in this batch)
            patient_names = [patient_name] * len(files)
            print(f"Using patient name '{patient_name}' for all {len(files)} images in batch")
            
            file_paths = []
            filenames = []
            for i, file in enumerate(files):
                print(f"Processing file {i+1}/{len(files)}: {file.filename}")
                if file and AppConfig.allowed_file(file.filename):
                    try:
                        # Generate a unique filename
                        filename = secure_filename(file.filename)
                        unique_filename = f"{uuid.uuid4()}_{filename}"
                        file_path = os.path.join(AppConfig.UPLOAD_DIR, unique_filename)
                        
                        # Save the file
                        print(f"Saving file to: {file_path}")
                        file.save(file_path)
                        
                        # Resize image to max 300x300 if needed
                        try:
                            from PIL import Image
                            img = Image.open(file_path)
                            img.thumbnail((300, 300))
                            img.save(file_path)
                            print(f"Resized image to max 300x300")
                        except Exception as e:
                            print(f"Warning: Could not resize image: {str(e)}")
                        
                        file_paths.append(file_path)
                        filenames.append(filename)
                    except Exception as e:
                        error_msg = f"Error processing {file.filename}: {str(e)}"
                        print(f"Error: {error_msg}")
                        error_messages.append(error_msg)
                        import traceback
                        traceback.print_exc()
                else:
                    error_msg = f"Invalid file type for {file.filename}. Allowed types: {', '.join(AppConfig.ALLOWED_EXTENSIONS)}"
                    print(f"Error: {error_msg}")
                    error_messages.append(error_msg)
            
            # Process batch predictions if we have valid files
            if file_paths:
                try:
                    print(f"Making batch predictions for {len(file_paths)} files with patient name: {patient_name}")
                    # Pass patient_names list to predict_batch
                    batch_results = predictor.predict_batch(file_paths, patient_names)
                    
                    # Process each result
                    for i, result in enumerate(batch_results):
                        if i >= len(filenames):
                            continue
                            
                        filename = filenames[i]
                        file_path = file_paths[i]
                        
                        # Generate XAI visualization
                        try:
                            print(f"Generating XAI visualization for {filename}")
                            _, transform = get_transforms()
                            xai_result = generate_xai_visualization(
                                model=predictor.model,
                                image_path=file_path,
                                transform=transform,
                                class_names=predictor.class_names,
                                device=predictor.device
                            )
                        except Exception as e:
                            print(f"Warning: Could not generate XAI visualization for {filename}: {e}")
                            xai_result = None
                        
                        # Save prediction to database and file
                        try:
                            save_result = predictor.save_prediction(result, save_to_database=True)
                            if save_result.get('database_id'):
                                print(f"Saved prediction to database with ID: {save_result['database_id']}")
                                # Add database ID to result for display
                                result['database_id'] = save_result['database_id']
                            else:
                                print(f"Warning: Prediction for {filename} saved to file but not to database")
                        except Exception as e:
                            print(f"Warning: Could not save prediction for {filename}: {e}")
                        
                        # Convert image to base64 for display
                        try:
                            with open(file_path, "rb") as image_file:
                                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                            print(f"Encoded image to base64")
                            
                            # Add result to list
                            results.append({
                                'filename': filename,
                                'result': result,
                                'image_data': encoded_image,
                                'xai_data': xai_result['xai_image'] if xai_result else None,
                                'database_id': result.get('database_id')
                            })
                            print(f"Added result for {filename}")
                        except Exception as e:
                            error_msg = f"Error encoding image {filename}: {str(e)}"
                            print(f"Error: {error_msg}")
                            error_messages.append(error_msg)
                except Exception as e:
                    error_msg = f"Error processing batch predictions: {str(e)}"
                    print(f"Error: {error_msg}")
                    error_messages.append(error_msg)
                    import traceback
                    traceback.print_exc()
            
            if results:
                # Show any errors that occurred during processing
                print(f"Processing complete. Successful results: {len(results)}, Errors: {len(error_messages)}")
                for error in error_messages:
                    flash(error, 'warning')
                
                # Get model name from the path
                model_name = os.path.basename(model_path) if model_path else "Default Model"
                
                return render_template('batch_validation_results.html', 
                                       results=results,
                                       model_name=model_name)
            else:
                # If no results were processed, show all errors
                print("No valid results were processed")
                for error in error_messages:
                    flash(error, 'danger')
                return redirect(request.url)
        
        # GET request - show upload form
        return render_template('batch_validate.html', 
                              models=models, 
                              default_model_index=default_model_index)
    
    except Exception as e:
        # Catch-all exception handler
        error_msg = f"Unexpected error in batch validation: {str(e)}"
        print(f"CRITICAL ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        flash(error_msg, 'danger')
        return redirect(url_for('index'))

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page."""
    next_url = request.args.get('next', url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            flash('Please provide both email and password', 'danger')
            return render_template('login.html')
        
        # Authenticate user
        user = auth_manager.authenticate_user(email, password)
        
        if user and 'error' not in user:
            # Store user info in session
            session['user_id'] = user['id']
            session['email'] = user['email']
            
            flash('You have been logged in successfully', 'success')
            return redirect(next_url)
        else:
            error_message = user.get('error', 'Authentication failed') if user else 'Authentication failed'
            flash(error_message, 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page."""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not email or not password or not confirm_password:
            flash('Please fill all required fields', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        
        # Register new user
        result = auth_manager.register_user(email, password)
        
        if result and 'error' not in result:
            flash('Registration successful! You can now log in', 'success')
            return redirect(url_for('login'))
        else:
            error_message = result.get('error', 'Registration failed') if result else 'Registration failed'
            flash(error_message, 'danger')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """User logout."""
    # Clear session
    session.pop('user_id', None)
    session.pop('email', None)
    session.clear()
    
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/patient_report/<int:prediction_id>')
@login_required
def patient_report(prediction_id):
    """View detailed patient prediction report."""
    try:
        print(f"Retrieving patient report for prediction_id: {prediction_id}")
        
        # Connect to the database
        connection = get_db_connection()
        
        if not connection:
            print("Failed to establish database connection")
            flash("Failed to connect to the database", "danger")
            return redirect(url_for('dashboard'))
        
        cursor = connection.cursor(dictionary=True)
        prediction = None
        probabilities = []
        
        try:
            # First get the prediction details directly with a simple query
            query = """
                SELECT 
                    id, image_path, patient_name, predicted_class_index, 
                    predicted_class, confidence, timestamp, created_at
                FROM 
                    predictions 
                WHERE 
                    id = %s
            """
            cursor.execute(query, (prediction_id,))
            prediction = cursor.fetchone()
            print(f"Prediction details: {str(prediction) if prediction else 'None'}")
            
            if prediction:
                # If we have prediction details, get the class probabilities
                prob_query = """
                    SELECT 
                        class_name, probability
                    FROM 
                        class_probabilities
                    WHERE 
                        prediction_id = %s
                    ORDER BY 
                        probability DESC
                """
                cursor.execute(prob_query, (prediction_id,))
                probabilities = cursor.fetchall()
                print(f"Retrieved {len(probabilities)} probability records")
        finally:
            cursor.close()
            connection.close()
        
        if not prediction:
            print(f"No prediction found for id: {prediction_id}")
            flash('Prediction not found', 'danger')
            return redirect(url_for('dashboard'))
        
        # Ensure image_path contains the full filename
        if prediction.get('image_path'):
            prediction['filename'] = os.path.basename(prediction['image_path'])
        else:
            prediction['filename'] = ''
        
        # Generate XAI visualization (heatmap) if image path exists
        xai_data = None
        if prediction.get('image_path') and os.path.exists(prediction['image_path']):
            try:
                print(f"Generating XAI visualization for {prediction['image_path']}")
                # Find latest model
                model_path = find_latest_model(AppConfig.MODELS_DIR)
                if model_path:
                    # Initialize predictor
                    predictor = EmbryoPredictor(model_path)
                    
                    # Generate XAI visualization
                    _, transform = get_transforms()
                    xai_result = generate_xai_visualization(
                        model=predictor.model,
                        image_path=prediction['image_path'],
                        transform=transform,
                        class_names=predictor.class_names,
                        device=predictor.device
                    )
                    
                    # Get the XAI image data
                    if xai_result and 'xai_image' in xai_result:
                        xai_data = xai_result['xai_image']
                        print("XAI visualization generated successfully")
                    else:
                        print("XAI result does not contain xai_image")
                else:
                    print("No model found for XAI visualization")
            except Exception as e:
                print(f"Error generating XAI visualization: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Image path not found or invalid: {prediction.get('image_path')}")
        
        return render_template('patient_report.html', 
                              prediction=prediction, 
                              probabilities=probabilities,
                              xai_data=xai_data)
    
    except Exception as e:
        print(f"Error in patient_report: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error retrieving prediction data: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/test_db')
@login_required
def test_db():
    """Test database connection and show patient predictions."""
    try:
        # Connect to the database
        connection = get_db_connection()
        
        if not connection:
            return "<h1>Database Error</h1><p>Failed to connect to the database</p>"
        
        cursor = connection.cursor(dictionary=True)
        try:
            # Get all predictions
            cursor.execute("""
                SELECT * FROM predictions
                ORDER BY timestamp DESC
            """)
            predictions = cursor.fetchall()
            
            # Get all class probabilities
            cursor.execute("""
                SELECT * FROM class_probabilities
                LIMIT 10
            """)
            probabilities = cursor.fetchall()
        finally:
            cursor.close()
            connection.close()
        
        # Return a simple HTML display of the data
        html = "<h1>Database Test - Predictions</h1>"
        html += f"<p>Found {len(predictions)} predictions</p>"
        html += "<table border='1'><tr><th>ID</th><th>Image</th><th>Patient</th><th>Class</th><th>Confidence</th><th>Timestamp</th></tr>"
        
        for p in predictions:
            html += f"<tr><td>{p['id']}</td><td>{p.get('image_path', 'N/A')}</td><td>{p.get('patient_name', 'N/A')}</td>"
            html += f"<td>{p.get('predicted_class', 'N/A')}</td><td>{p.get('confidence', 'N/A')}</td><td>{p.get('timestamp', 'N/A')}</td></tr>"
        
        html += "</table>"
        
        html += "<h2>Class Probabilities Sample</h2>"
        html += f"<p>Found {len(probabilities)} probability records</p>"
        html += "<table border='1'><tr><th>ID</th><th>Prediction ID</th><th>Class</th><th>Probability</th></tr>"
        
        for p in probabilities:
            html += f"<tr><td>{p['id']}</td><td>{p['prediction_id']}</td><td>{p['class_name']}</td><td>{p['probability']}</td></tr>"
        
        html += "</table>"
        
        return html
    
    except Exception as e:
        return f"<h1>Database Error</h1><p>Error: {str(e)}</p>"

@app.route('/db_viewer')
@login_required
def db_viewer():
    """Database viewer to examine tables and their contents."""
    try:
        # Connect to the database
        connection = get_db_connection()
        
        if not connection:
            return "<h1>Database Error</h1><p>Failed to connect to the database</p>"
        
        cursor = connection.cursor(dictionary=True)
        try:
            # Get list of tables
            cursor.execute("SHOW TABLES")
            tables = [table[f"Tables_in_{connection.database}"] for table in cursor.fetchall()]
            
            # Get data for each table
            tables_data = {}
            table_columns = {}
            
            for table in tables:
                # Get column information
                cursor.execute(f"SHOW COLUMNS FROM {table}")
                columns = [col['Field'] for col in cursor.fetchall()]
                table_columns[table] = columns
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                count = cursor.fetchone()['count']
                
                # Get sample data (up to 10 rows)
                cursor.execute(f"SELECT * FROM {table} LIMIT 10")
                sample_data = cursor.fetchall()
                
                tables_data[table] = {
                    'count': count,
                    'sample_data': sample_data
                }
        finally:
            cursor.close()
            connection.close()
        
        # Return a simple HTML display of the data
        html = "<h1>Database Viewer - embryo_predictions</h1>"
        
        for table in tables:
            html += f"<h2>Table: {table} ({tables_data[table]['count']} rows)</h2>"
            
            if tables_data[table]['count'] > 0:
                html += "<table border='1'><tr>"
                
                # Table headers
                for col in table_columns[table]:
                    html += f"<th>{col}</th>"
                html += "</tr>"
                
                # Table data
                for row in tables_data[table]['sample_data']:
                    html += "<tr>"
                    for col in table_columns[table]:
                        html += f"<td>{row.get(col, '')}</td>"
                    html += "</tr>"
                
                html += "</table>"
            else:
                html += "<p>No data in this table</p>"
        
        return html
    
    except Exception as e:
        return f"<h1>Database Error</h1><p>Error: {str(e)}</p>"

@app.route('/check_predictions')
@login_required
def check_predictions():
    """Check and display patient predictions in a simple format."""
    try:
        # Connect to the database
        connection = get_db_connection()
        
        if not connection:
            return "<h1>Database Error</h1><p>Failed to connect to the database</p>"
        
        cursor = connection.cursor(dictionary=True)
        try:
            # Try multiple ways to get predictions
            
            # Method 1: Standard query
            cursor.execute("""
                SELECT id, image_path, patient_name, predicted_class, confidence, timestamp
                FROM predictions 
                ORDER BY timestamp DESC
            """)
            standard_predictions = cursor.fetchall()
            
            # Method 2: Using GetPredictionsByPatient stored procedure
            # Try with a sample patient name
            cursor.execute("SELECT DISTINCT patient_name FROM predictions LIMIT 1")
            sample_result = cursor.fetchone()
            sample_patient = sample_result['patient_name'] if sample_result else None
            
            patient_predictions = []
            if sample_patient:
                cursor.callproc('GetPredictionsByPatient', [sample_patient])
                patient_predictions = cursor.fetchall()
        finally:
            cursor.close()
            connection.close()
        
        # Create HTML response
        html = "<h1>Patient Predictions Check</h1>"
        
        html += f"<h2>Standard Query Results ({len(standard_predictions)} records)</h2>"
        if standard_predictions:
            html += "<table border='1'><tr><th>ID</th><th>Image</th><th>Patient</th><th>Class</th><th>Confidence</th><th>Timestamp</th></tr>"
            for p in standard_predictions:
                html += f"<tr><td>{p['id']}</td><td>{p.get('image_path', 'N/A')}</td><td>{p.get('patient_name', 'N/A')}</td>"
                html += f"<td>{p.get('predicted_class', 'N/A')}</td><td>{p.get('confidence', 'N/A')}</td><td>{p.get('timestamp', 'N/A')}</td></tr>"
            html += "</table>"
        else:
            html += "<p>No records found with standard query</p>"
        
        if sample_patient:
            html += f"<h2>GetPredictionsByPatient Results for '{sample_patient}' ({len(patient_predictions)} records)</h2>"
            if patient_predictions:
                html += "<table border='1'><tr><th>ID</th><th>Image</th><th>Patient</th><th>Class</th><th>Confidence</th><th>Timestamp</th></tr>"
                for p in patient_predictions:
                    html += f"<tr><td>{p['id']}</td><td>{p.get('image_path', 'N/A')}</td><td>{p.get('patient_name', 'N/A')}</td>"
                    html += f"<td>{p.get('predicted_class', 'N/A')}</td><td>{p.get('confidence', 'N/A')}</td><td>{p.get('timestamp', 'N/A')}</td></tr>"
                html += "</table>"
            else:
                html += "<p>No records found for this patient</p>"
        
        html += "<h2>Debug Information</h2>"
        html += "<p>Check the following:</p>"
        html += "<ul>"
        html += "<li>Are there patient_name values in your records?</li>"
        html += "<li>Do all required columns exist? (id, image_path, patient_name, predicted_class, confidence, timestamp)</li>"
        html += "<li>Check the dashboard for any JavaScript errors (check browser console)</li>"
        html += "</ul>"
        
        html += "<p><a href='/dashboard'>Go to Dashboard</a> | <a href='/db_viewer'>View All Tables</a></p>"
        
        return html
        
    except Exception as e:
        return f"<h1>Error Checking Predictions</h1><p>Error: {str(e)}</p>"

@app.route('/check_uploads')
@login_required
def check_uploads():
    """Check upload directory and files."""
    try:
        # Get upload directory info
        upload_dir = AppConfig.UPLOAD_DIR
        
        # Ensure the directory exists
        os.makedirs(upload_dir, exist_ok=True)
        
        # Check if the directory exists
        dir_exists = os.path.isdir(upload_dir)
        
        # List files in the directory
        if dir_exists:
            files = os.listdir(upload_dir)
            file_info = []
            
            for filename in files:
                file_path = os.path.join(upload_dir, filename)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    file_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                    # Check if it's an image
                    is_image = any(filename.lower().endswith(ext) for ext in AppConfig.ALLOWED_EXTENSIONS)
                    
                    file_info.append({
                        'name': filename,
                        'path': file_path,
                        'size': file_size,
                        'date': file_date,
                        'is_image': is_image
                    })
            
            # Sort by date (newest first)
            file_info.sort(key=lambda x: x['date'], reverse=True)
            
            # Create HTML response
            html = f"<h1>Upload Directory Check</h1>"
            html += f"<p>Directory path: {upload_dir}</p>"
            html += f"<p>Directory exists: {dir_exists}</p>"
            html += f"<p>File count: {len(files)}</p>"
            
            # Display files
            html += f"<h2>Files in Upload Directory</h2>"
            if file_info:
                html += "<table border='1'><tr><th>Name</th><th>Size</th><th>Modified</th><th>Image</th><th>Preview</th></tr>"
                for file in file_info:
                    html += f"<tr><td>{file['name']}</td><td>{file['size']} bytes</td><td>{file['date']}</td>"
                    html += f"<td>{'Yes' if file['is_image'] else 'No'}</td>"
                    if file['is_image']:
                        html += f"<td><img src='/uploads/{file['name']}' style='max-width: 100px; max-height: 100px;'></td>"
                    else:
                        html += f"<td>N/A</td>"
                    html += "</tr>"
                html += "</table>"
            else:
                html += "<p>No files found in upload directory</p>"
                
            # Test image path construction
            html += f"<h2>Image Path Test</h2>"
            if file_info and any(file['is_image'] for file in file_info):
                test_file = next((file for file in file_info if file['is_image']), None)
                if test_file:
                    full_path = test_file['path']
                    filename = test_file['name']
                    url = url_for('uploaded_file', filename=filename)
                    
                    html += f"<p>Test file: {filename}</p>"
                    html += f"<p>Full path: {full_path}</p>"
                    html += f"<p>URL: {url}</p>"
                    html += f"<p>Preview: <img src='{url}' style='max-width: 200px; max-height: 200px;'></p>"
            
            # Path format test for sample database record
            html += f"<h2>Database Image Path Format Test</h2>"
            # Connect to the database
            connection = get_db_connection()
            if connection:
                cursor = connection.cursor(dictionary=True)
                try:
                    # Get a sample record
                    cursor.execute("SELECT id, image_path FROM predictions LIMIT 1")
                    sample = cursor.fetchone()
                    
                    if sample:
                        db_image_path = sample['image_path']
                        db_image_filename = os.path.basename(db_image_path)
                        html += f"<p>Database ID: {sample['id']}</p>"
                        html += f"<p>Database image_path: {db_image_path}</p>"
                        html += f"<p>Extracted filename: {db_image_filename}</p>"
                        
                        # Test if file exists in uploads
                        test_path = os.path.join(upload_dir, db_image_filename)
                        file_exists = os.path.isfile(test_path)
                        html += f"<p>File exists in upload dir: {file_exists}</p>"
                        
                        if file_exists:
                            url = url_for('uploaded_file', filename=db_image_filename)
                            html += f"<p>URL: {url}</p>"
                            html += f"<p>Preview: <img src='{url}' style='max-width: 200px; max-height: 200px;'></p>"
                finally:
                    cursor.close()
                    connection.close()
            
            return html
        else:
            return f"<h1>Error</h1><p>Upload directory does not exist: {upload_dir}</p>"
    
    except Exception as e:
        import traceback
        error = traceback.format_exc()
        return f"<h1>Error</h1><p>Error checking upload directory: {str(e)}</p><pre>{error}</pre>"

if __name__ == '__main__':
    # Ensure uploads directory exists
    print(f"Ensuring uploads directory exists: {AppConfig.UPLOAD_DIR}")
    os.makedirs(AppConfig.UPLOAD_DIR, exist_ok=True)
    
    # List any files in the uploads directory
    upload_files = os.listdir(AppConfig.UPLOAD_DIR) if os.path.exists(AppConfig.UPLOAD_DIR) else []
    print(f"Found {len(upload_files)} files in uploads directory")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
