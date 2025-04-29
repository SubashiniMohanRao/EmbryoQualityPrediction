import os
import sys
import json
import glob
import pandas as pd
import uuid
import base64
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from datetime import datetime
import markdown
import bleach

# Get the absolute path to the project root directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.append(PROJECT_ROOT)

# Import evaluation and prediction modules
from src.evaluate_model import ModelEvaluator, find_latest_model
from src.predict_image import EmbryoPredictor
from src.xai_utils import generate_xai_visualization
from src.train_model import get_transforms

app = Flask(__name__)
app.secret_key = 'embryo_quality_prediction_app'

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


@app.route('/')
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
    
    if model_data:
        # Extract class names from the first model's data
        for key in model_data[0].keys():
            if key.endswith('_f1') and not key == 'f1_score':
                class_name = key.replace('_f1', '')
                class_names.append(class_name)
                class_f1_keys.append(key)
        
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
                          class_f1_data=class_f1_data)


@app.route('/evaluate', methods=['POST'])
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
                    result = predictor.predict(file_path)
                    if result is None:
                        raise ValueError("Prediction returned None result")
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
                
                # Save prediction
                try:
                    predictor.save_prediction(result)
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


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(AppConfig.UPLOAD_DIR, filename)


@app.route('/results/<path:filepath>')
def serve_results(filepath):
    """Serve files from the results directory (images, plots, etc.)."""
    # Extract the directory part from the filepath
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    
    # Construct the full directory path
    full_dir_path = os.path.join(AppConfig.RESULTS_DIR, directory)
    
    return send_from_directory(full_dir_path, filename)


@app.route('/docs/<doc_name>')
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
                        
                        # Make prediction
                        print(f"Making prediction for {filename}")
                        result = predictor.predict(file_path)
                        if result is None:
                            raise ValueError(f"Prediction returned None result for {filename}")
                        
                        # Generate XAI visualization
                        try:
                            print(f"Generating XAI visualization")
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
                        
                        # Save prediction
                        try:
                            predictor.save_prediction(result)
                            print(f"Saved prediction")
                        except Exception as e:
                            print(f"Warning: Could not save prediction for {filename}: {e}")
                        
                        # Convert image to base64 for display
                        try:
                            with open(file_path, "rb") as image_file:
                                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                            print(f"Encoded image to base64")
                        except Exception as e:
                            error_msg = f"Error encoding image {filename}: {str(e)}"
                            print(f"Error: {error_msg}")
                            error_messages.append(error_msg)
                            continue
                        
                        # Add result to list
                        results.append({
                            'filename': filename,
                            'result': result,
                            'image_data': encoded_image,
                            'xai_data': xai_result['xai_image'] if xai_result else None
                        })
                        print(f"Added result for {filename}")
                        
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
