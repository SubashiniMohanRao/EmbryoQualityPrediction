#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Embryo Classification Complete Workflow
This script runs the entire embryo classification workflow in sequence:
1. Label data
2. Check image sizes
3. Clean and verify data
4. Normalize images
5. Augment images
6. Split dataset
7. Train model
"""

import os
import sys
import time
import importlib
from os import path
from datetime import datetime
import torch
import argparse

# Import report generator and model report manager
from src.simple_report import SimpleReportGenerator as ReportGenerator
from src.model_report_generator import ModelReportManager

# Import database utilities
from src.setup_database import setup_database

# Import authentication utilities
from src.auth_utils import setup_auth_database

# Check if running in Google Colab
IN_COLAB = 'google.colab' in str(globals())

# Get the absolute path to the project root directory
SCRIPT_DIR = path.dirname(path.abspath(__file__))

# Set project root based on environment
if IN_COLAB:
    try:
        from google.colab import drive
        print("Google Colab detected. Mounting Google Drive...")
        drive.mount('/content/drive')
        
        # Ask user for project location in Google Drive
        print("\nPlease specify the location of the project in Google Drive:")
        print("1. /content/drive/MyDrive/embryo-quality-prediction (default)")
        print("2. /content/embryo-quality-prediction")
        print("3. Specify custom path")
        
        drive_option = input("\nSelect option (1-3) or press Enter for default: ")
        
        if drive_option == "" or drive_option == "1":
            PROJECT_ROOT = "/content/drive/MyDrive/embryo-quality-prediction"
        elif drive_option == "2":
            PROJECT_ROOT = "/content/embryo-quality-prediction"
        elif drive_option == "3":
            custom_path = input("Enter the full path to the project: ")
            PROJECT_ROOT = custom_path
        else:
            PROJECT_ROOT = "/content/drive/MyDrive/embryo-quality-prediction"
            
        print(f"Project root set to: {PROJECT_ROOT}")
        
        # Create project directory if it doesn't exist
        os.makedirs(PROJECT_ROOT, exist_ok=True)
        
        # Install required packages if in Colab
        print("\nInstalling required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision", "tqdm", "opencv-python", "Pillow", "scikit-learn", "matplotlib", "seaborn", "lion-pytorch"])
        print("Package installation complete.")
        
    except ImportError:
        print("Google Colab detected but drive module not available.")
        PROJECT_ROOT = "/content/embryo-quality-prediction"
else:
    PROJECT_ROOT = SCRIPT_DIR

# Add src directory to path for imports
sys.path.append(path.join(PROJECT_ROOT, "src"))

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)
    
def run_module(module_name, module_path=None):
    """Run a module by importing and executing its main code"""
    print_section_header(f"Running {module_name}")
    
    start_time = time.time()
    
    try:
        # If module_path is provided, use it to import the module
        if module_path:
            sys.path.insert(0, path.dirname(module_path))
            module = importlib.import_module(path.basename(module_path).replace('.py', ''))
            sys.path.pop(0)
        else:
            # Otherwise import from src package
            module = importlib.import_module(f"src.{module_name}")
        
        # If the module has a main function, call it
        if hasattr(module, 'main'):
            module.main()
            
        elapsed_time = time.time() - start_time
        print(f"\n✅ {module_name} completed successfully in {elapsed_time:.2f} seconds")
        return True
    except ImportError as ie:
        if IN_COLAB:
            print(f"\n⚠️ Import error in {module_name}: {str(ie)}")
            print("This may be due to missing packages in Colab environment.")
            print("Installing required packages...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision", "tqdm", "opencv-python", "Pillow", "scikit-learn", "matplotlib", "seaborn", "lion-pytorch"])
            print("Retrying import...")
            try:
                if module_path:
                    sys.path.insert(0, path.dirname(module_path))
                    module = importlib.import_module(path.basename(module_path).replace('.py', ''))
                    sys.path.pop(0)
                else:
                    module = importlib.import_module(f"src.{module_name}")
                
                if hasattr(module, 'main'):
                    module.main()
                
                elapsed_time = time.time() - start_time
                print(f"\n✅ {module_name} completed successfully in {elapsed_time:.2f} seconds")
                return True
            except Exception as e2:
                print(f"\n❌ Error in {module_name} after package installation: {str(e2)}")
                return False
        else:
            print(f"\n❌ Import error in {module_name}: {str(ie)}")
            return False
    except Exception as e:
        print(f"\n❌ Error in {module_name}: {str(e)}")
        return False

def main():
    """Run the complete embryo classification workflow"""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Embryo Classification Workflow')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Add setup-db command
    setup_db_parser = subparsers.add_parser('setup-db', help='Setup the database')
    setup_db_parser.add_argument('--host', default='localhost', help='Database host')
    setup_db_parser.add_argument('--user', default='suba', help='Database username')
    setup_db_parser.add_argument('--password', default='Suba@123', help='Database password')
    setup_db_parser.add_argument('--database', default='embryo_predictions', help='Database name')
    
    # Add setup-auth command
    setup_auth_parser = subparsers.add_parser('setup-auth', help='Setup the authentication database')
    setup_auth_parser.add_argument('--host', default='localhost', help='Database host')
    setup_auth_parser.add_argument('--user', default='suba', help='Database username')
    setup_auth_parser.add_argument('--password', default='Suba@123', help='Database password')
    setup_auth_parser.add_argument('--database', default='embryo_predictions', help='Database name')
    
    # Add web command
    web_parser = subparsers.add_parser('web', help='Start the web application')
    web_parser.add_argument('--host', default='0.0.0.0', help='Web server host')
    web_parser.add_argument('--port', type=int, default=5000, help='Web server port')
    web_parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'setup-db':
        print_section_header("DATABASE SETUP")
        success = setup_database(
            host=args.host,
            user=args.user,
            password=args.password,
            database=args.database
        )
        if success:
            print("\n✅ Database setup completed successfully!")
        else:
            print("\n❌ Database setup failed. Check logs for details.")
        return
    
    elif args.command == 'setup-auth':
        print_section_header("AUTHENTICATION DATABASE SETUP")
        success = setup_auth_database(
            host=args.host,
            user=args.user,
            password=args.password,
            database=args.database
        )
        if success:
            print("\n✅ Authentication database setup completed successfully!")
            print("\nTo start the application, run:")
            print(f"python {__file__} web")
            print("\nTo register a new user, open http://localhost:5000/register in your browser")
        else:
            print("\n❌ Authentication database setup failed. Check logs for details.")
        return
    
    elif args.command == 'web':
        print_section_header("STARTING WEB APPLICATION")
        # Import app here to avoid circular imports
        sys.path.append(path.join(PROJECT_ROOT, "app"))
        from app.app import app
        print(f"Starting web server on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug)
        return
    
    # Continue with regular workflow if no command specified
    print_section_header("EMBRYO CLASSIFICATION WORKFLOW")
    print(f"Starting workflow at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {PROJECT_ROOT}")
    
    # Create necessary directories
    os.makedirs(path.join(PROJECT_ROOT, "data", "raw"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "data", "raw", "roboflow"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "data", "raw", "other"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "data", "sorted"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "data", "augmented"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "data", "normalized"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "data", "split"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "models"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "outputs", "plots"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "outputs", "results"), exist_ok=True)
    
    # Ask user for workflow mode
    print_section_header("WORKFLOW MODE SELECTION")
    print("Select workflow mode:")
    print("1. Automatic mode (use default values for all selections)")
    print("2. Interactive mode (prompt for each selection)")
    print("3. Step-by-Step mode (run each workflow step individually)")
    
    # Default to interactive mode
    workflow_mode = "interactive"
    
    while True:
        try:
            mode_selection = input("\nSelect mode (1-3) or press Enter for interactive mode: ")
            if mode_selection == "" or mode_selection == "2":
                workflow_mode = "interactive"
                print("Selected: Interactive mode")
                break
            elif mode_selection == "1":
                workflow_mode = "automatic"
                print("Selected: Automatic mode")
                break
            elif mode_selection == "3":
                workflow_mode = "step_by_step"
                print("Selected: Step-by-Step mode")
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Check if data directories exist and provide dataset selection options
    roboflow_dir = path.join(PROJECT_ROOT, "data", "raw", "roboflow")
    other_dir = path.join(PROJECT_ROOT, "data", "raw", "other")
    
    roboflow_has_data = os.path.exists(roboflow_dir) and len(os.listdir(roboflow_dir)) > 0
    other_has_data = os.path.exists(other_dir) and len(os.listdir(other_dir)) > 0
    
    # Set environment variable for dataset selection
    os.environ["EMBRYO_DATASET_SELECTION"] = ""
    
    if not roboflow_has_data and not other_has_data:
        print("\n⚠️ Warning: No data found in either raw/roboflow or raw/other directories.")
        print("Please ensure you have placed your data in at least one of these directories before proceeding.")
        user_input = input("Do you want to continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            print("Workflow stopped due to missing data")
            return
    
    # Function to handle dataset selection
    def select_dataset():
        available_options = []
        if roboflow_has_data:
            available_options.append("roboflow")
        
        if other_has_data:
            available_options.append("other")
        
        if roboflow_has_data and other_has_data:
            available_options.append("both")
        
        # In automatic mode, choose the most comprehensive option available
        if workflow_mode == "automatic":
            if "both" in available_options:
                dataset_choice = "both"
            elif "roboflow" in available_options:
                dataset_choice = "roboflow"
            elif "other" in available_options:
                dataset_choice = "other"
            else:
                dataset_choice = ""
            
            print(f"\nAutomatic mode selected: {dataset_choice.capitalize() if dataset_choice else 'No'} dataset")
            os.environ["EMBRYO_DATASET_SELECTION"] = dataset_choice
            return True
        else:
            # Interactive mode - prompt user for selection
            print("\n" + "=" * 50)
            print("DATASET SELECTION")
            print("=" * 50)
            
            if roboflow_has_data:
                print("1. Roboflow dataset (original dataset)")
            
            if other_has_data:
                print(f"{2 if roboflow_has_data else 1}. Other dataset (new dataset)")
            
            if roboflow_has_data and other_has_data:
                print("3. Both datasets")
            
            while True:
                try:
                    selection = input("\nSelect which dataset to use (enter the number): ")
                    selection_idx = int(selection) - 1
                    
                    if 0 <= selection_idx < len(available_options):
                        dataset_choice = available_options[selection_idx]
                        print(f"Selected: {dataset_choice.capitalize()} dataset")
                        os.environ["EMBRYO_DATASET_SELECTION"] = dataset_choice
                        return True
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
    
    # Only select dataset now if not in step-by-step mode
    if workflow_mode != "step_by_step":
        select_dataset()
    
    # Define model selection function
    def select_model():
        model_mapping = {
            "1": "resnet152",
            "2": "densenet201",
            "3": "efficientnet_b7",
            "4": "convnext_base",
            "5": "swinv2",
            "6": "efficientvit"
        }
        
        # Always show model selection options, regardless of mode
        print_section_header("MODEL SELECTION")
        print("Select a model architecture for training:")
        print("1. ResNet152 (default)")
        print("2. DenseNet201")
        print("3. EfficientNet-B7")
        print("4. ConvNeXt Base")
        print("5. SwinV2")
        print("6. EfficientViT")
        
        if workflow_mode == "automatic":
            # In automatic mode, still show options but use default without prompting
            model_choice = "resnet152"
            print(f"\nAutomatic mode selected: {model_choice} model")
        else:
            # Interactive mode - prompt user for selection
            while True:
                try:
                    selection = input("\nSelect which model to use (enter the number or press Enter for default): ")
                    if selection == "":
                        model_choice = "resnet152"  # Default
                        break
                    
                    if selection in model_mapping:
                        model_choice = model_mapping[selection]
                        break
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
            
            print(f"Selected model: {model_choice}")
        
        os.environ["EMBRYO_MODEL_SELECTION"] = model_choice
        return True
        
    # Function to check GPU and confirm training
    def check_gpu_for_training():
        print_section_header("GPU CHECK")
        print("Checking GPU availability...")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
            print("✅ GPU is available for training")
            return True
        else:
            print("⚠️ WARNING: No GPU available. Training will be slow on CPU.")
            
            # In automatic mode, proceed with CPU training
            if workflow_mode == "automatic":
                print("Automatic mode: Proceeding with CPU training")
                return True
            else:
                # In interactive mode, ask for confirmation
                user_input = input("Do you want to continue with CPU training? (y/n): ")
                if user_input.lower() != 'y':
                    print("Training step skipped due to no GPU availability")
                    return False
                return True
    
    # Define all workflow steps (combining both phases)
    workflow_steps = [
        {"name": "Select dataset", "module": None, "function": select_dataset, "error_msg": "Workflow stopped due to error in dataset selection", "phase": "Data Preparation"},
        {"name": "Label data", "module": "label", "function": None, "error_msg": "Workflow stopped due to error in label.py", "phase": "Data Preparation"},
        {"name": "Check image sizes", "module": "check_image_size", "function": None, "error_msg": "Workflow stopped due to error in check_image_size.py", "phase": "Data Preparation"},
        {"name": "Clean and verify data", "module": "CleanAndVerify", "function": None, "error_msg": "Workflow stopped due to error in CleanAndVerify.py", "phase": "Data Preparation"},
        {"name": "Augment images", "module": "imgaug", "function": None, "error_msg": "Workflow stopped due to error in imgaug.py", "phase": "Data Preparation"},
        {"name": "Normalize images", "module": "normalize", "function": None, "error_msg": "Workflow stopped due to error in normalize.py", "phase": "Data Preparation"},
        {"name": "Split dataset", "module": "split_dataset", "function": None, "error_msg": "Workflow stopped due to error in split_dataset.py", "phase": "Data Preparation"},
        {"name": "Check GPU availability", "module": None, "function": check_gpu_for_training, "error_msg": "Workflow stopped due to GPU check failure", "phase": "Model Development"},
        {"name": "Select model architecture", "module": None, "function": select_model, "error_msg": "Workflow stopped due to error in model selection", "phase": "Model Development"},
        {"name": "Train model", "module": "train_model", "function": None, "error_msg": "Workflow stopped due to error in train_model.py", "phase": "Model Development"}
    ]
    
    # Print workflow phases
    print_section_header("WORKFLOW PHASES")
    
    if workflow_mode == "step_by_step":
        # Step-by-Step mode: Ask user which step to run
        print("\nSelect a workflow step to run:")
        for i, step in enumerate(workflow_steps, 1):
            print(f"{i}. {step['name']} ({step['phase']})")
        print(f"{len(workflow_steps) + 1}. Run all steps in sequence")
        
        while True:
            try:
                step_selection = input("\nEnter step number to run (or 'q' to quit): ")
                if step_selection.lower() == 'q':
                    print("Workflow stopped by user")
                    return
                
                step_idx = int(step_selection) - 1
                if 0 <= step_idx < len(workflow_steps):
                    # Run single step
                    step = workflow_steps[step_idx]
                    print(f"\nRunning step: {step['name']}")
                    
                    # Execute the step (either a module or a function)
                    success = False
                    if step['module'] is not None:
                        success = run_module(step['module'])
                    elif step['function'] is not None:
                        success = step['function']()
                    
                    if not success:
                        print(step['error_msg'])
                        return
                        
                    print(f"\n✅ Step completed: {step['name']}")
                    continue  # Ask for next step
                elif step_idx == len(workflow_steps):
                    # Run all steps
                    print("\nRunning all steps in sequence...")
                    break  # Exit loop and continue with all steps
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    # Run all steps in sequence for automatic, interactive, or if "run all" was selected in step-by-step mode
    if workflow_mode != "step_by_step" or step_idx == len(workflow_steps):
        # Skip the dataset selection step since we already did it for non-step-by-step modes
        steps_to_run = workflow_steps[1:] if workflow_mode != "step_by_step" else workflow_steps
        
        for step in steps_to_run:
            if step['module'] is not None:
                if not run_module(step['module']):
                    print(step['error_msg'])
                    return
            elif step['function'] is not None:
                if not step['function']():
                    print(step['error_msg'])
                    return
    
    # Verify split directories exist and contain data
    split_dir = path.join(PROJECT_ROOT, "data", "split")
    train_dir = path.join(split_dir, "train")
    val_dir = path.join(split_dir, "val")
    test_dir = path.join(split_dir, "test")
    
    # Check if directories exist and contain class subdirectories
    train_has_classes = path.exists(train_dir) and len(os.listdir(train_dir)) > 0
    val_has_classes = path.exists(val_dir) and len(os.listdir(val_dir)) > 0
    test_has_classes = path.exists(test_dir) and len(os.listdir(test_dir)) > 0
    
    if not (train_has_classes and val_has_classes and test_has_classes):
        print("\n⚠️ Warning: Split directories are missing or empty. Running split_dataset again...")
        if not run_module("split_dataset"):
            print("Workflow stopped due to error in split_dataset.py")
            return
        
        # Double-check after running split_dataset
        train_has_classes = path.exists(train_dir) and len(os.listdir(train_dir)) > 0
        val_has_classes = path.exists(val_dir) and len(os.listdir(val_dir)) > 0
        test_has_classes = path.exists(test_dir) and len(os.listdir(test_dir)) > 0
        
        if not (train_has_classes and val_has_classes and test_has_classes):
            print("\n❌ Error: Split directories still missing or empty after running split_dataset.py")
            print("Please ensure you have data in the 'data/augmented' directory")
            return
        else:
            print("\n✅ Split directories successfully created and contain data.")
    
    # For non-step-by-step modes, we need to run the phases in order
    if workflow_mode != "step_by_step":
        # Phase 1: Data Preparation
        print_section_header("PHASE 1: DATA PREPARATION")
        
        # Run all data preparation steps
        data_prep_steps = [step for step in workflow_steps if step['phase'] == "Data Preparation" and step['name'] != "Select dataset"]
        for step in data_prep_steps:
            if step['module'] is not None:
                if not run_module(step['module']):
                    print(step['error_msg'])
                    return
            elif step['function'] is not None:
                if not step['function']():
                    print(step['error_msg'])
                    return
        
        # Verify split directories exist and contain data
        split_dir = path.join(PROJECT_ROOT, "data", "split")
        train_dir = path.join(split_dir, "train")
        val_dir = path.join(split_dir, "val")
        test_dir = path.join(split_dir, "test")
        
        # Check if directories exist and contain class subdirectories
        train_has_classes = path.exists(train_dir) and len(os.listdir(train_dir)) > 0
        val_has_classes = path.exists(val_dir) and len(os.listdir(val_dir)) > 0
        test_has_classes = path.exists(test_dir) and len(os.listdir(test_dir)) > 0
        
        if not (train_has_classes and val_has_classes and test_has_classes):
            print("\n⚠️ Warning: Split directories are missing or empty. Running split_dataset again...")
            if not run_module("split_dataset"):
                print("Workflow stopped due to error in split_dataset.py")
                return
            
            # Double-check after running split_dataset
            train_has_classes = path.exists(train_dir) and len(os.listdir(train_dir)) > 0
            val_has_classes = path.exists(val_dir) and len(os.listdir(val_dir)) > 0
            test_has_classes = path.exists(test_dir) and len(os.listdir(test_dir)) > 0
            
            if not (train_has_classes and val_has_classes and test_has_classes):
                print("\n❌ Error: Split directories still missing or empty after running split_dataset.py")
                print("Please ensure you have data in the 'data/augmented' directory")
                return
            else:
                print("\n✅ Split directories successfully created and contain data.")
        
        # Phase 2: Model Development
        print_section_header("PHASE 2: MODEL DEVELOPMENT")
        
        # Always select model first, regardless of mode
        select_model()
        
        # Check GPU availability before training
        if not check_gpu_for_training():
            return
        
        # Train model
        if not run_module("train_model"):
            print("Workflow stopped due to error in train_model.py")
            return
    
    # Generate final report
    print_section_header("GENERATING FINAL REPORT")
    try:
        # Get the model name from environment variable
        model_name = os.environ.get("EMBRYO_MODEL_SELECTION", "model")
        
        # Create timestamp for this report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize report generator with model name
        report_generator = ReportGenerator(PROJECT_ROOT, model_name=model_name)
        
        # Get the report path from the generator
        report_path = report_generator.report_path
        
        print(f"✅ Final report generated successfully")
        print(f"📊 Report saved to: {report_path}")
        
        # Generate a summary of all models
        try:
            print("\nGenerating model comparison summary...")
            model_manager = ModelReportManager(PROJECT_ROOT)
            models_df = model_manager.list_models(sort_by="accuracy", ascending=False)
            print(f"✅ Model summary updated with {len(models_df)} models")
        except Exception as summary_err:
            print(f"⚠️ Warning: Could not generate model summary: {str(summary_err)}")
        
        # Open the report in browser or display in Colab
        if IN_COLAB:
            try:
                from google.colab import files
                from IPython.display import HTML, display
                print("\nTo view the report in Colab, you can:")
                print("1. Download the HTML file using the link below")
                print("2. Or view it directly in the output cell")
                
                # Provide download link
                files.download(report_path)
                
                # Also try to display the HTML directly
                with open(report_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                display(HTML(html_content))
                print("\nReport displayed above and available for download")
            except Exception as colab_err:
                print(f"Note: Could not display report in Colab: {str(colab_err)}")
                print(f"The report is still saved at: {report_path}")
        else:
            import webbrowser
            print("Opening report in browser...")
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
    except Exception as e:
        print(f"⚠️ Warning: Could not generate final report: {str(e)}")
    
    # Workflow complete
    print_section_header("WORKFLOW COMPLETE")
    print(f"Workflow completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("All steps executed successfully!")
    
    # Provide information about viewing model reports
    print("\nTo view and compare all model reports, run:")
    print(f"python {os.path.join('src', 'model_report_generator.py')}")

def predict_image(image_path, patient_name=None, model_path=None, output_dir=None, save_to_db=True, db_config=None):
    """Predict the quality of an embryo image and save results.
    
    Args:
        image_path (str): Path to the image file
        patient_name (str, optional): Name of the patient
        model_path (str, optional): Path to the model file
        output_dir (str, optional): Directory to save results
        save_to_db (bool, optional): Whether to save to database
        db_config (dict, optional): Database configuration
        
    Returns:
        dict: Prediction results
    """
    print_section_header("EMBRYO QUALITY PREDICTION")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return None
        
    try:
        # Import predictor class
        from src.predict_image import EmbryoPredictor
        
        # Initialize predictor
        print(f"Initializing predictor with model: {model_path or 'latest model'}")
        predictor = EmbryoPredictor(model_path)
        
        # Make prediction
        print(f"Predicting image: {image_path}")
        if patient_name:
            print(f"Patient: {patient_name}")
        result = predictor.predict(image_path, patient_name)
        
        # Prepare database config
        if db_config is None and save_to_db:
            db_config = {
                'host': 'localhost',
                'user': 'suba',
                'password': 'Suba@123',
                'database': 'embryo_predictions'
            }
        
        # Save prediction
        save_result = predictor.save_prediction(
            result, 
            output_dir, 
            save_to_database=save_to_db,
            db_config=db_config
        )
        
        # Print results
        print("\n===== Prediction Results =====")
        print(f"Image: {result['image_path']}")
        if result.get('patient_name'):
            print(f"Patient: {result['patient_name']}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        if save_result:
            print(f"Results saved to: {save_result.get('file_path', 'N/A')}")
            if save_result.get('database_id'):
                print(f"Database record ID: {save_result['database_id']}")
        
        return result
    except Exception as e:
        print(f"❌ Error predicting image: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run embryo quality prediction workflow')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Workflow command (default)
    workflow_parser = subparsers.add_parser('workflow', help='Run the embryo quality prediction workflow')
    workflow_parser.add_argument('--test', action='store_true', help='Run in test mode with just 1 epoch')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict embryo quality from an image')
    predict_parser.add_argument('--image', type=str, required=True, help='Path to image file')
    predict_parser.add_argument('--patient', type=str, help='Patient name')
    predict_parser.add_argument('--model', type=str, help='Path to model file')
    predict_parser.add_argument('--output', type=str, help='Output directory for prediction results')
    predict_parser.add_argument('--no-db', action='store_true', help='Do not save to database')
    predict_parser.add_argument('--db-host', type=str, default='localhost', help='Database host')
    predict_parser.add_argument('--db-user', type=str, default='suba', help='Database username')
    predict_parser.add_argument('--db-password', type=str, default='Suba@123', help='Database password')
    predict_parser.add_argument('--db-name', type=str, default='embryo_predictions', help='Database name')
    
    # Setup database command
    setup_db_parser = subparsers.add_parser('setup-db', help='Set up the database for storing predictions')
    setup_db_parser.add_argument('--host', type=str, default='localhost', help='Database host')
    setup_db_parser.add_argument('--user', type=str, default='suba', help='Database username')
    setup_db_parser.add_argument('--password', type=str, default='Suba@123', help='Database password')
    setup_db_parser.add_argument('--database', type=str, default='embryo_predictions', help='Database name')
    
    # Setup authentication command
    setup_auth_parser = subparsers.add_parser('setup-auth', help='Set up the authentication database')
    setup_auth_parser.add_argument('--host', type=str, default='localhost', help='Database host')
    setup_auth_parser.add_argument('--user', type=str, default='suba', help='Database username')
    setup_auth_parser.add_argument('--password', type=str, default='Suba@123', help='Database password')
    setup_auth_parser.add_argument('--database', type=str, default='embryo_predictions', help='Database name')
    
    # Web command
    web_parser = subparsers.add_parser('web', help='Start the web application')
    web_parser.add_argument('--host', default='0.0.0.0', help='Web server host')
    web_parser.add_argument('--port', type=int, default=5000, help='Web server port')
    web_parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    try:
        # Handle different commands
        if args.command == 'predict':
            # Configure database
            db_config = {
                'host': args.db_host,
                'user': args.db_user,
                'password': args.db_password,
                'database': args.db_name
            }
            
            # Run prediction
            predict_image(args.image, args.patient, args.model, args.output, not args.no_db, db_config)
        elif args.command == 'setup-db':
            # Import and run database setup
            from src.setup_database import setup_database
            setup_database(args.host, args.user, args.password, args.database)
        elif args.command == 'setup-auth':
            # Import and run authentication setup
            from src.auth_utils import setup_auth_database
            setup_auth_database(args.host, args.user, args.password, args.database)
        elif args.command == 'web':
            # Import and run web application
            sys.path.append(path.join(PROJECT_ROOT, "app"))
            from app.app import app
            app.run(host=args.host, port=args.port, debug=args.debug)
        else:  # Default to workflow
            # If in test mode, override Config epochs
            if hasattr(args, 'test') and args.test:
                print("\n🧪 RUNNING IN TEST MODE - ONLY 1 EPOCH\n")
                # Import and modify Config
                from src.train_model import Config
                Config.num_epochs = 1
                print(f"Training will run for {Config.num_epochs} epoch only")
            
            # Run the main workflow
            main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        import traceback
        print(f"\nUnexpected error: {str(e)}")
        traceback.print_exc()
        print("\nIf you're running in Google Colab and encountering package-related errors,")
        print("try running the following commands:")
        print("python -m pip install --force-reinstall blinker")
        print("python -m pip install --ignore-installed torch torchvision tqdm opencv-python Pillow scikit-learn matplotlib seaborn lion-pytorch mysql-connector-python")
