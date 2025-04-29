import os
import sys
import json

# Get the absolute path to the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBRYO_DIR = os.path.join(SCRIPT_DIR, "embryo-quality-prediction")

# Add the embryo-quality-prediction directory to the Python path
sys.path.append(EMBRYO_DIR)

# Import the predictor class
from src.predict_image import EmbryoPredictor

def test_path_and_patient_name_handling():
    """Test the handling of paths and patient names in the prediction process."""
    # Find a model file
    model_files = []
    models_dir = os.path.join(EMBRYO_DIR, "models")
    for file in os.listdir(models_dir):
        if file.endswith(".pth"):
            model_files.append(os.path.join(models_dir, file))
    
    if not model_files:
        print("No model files found. Please train a model first.")
        return
    
    # Use the first model file
    model_path = model_files[0]
    print(f"Using model: {model_path}")
    
    # Initialize predictor
    predictor = EmbryoPredictor(model_path)
    
    # Find a test image
    test_image = None
    test_dirs = [
        os.path.join(EMBRYO_DIR, "data", "test"),
        os.path.join(EMBRYO_DIR, "data", "sample_images"),
        os.path.join(EMBRYO_DIR, "data"),
        os.path.join(EMBRYO_DIR, "uploads")
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for root, _, files in os.walk(test_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        test_image = os.path.join(root, file)
                        break
                if test_image:
                    break
        if test_image:
            break
    
    if not test_image:
        print("No test images found. Please add some test images.")
        return
    
    print(f"Using test image: {test_image}")
    
    # Test scenarios
    scenarios = [
        {"name": "With patient name", "patient_name": "Test Patient"},
        {"name": "With empty patient name", "patient_name": ""},
        {"name": "With whitespace patient name", "patient_name": "   "},
        {"name": "With None patient name", "patient_name": None}
    ]
    
    # Run tests
    for scenario in scenarios:
        print(f"\n=== Testing {scenario['name']} ===")
        try:
            result = predictor.predict(test_image, scenario['patient_name'])
            print(f"Result patient_name: {result['patient_name']}")
            print(f"Result image_path: {result['image_path']}")
            print(f"Original image_path: {result.get('original_image_path', 'N/A')}")
            
            # Save the prediction
            save_result = predictor.save_prediction(result, save_to_database=True)
            if save_result.get('database_id'):
                print(f"Saved to database with ID: {save_result['database_id']}")
            else:
                print("Not saved to database")
            
            # Check the saved file
            if save_result.get('file_path'):
                print(f"Saved to file: {save_result['file_path']}")
                with open(save_result['file_path'], 'r') as f:
                    saved_data = json.load(f)
                    print(f"File patient_name: {saved_data.get('patient_name')}")
                    print(f"File image_path: {saved_data.get('image_path')}")
                    print(f"File abs_image_path: {saved_data.get('abs_image_path', 'N/A')}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_path_and_patient_name_handling() 