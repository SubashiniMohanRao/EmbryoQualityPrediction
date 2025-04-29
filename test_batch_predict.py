import os
import sys
import glob

# Get the absolute path to the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBRYO_DIR = os.path.join(SCRIPT_DIR, "embryo-quality-prediction")

# Add the embryo-quality-prediction directory to the Python path
sys.path.append(EMBRYO_DIR)

# Import the predictor class
from src.predict_image import EmbryoPredictor

def test_batch_prediction():
    """Test batch prediction with various patient name cases."""
    print("Entering test_batch_prediction function...")
    
    try:
        # Find a model file
        model_files = []
        models_dir = os.path.join(EMBRYO_DIR, "models")
        print(f"Looking for models in {models_dir}")
        
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith(".pth"):
                    model_files.append(os.path.join(models_dir, file))
                    print(f"Found model: {file}")
        else:
            print(f"Models directory {models_dir} does not exist")
        
        if not model_files:
            print("No model files found. Please train a model first.")
            return
        
        # Use the first model file
        model_path = model_files[0]
        print(f"Using model: {model_path}")
        
        # Initialize predictor
        predictor = EmbryoPredictor(model_path)
        print("Predictor initialized successfully")
        
        # Create test images - just use the same image multiple times
        # This should work if we found a test image in the previous test
        uploads_dir = os.path.join(EMBRYO_DIR, "uploads")
        data_dir = os.path.join(EMBRYO_DIR, "data")
        
        # Try to find a sample image
        test_image = None
        
        # Search paths in order
        search_paths = [
            uploads_dir,
            data_dir,
            os.path.join(data_dir, "raw"),
            os.path.join(data_dir, "test"),
            os.path.join(data_dir, "sample_images")
        ]
        
        for search_path in search_paths:
            print(f"Searching for images in {search_path}")
            if os.path.exists(search_path):
                # Walk through directories
                for root, _, files in os.walk(search_path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                            test_image = os.path.join(root, file)
                            print(f"Found test image: {test_image}")
                            break
                    if test_image:
                        break
            if test_image:
                break
        
        if not test_image:
            print("No test images found. Please add some test images.")
            return
        
        # Create a list with the same image multiple times
        test_images = [test_image, test_image, test_image]
        
        print(f"Using test image: {test_image} (repeated 3 times)")
        
        # Create test patient names - mix of values and None
        patient_names = ["Test Patient 1", "Test Patient 2", None]
        
        print("\n=== Testing batch prediction with patient names ===")
        
        # Make batch prediction
        results = predictor.predict_batch(test_images, patient_names)
        
        # Check results
        for i, result in enumerate(results):
            if 'error' in result:
                print(f"Error for image {i+1}: {result['error']}")
                continue
                
            print(f"\nResult {i+1}:")
            print(f"Patient name: {result['patient_name']!r}")
            print(f"Image path: {result['image_path']}")
            print(f"Predicted class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.2%}")
            
            # Save the prediction
            save_result = predictor.save_prediction(result, save_to_database=True)
            if save_result.get('database_id'):
                print(f"Saved to database with ID: {save_result['database_id']}")
            else:
                print("Not saved to database")
    except Exception as e:
        print(f"Error in test_batch_prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        print("Starting batch prediction test...")
        test_batch_prediction()
        print("Batch prediction test completed.")
    except Exception as e:
        print(f"Error in batch prediction test: {e}")
        import traceback
        traceback.print_exc() 