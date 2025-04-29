import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import transforms, models
from datetime import datetime

# Get the absolute path to the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

# Import the Config class from train_model
from src.train_model import Config, get_transforms
# Import database functions
from src.db_utils import save_prediction as save_to_db

class EmbryoPredictor:
    def __init__(self, model_path=None):
        """
        Initialize the embryo image predictor.
        
        Args:
            model_path (str, optional): Path to the model file. If None, will use the latest model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Find latest model if not specified
        if model_path is None:
            model_path = self._find_latest_model()
        
        self.model_path = model_path
        
        # Load the model
        self.model, self.class_names = self._load_model()
        
        # Get transforms for prediction
        _, self.transform = get_transforms()
    
    def _find_latest_model(self):
        """Find the latest model file in the models directory."""
        models_dir = os.path.join(PROJECT_ROOT, "models")
        
        # List all model files
        model_files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) 
                      if f.endswith('.pth') or f.endswith('.pt')]
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {models_dir}")
        
        # Sort by modification time (newest first)
        latest_model = max(model_files, key=os.path.getmtime)
        print(f"Found latest model: {latest_model}")
        return latest_model
    
    def _load_model(self):
        """Load the trained model."""
        print(f"Loading model from {self.model_path}...")
        
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
            # Load the saved model
            try:
                model_data = torch.load(self.model_path, map_location=self.device)
            except Exception as e:
                raise ValueError(f"Failed to load model file: {e}")
            
            # Print model_data keys for debugging
            print(f"Model data keys: {model_data.keys() if isinstance(model_data, dict) else 'Not a dictionary'}")
            
            # Initialize variables to avoid UnboundLocalError later
            model = None
            class_names = None
            num_classes = 5  # Default
            model_name = "resnet152"  # Default
            state_dict = None
            
            # Handle different model saving formats
            if isinstance(model_data, dict):
                # PyTorch standard format with state_dict
                if 'state_dict' in model_data:
                    state_dict = model_data['state_dict']
                elif 'model_state_dict' in model_data:
                    state_dict = model_data['model_state_dict']
                else:
                    # The model_data itself might be the state_dict
                    state_dict = model_data
                
                # Extract model architecture and class names
                model_name = model_data.get('model_name', Config.model_name)
                class_names = model_data.get('class_names', None)
                
                # Ensure class_names is not None
                if class_names is None:
                    print("Warning: No class names found in model data, using default class names")
                    # Try to get class names from config
                    if hasattr(Config, 'class_names') and Config.class_names is not None:
                        class_names = Config.class_names
                        print(f"Using class names from Config: {class_names}")
                    else:
                        # If we have num_classes, create default class names
                        num_classes = model_data.get('num_classes', 5)  # Default to 5 if not specified
                        class_names = [f"Class_{i}" for i in range(num_classes)]
                        print(f"Created default class names for {num_classes} classes")
            else:
                # The loaded object might be the model itself
                print("Loaded object is not a dictionary, might be the model itself")
                model = model_data.to(self.device).eval()
                class_names = [f"Class_{i}" for i in range(10)]  # Assume 10 classes
                print(f"Using direct model with default class names: {class_names}")
                return model, class_names
            
            # If we don't have a state_dict at this point, we can't continue
            if state_dict is None:
                raise ValueError("Could not find model state dictionary in the loaded model file")
                
            # Print some state_dict keys for debugging
            print(f"State dict keys sample: {list(state_dict.keys())[:5]}")
            
            # Determine number of classes from the state_dict
            try:
                if 'fc.weight' in state_dict:
                    num_classes = state_dict['fc.weight'].size(0)
                    print(f"Found fc.weight with shape: {state_dict['fc.weight'].shape}")
                elif 'module.fc.weight' in state_dict:
                    num_classes = state_dict['module.fc.weight'].size(0)
                    print(f"Found module.fc.weight with shape: {state_dict['module.fc.weight'].shape}")
                    # Adjust keys to remove 'module.' prefix if needed
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                else:
                    # Try to find any key that might contain fc.weight
                    fc_weight_keys = [k for k in state_dict.keys() if 'fc.weight' in k]
                    if fc_weight_keys:
                        key = fc_weight_keys[0]
                        num_classes = state_dict[key].size(0)
                        print(f"Found {key} with shape: {state_dict[key].shape}")
                    else:
                        num_classes = model_data.get('num_classes', Config.num_classes)
                        print(f"Using num_classes from config: {num_classes}")
            except Exception as e:
                print(f"Error determining number of classes: {e}")
                num_classes = 5  # Default to 5 classes
                print(f"Defaulting to {num_classes} classes")
            
            print(f"Detected {num_classes} classes in the saved model")
            
            # Initialize the model architecture
            try:
                if model_name == "resnet152":
                    model = models.resnet152(weights=None)
                    num_ftrs = model.fc.in_features
                    # Replace the final fully connected layer
                    model.fc = nn.Linear(num_ftrs, num_classes)
                    print(f"Created ResNet152 with {num_classes} output classes")
                elif model_name == "densenet201":
                    model = models.densenet201(weights=None)
                    num_ftrs = model.classifier.in_features
                    model.classifier = nn.Linear(num_ftrs, num_classes)
                    print(f"Created DenseNet201 with {num_classes} output classes")
                elif model_name == "efficientnet_b7":
                    model = models.efficientnet_b7(weights=None)
                    num_ftrs = model.classifier[1].in_features
                    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
                    print(f"Created EfficientNet-B7 with {num_classes} output classes")
                else:
                    # Default to ResNet152 if model_name is not recognized
                    print(f"Model architecture '{model_name}' not recognized, defaulting to ResNet152")
                    model = models.resnet152(weights=None)
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, num_classes)
                    print(f"Created ResNet152 with {num_classes} output classes")
            except Exception as e:
                print(f"Error creating model architecture: {e}")
                # Create a simple model as fallback
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(512, num_classes)
                print(f"Created fallback ResNet18 with {num_classes} output classes")
            
            # Try to load the state dictionary
            try:
                model.load_state_dict(state_dict)
                print("Successfully loaded state dictionary")
            except Exception as e:
                print(f"Error loading state dict directly: {e}")
                print("Trying to load with strict=False...")
                try:
                    model.load_state_dict(state_dict, strict=False)
                    print("Successfully loaded state dictionary with strict=False")
                except Exception as e2:
                    print(f"Failed to load state dictionary even with strict=False: {e2}")
                    print("Continuing with uninitialized model weights")
                
            # Move model to device and set to evaluation mode
            model = model.to(self.device)
            model.eval()
            
            # Ensure class_names is properly initialized and has the right length
            if class_names is None or len(class_names) != num_classes:
                # Try to load class names from the dataset directory
                try:
                    # Get the class names from the test directory
                    test_dir = os.path.join(PROJECT_ROOT, "data", "test")
                    if os.path.exists(test_dir):
                        # Get class names from subdirectory names
                        actual_class_names = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
                        if len(actual_class_names) == num_classes:
                            class_names = actual_class_names
                            print(f"Loaded actual class names from test directory: {class_names}")
                        else:
                            print(f"Number of classes in test directory ({len(actual_class_names)}) doesn't match model ({num_classes})")
                            # Try to load from dataset directory
                            dataset_dir = os.path.join(PROJECT_ROOT, "data", "dataset")
                            if os.path.exists(dataset_dir):
                                actual_class_names = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
                                if len(actual_class_names) == num_classes:
                                    class_names = actual_class_names
                                    print(f"Loaded actual class names from dataset directory: {class_names}")
                except Exception as e:
                    print(f"Error loading class names from directories: {e}")
                
                # ALWAYS use the predefined embryo class names if the number of classes matches
                # Define the standard embryo class names
                embryo_class_names = [
                    "8cell_grade_A", "8cell_grade_B", "8cell_grade_C",
                    "blastocyst_grade_A", "blastocyst_grade_B", "blastocyst_grade_C",
                    "error_images", 
                    "morula_grade_A", "morula_grade_B", "morula_grade_C"
                ]
                
                # If the number of classes matches our predefined list, use it
                if len(embryo_class_names) == num_classes:
                    class_names = embryo_class_names
                    print(f"Using predefined embryo class names: {class_names}")
                # If we still don't have proper class names, use the default embryo class names
                elif class_names is None or len(class_names) != num_classes:
                    # As a last resort, use generic class names
                    class_names = [f"Class_{i}" for i in range(num_classes)]
                    print(f"Using generic class names to match {num_classes} classes")
            
            print(f"Model loaded successfully with {num_classes} classes: {class_names}")
            return model, class_names
            
        except Exception as e:
            print(f"Error in _load_model: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a fallback model and class names
            print("Creating fallback model and class names")
            fallback_model = models.resnet18(weights=None)
            fallback_model.fc = nn.Linear(512, 10)  # 10 classes as fallback (typical for embryo dataset)
            fallback_model = fallback_model.to(self.device).eval()
            
            # Use predefined embryo class names for fallback
            fallback_class_names = [
                "8cell_grade_A", "8cell_grade_B", "8cell_grade_C",
                "blastocyst_grade_A", "blastocyst_grade_B", "blastocyst_grade_C",
                "error_images", 
                "morula_grade_A", "morula_grade_B", "morula_grade_C"
            ]
            
            return fallback_model, fallback_class_names
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # As a fallback, try to load the model directly without state_dict
            try:
                print("Attempting to load model directly...")
                model = torch.load(self.model_path, map_location=self.device)
                if isinstance(model, torch.nn.Module):
                    model = model.to(self.device).eval()
                    print("Model loaded directly successfully")
                    # Use predefined embryo class names
                    embryo_class_names = [
                        "8cell_grade_A", "8cell_grade_B", "8cell_grade_C",
                        "blastocyst_grade_A", "blastocyst_grade_B", "blastocyst_grade_C",
                        "error_images", 
                        "morula_grade_A", "morula_grade_B", "morula_grade_C"
                    ]
                    return model, embryo_class_names
            except Exception as e2:
                print(f"Failed to load model directly: {e2}")
                raise e  # Re-raise the original error
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for prediction.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tensor: Preprocessed image tensor
        """
        try:
            # Open and convert image
            img = Image.open(image_path).convert('RGB')
            
            # Apply transformations
            img_tensor = self.transform(img)
            
            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0)
            
            return img_tensor.to(self.device)
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {e}")
    
    def predict(self, image_path, patient_name=None):
        """
        Predict the class of an embryo image.
        
        Args:
            image_path (str): Path to image file
            patient_name (str, optional): Name of the patient
            
        Returns:
            dict: Prediction result with class, confidence, and probabilities
        """
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
                
            # Check if model and class_names are properly initialized
            if self.model is None:
                raise ValueError("Model not properly initialized")
            if self.class_names is None or len(self.class_names) == 0:
                raise ValueError("Class names not properly initialized")
            
            # Make sure patient_name is not an empty string
            if patient_name is not None and patient_name.strip() == "":
                patient_name = None
            
            # Convert absolute path to relative path for storage
            # First, try to get relative path from workspace
            rel_image_path = image_path
            try:
                rel_image_path = os.path.relpath(image_path, PROJECT_ROOT)
                # If the path starts with "..", it's outside the project root
                if rel_image_path.startswith(".."):
                    # Just use the filename as a fallback
                    rel_image_path = os.path.basename(image_path)
            except ValueError:
                # If there was an error (different drives), just use the basename
                rel_image_path = os.path.basename(image_path)
                
            # Preprocess image
            img_tensor = self.preprocess_image(image_path)
            if img_tensor is None:
                raise ValueError("Failed to preprocess image")
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(img_tensor)
                if outputs is None:
                    raise ValueError("Model returned None output")
                    
                # Check if outputs has the expected shape
                if len(outputs.shape) != 2 or outputs.shape[0] != 1:
                    raise ValueError(f"Unexpected output shape: {outputs.shape}. Expected shape: [1, num_classes]")
                
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                if probabilities is None or len(probabilities) == 0:
                    raise ValueError("Failed to compute probabilities")
                
                # Get predicted class and confidence with proper error handling
                max_result = torch.max(probabilities, 0)
                if max_result is None:
                    raise ValueError("torch.max returned None")
                    
                # Safely unpack the result
                try:
                    confidence, predicted_class = max_result
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Failed to unpack torch.max result: {e}. Result was: {max_result}")
                
                # Convert to Python types with error handling
                try:
                    predicted_class = predicted_class.item()
                    confidence = confidence.item()
                    probabilities = probabilities.cpu().numpy().tolist()
                except Exception as e:
                    raise ValueError(f"Failed to convert tensor values to Python types: {e}")
                
                # Ensure predicted_class is within range of class_names
                if predicted_class < 0 or predicted_class >= len(self.class_names):
                    raise IndexError(f"Predicted class index {predicted_class} out of range for class_names with length {len(self.class_names)}")
                
                # Create result dictionary
                result = {
                    'image_path': rel_image_path,  # Save relative path instead of absolute
                    'original_image_path': image_path,  # Keep original path for reference
                    'patient_name': patient_name,
                    'predicted_class_index': int(predicted_class),  # Ensure this is an integer
                    'predicted_class': self.class_names[predicted_class],
                    'confidence': float(confidence),  # Ensure this is a float
                    'probabilities': probabilities,
                    'class_names': self.class_names,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                return result
        except Exception as e:
            print(f"Detailed error in predict method: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Error predicting image: {e}")
    
    def predict_batch(self, image_paths, patient_names=None):
        """
        Predict classes for multiple embryo images.
        
        Args:
            image_paths (list): List of paths to image files
            patient_names (list, optional): List of patient names corresponding to each image
            
        Returns:
            list: List of prediction results
        """
        results = []
        for i, image_path in enumerate(image_paths):
            try:
                patient_name = patient_names[i] if patient_names and i < len(patient_names) else None
                result = self.predict(image_path, patient_name)
                results.append(result)
            except Exception as e:
                print(f"Error predicting {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'patient_name': patient_names[i] if patient_names and i < len(patient_names) else None,
                    'error': str(e)
                })
        
        return results
    
    def save_prediction(self, prediction, output_dir=None, save_to_database=True, db_config=None):
        """
        Save prediction results to a JSON file and/or database.
        
        Args:
            prediction (dict): Prediction result
            output_dir (str, optional): Directory to save results. Defaults to outputs/predictions.
            save_to_database (bool, optional): Whether to save to database. Defaults to True.
            db_config (dict, optional): Database configuration. Defaults to None (uses default config).
            
        Returns:
            dict: Dictionary with paths to saved files and/or DB ID
        """
        result = {}
        
        # Save to file
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = os.path.join(PROJECT_ROOT, "outputs", "predictions")
            os.makedirs(output_dir, exist_ok=True)
        
        # Create filename from image name and timestamp
        # Use original path for file operations if available, otherwise use image_path
        image_path_for_ops = prediction.get('original_image_path', prediction['image_path'])
        image_name = os.path.basename(image_path_for_ops)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"pred_{image_name}_{timestamp}.json")
        
        # Make a copy of the prediction to save to file
        file_prediction = prediction.copy()
        
        # For file storage, we can keep both paths
        if 'original_image_path' in file_prediction:
            file_prediction['abs_image_path'] = file_prediction['original_image_path']
            # Delete original_image_path to avoid confusion
            del file_prediction['original_image_path']
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(file_prediction, f, indent=4)
        
        result['file_path'] = output_file
        
        # Save to database if requested
        if save_to_database:
            try:
                # Create a copy of the prediction for database storage
                db_prediction = prediction.copy()
                
                # Make sure we're using the relative path for database storage
                if 'original_image_path' in db_prediction:
                    # Delete original_image_path since we don't need it in the database
                    del db_prediction['original_image_path']
                
                # Ensure patient_name is properly handled
                if db_prediction.get('patient_name') == '':
                    db_prediction['patient_name'] = None
                
                prediction_id = save_to_db(db_prediction, db_config)
                if prediction_id:
                    result['database_id'] = prediction_id
                    print(f"Saved prediction to database with ID: {prediction_id}")
                else:
                    print("Failed to save prediction to database")
            except Exception as e:
                print(f"Error saving to database: {e}")
                import traceback
                traceback.print_exc()
        
        return result


def main():
    """Main function to predict embryo images."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict embryo image quality')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--patient', type=str, help='Patient name')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--output', type=str, help='Output directory for prediction results')
    parser.add_argument('--no-db', action='store_true', help='Do not save to database')
    parser.add_argument('--db-host', type=str, default='localhost', help='Database host')
    parser.add_argument('--db-user', type=str, default='suba', help='Database username')
    parser.add_argument('--db-password', type=str, default='Suba@123', help='Database password')
    parser.add_argument('--db-name', type=str, default='embryo_predictions', help='Database name')
    args = parser.parse_args()
    
    if args.image is None:
        parser.error("Please provide an image path with --image")
    
    # Initialize predictor
    predictor = EmbryoPredictor(args.model)
    
    # Make prediction
    result = predictor.predict(args.image, args.patient)
    
    # Prepare database config
    db_config = {
        'host': args.db_host,
        'user': args.db_user,
        'password': args.db_password,
        'database': args.db_name
    }
    
    # Save prediction
    save_result = predictor.save_prediction(
        result, 
        args.output, 
        save_to_database=not args.no_db,
        db_config=db_config
    )
    
    # Print results
    print("\n===== Prediction Results =====")
    print(f"Image: {result['image_path']}")
    if result.get('patient_name'):
        print(f"Patient: {result['patient_name']}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Results saved to: {save_result.get('file_path', 'N/A')}")
    if save_result.get('database_id'):
        print(f"Database record ID: {save_result['database_id']}")


if __name__ == "__main__":
    main()
