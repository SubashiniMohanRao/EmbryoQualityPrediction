import argparse
import os
import sys
import mysql.connector
from mysql.connector import Error
from datetime import datetime

# Get the absolute path to the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

# Import the prediction module
from src.predict_image import EmbryoPredictor
from src.db_utils import save_prediction

def test_database_save(host, user, password, database, image_path, patient_name):
    """Test saving a prediction to the database with a patient name."""
    print(f"Testing database save with patient name: {patient_name}")
    
    try:
        # Connect to MySQL server to verify the database structure
        print(f"Connecting to MySQL server at {host}...")
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        if connection.is_connected():
            print("Connected to MySQL server successfully.")
            
            # Create a cursor
            cursor = connection.cursor(dictionary=True)
            
            # Check if patient_name column exists
            print("Checking if patient_name column exists in predictions table...")
            cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = %s
            AND TABLE_NAME = 'predictions' 
            AND COLUMN_NAME = 'patient_name'
            """, (database,))
            
            column_info = cursor.fetchone()
            if column_info:
                print(f"Patient name column exists with type: {column_info['DATA_TYPE']}, max length: {column_info['CHARACTER_MAXIMUM_LENGTH']}")
            else:
                print("Patient name column does not exist in the predictions table!")
                return
            
            # Check the SavePrediction stored procedure
            print("Checking SavePrediction stored procedure parameters...")
            cursor.execute("""
            SELECT PARAMETER_NAME, PARAMETER_MODE, DTD_IDENTIFIER
            FROM INFORMATION_SCHEMA.PARAMETERS
            WHERE SPECIFIC_SCHEMA = %s
            AND SPECIFIC_NAME = 'SavePrediction'
            ORDER BY ORDINAL_POSITION
            """, (database,))
            
            params = cursor.fetchall()
            print("SavePrediction procedure parameters:")
            for param in params:
                print(f"  {param['PARAMETER_NAME']}: {param['PARAMETER_MODE']} {param['DTD_IDENTIFIER']}")
            
            # Close the database connection
            cursor.close()
            connection.close()
            
            # Initialize predictor
            print("Initializing predictor...")
            predictor = EmbryoPredictor()
            
            # Make prediction
            print(f"Making prediction for image: {image_path}")
            result = predictor.predict(image_path, patient_name)
            
            # Print prediction result
            print("\nPrediction result:")
            print(f"Image path: {result['image_path']}")
            print(f"Patient name: {result.get('patient_name', 'None')}")
            print(f"Predicted class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.2%}")
            
            # Save prediction to database
            print("\nSaving prediction to database...")
            db_config = {
                'host': host,
                'user': user,
                'password': password,
                'database': database
            }
            
            prediction_id = save_prediction(result, db_config)
            
            if prediction_id:
                print(f"Successfully saved prediction to database with ID: {prediction_id}")
                
                # Verify the saved data
                print("\nVerifying saved data...")
                connection = mysql.connector.connect(**db_config)
                cursor = connection.cursor(dictionary=True)
                
                cursor.execute("""
                SELECT id, image_path, patient_name, predicted_class, confidence, timestamp
                FROM predictions
                WHERE id = %s
                """, (prediction_id,))
                
                saved_data = cursor.fetchone()
                
                if saved_data:
                    print("Data saved in database:")
                    print(f"ID: {saved_data['id']}")
                    print(f"Image path: {saved_data['image_path']}")
                    print(f"Patient name: {saved_data['patient_name']}")
                    print(f"Predicted class: {saved_data['predicted_class']}")
                    print(f"Confidence: {saved_data['confidence']:.2%}")
                    print(f"Timestamp: {saved_data['timestamp']}")
                else:
                    print(f"Could not find saved prediction with ID: {prediction_id}")
                
                cursor.close()
                connection.close()
            else:
                print("Failed to save prediction to database!")
            
    except Error as e:
        print(f"Error: {e}")

def main():
    """Main function to test database save."""
    parser = argparse.ArgumentParser(description='Test saving a prediction to the database with a patient name')
    parser.add_argument('--host', type=str, default='localhost', help='Database host')
    parser.add_argument('--user', type=str, required=True, help='Database username')
    parser.add_argument('--password', type=str, required=True, help='Database password')
    parser.add_argument('--database', type=str, default='embryo_predictions', help='Database name')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--patient', type=str, required=True, help='Patient name')
    
    args = parser.parse_args()
    
    test_database_save(args.host, args.user, args.password, args.database, args.image, args.patient)
    print("Database test completed.")

if __name__ == "__main__":
    main()
