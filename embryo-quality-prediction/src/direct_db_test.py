import mysql.connector
from mysql.connector import Error
import json
from datetime import datetime

def test_direct_db_save(host, user, password, database, image_path, patient_name):
    """Test saving a prediction directly to the database with a patient name."""
    print(f"Testing direct database save with patient name: {patient_name}")
    
    try:
        # Connect to MySQL server
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
            
            # Create a test prediction
            timestamp = datetime.now()
            
            # Sample data for testing
            class_names = ["Good", "Fair", "Poor", "Very Poor", "Degenerated"]
            probabilities = [0.1, 0.2, 0.3, 0.2, 0.2]
            
            # Print the parameters we're going to use
            print("\nParameters for stored procedure:")
            print(f"1. Image path: {image_path}")
            print(f"2. Patient name: {patient_name}")
            print(f"3. Predicted class index: {2}")
            print(f"4. Predicted class: {class_names[2]}")
            print(f"5. Confidence: {0.8}")
            print(f"6. Timestamp: {timestamp}")
            print(f"7. Class names: {json.dumps(class_names)}")
            print(f"8. Probabilities: {json.dumps(probabilities)}")
            
            # Call the stored procedure directly
            args = [
                image_path,
                patient_name,
                2,  # predicted_class_index
                class_names[2],  # predicted_class
                0.8,  # confidence
                timestamp,
                json.dumps(class_names),
                json.dumps(probabilities)
            ]
            
            try:
                cursor.callproc('SavePrediction', args)
                
                # Get the result (prediction ID)
                prediction_id = None
                for result in cursor.stored_results():
                    result_row = result.fetchone()
                    if result_row:
                        if isinstance(result_row, dict) and 'prediction_id' in result_row:
                            prediction_id = result_row['prediction_id']
                        elif isinstance(result_row, tuple) and len(result_row) > 0:
                            prediction_id = result_row[0]
                
                connection.commit()
                
                if prediction_id:
                    print(f"\nSuccessfully saved prediction with ID: {prediction_id}")
                    
                    # Verify the saved data
                    print("\nVerifying saved data...")
                    cursor.execute("""
                    SELECT id, image_path, patient_name, predicted_class_index, predicted_class, confidence, timestamp
                    FROM predictions
                    WHERE id = %s
                    """, (prediction_id,))
                    
                    saved_data = cursor.fetchone()
                    
                    if saved_data:
                        print("Data saved in database:")
                        print(f"ID: {saved_data['id']}")
                        print(f"Image path: {saved_data['image_path']}")
                        print(f"Patient name: {saved_data['patient_name']}")
                        print(f"Predicted class index: {saved_data['predicted_class_index']}")
                        print(f"Predicted class: {saved_data['predicted_class']}")
                        print(f"Confidence: {saved_data['confidence']:.2%}")
                        print(f"Timestamp: {saved_data['timestamp']}")
                    else:
                        print(f"Could not find saved prediction with ID: {prediction_id}")
                else:
                    print("Failed to get prediction ID from stored procedure.")
                    
                    # Check if the data was still saved
                    print("\nChecking if data was saved anyway...")
                    cursor.execute("""
                    SELECT id, image_path, patient_name, predicted_class_index, predicted_class, confidence, timestamp
                    FROM predictions
                    WHERE image_path = %s
                    ORDER BY id DESC
                    LIMIT 1
                    """, (image_path,))
                    
                    saved_data = cursor.fetchone()
                    
                    if saved_data:
                        print("Found most recent prediction with matching image path:")
                        print(f"ID: {saved_data['id']}")
                        print(f"Image path: {saved_data['image_path']}")
                        print(f"Patient name: {saved_data['patient_name']}")
                        print(f"Predicted class index: {saved_data['predicted_class_index']}")
                        print(f"Predicted class: {saved_data['predicted_class']}")
                        print(f"Confidence: {saved_data['confidence']:.2%}")
                        print(f"Timestamp: {saved_data['timestamp']}")
                    else:
                        print("No matching prediction found.")
            
            except Error as e:
                print(f"\nError calling SavePrediction procedure: {e}")
                
                # Check the structure of the stored procedure
                print("\nChecking SavePrediction procedure definition...")
                cursor.execute("""
                SHOW CREATE PROCEDURE SavePrediction
                """)
                
                proc_def = cursor.fetchone()
                if proc_def:
                    print("SavePrediction procedure definition:")
                    for key, value in proc_def.items():
                        print(f"{key}: {value}")
                
                # Check the predictions table structure
                print("\nChecking predictions table structure...")
                cursor.execute("""
                DESCRIBE predictions
                """)
                
                table_structure = cursor.fetchall()
                print("Predictions table structure:")
                for column in table_structure:
                    print(f"{column['Field']}: {column['Type']} {column['Null']} {column['Key']} {column['Default']} {column['Extra']}")
            
            # Close the database connection
            cursor.close()
            connection.close()
            print("Connection closed.")
        
    except Error as e:
        print(f"Error: {e}")

def main():
    """Main function to test database save."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test saving a prediction directly to the database with a patient name')
    parser.add_argument('--host', type=str, default='localhost', help='Database host')
    parser.add_argument('--user', type=str, required=True, help='Database username')
    parser.add_argument('--password', type=str, required=True, help='Database password')
    parser.add_argument('--database', type=str, default='embryo_predictions', help='Database name')
    parser.add_argument('--image', type=str, default='test_image.jpg', help='Image path (for testing)')
    parser.add_argument('--patient', type=str, required=True, help='Patient name')
    
    args = parser.parse_args()
    
    test_direct_db_save(args.host, args.user, args.password, args.database, args.image, args.patient)
    print("Direct database test completed.")

if __name__ == "__main__":
    main()
