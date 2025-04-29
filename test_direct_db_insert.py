import mysql.connector
from mysql.connector import Error
import json
from datetime import datetime

def test_direct_insert():
    try:
        # Connect to the database
        connection = mysql.connector.connect(
            host="localhost",
            user="suba",
            password="Suba@123",
            database="embryo_predictions"
        )
        
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            
            # Test direct insert with patient name
            patient_name = "Test Patient Direct"
            image_path = "test/image/path.jpg"
            class_index = 1
            class_name = "Test Class"
            confidence = 0.95
            timestamp = datetime.now()
            class_names = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]
            probabilities = [0.01, 0.95, 0.01, 0.02, 0.01]
            
            print(f"Inserting with patient_name: {patient_name!r}")
            
            # Direct SQL insert
            insert_query = """
            INSERT INTO predictions (
                image_path, 
                patient_name,
                predicted_class_index, 
                predicted_class, 
                confidence, 
                timestamp
            ) VALUES (
                %s, %s, %s, %s, %s, %s
            )
            """
            
            cursor.execute(insert_query, (
                image_path,
                patient_name,  # This should not be NULL
                class_index,
                class_name,
                confidence,
                timestamp
            ))
            
            connection.commit()
            
            prediction_id = cursor.lastrowid
            print(f"Inserted prediction with ID: {prediction_id}")
            
            # Now check if the patient name was correctly stored
            cursor.execute("SELECT * FROM predictions WHERE id = %s", (prediction_id,))
            result = cursor.fetchone()
            
            if result:
                print(f"Retrieved record: {result}")
                print(f"Patient name in database: {result['patient_name']!r}")
                
                if result['patient_name'] != patient_name:
                    print("ERROR: Patient name in database does not match what we inserted!")
                else:
                    print("SUCCESS: Patient name correctly stored in database.")
            else:
                print("ERROR: Could not retrieve inserted record.")
            
    except Error as e:
        print(f"Error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

if __name__ == "__main__":
    test_direct_insert() 