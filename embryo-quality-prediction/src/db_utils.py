import mysql.connector
from mysql.connector import Error
import os
import sys
import json
from datetime import datetime

# Get the absolute path to the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

class DatabaseConnection:
    def __init__(self, host="localhost", user="suba", password="Suba@123", database="embryo_predictions"):
        """
        Initialize database connection.
        
        Args:
            host (str): Database host address
            user (str): Database username
            password (str): Database password
            database (str): Database name
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        
    def connect(self):
        """Establish connection to the MySQL database."""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password
            )
            
            if self.connection.is_connected():
                cursor = self.connection.cursor()
                
                # Create database if it doesn't exist
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
                
                # Use the specified database
                cursor.execute(f"USE {self.database}")
                
                # Create tables if they don't exist
                self._create_tables(cursor)
                
                return True
                
        except Error as e:
            print(f"Error connecting to MySQL Database: {e}")
            return False
            
    def _create_tables(self, cursor):
        """Create required tables if they don't exist."""
        try:
            # Create predictions table
            predictions_table_query = """
            CREATE TABLE IF NOT EXISTS predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image_path VARCHAR(255) NOT NULL,
                patient_name VARCHAR(100) DEFAULT NULL,
                predicted_class_index INT NOT NULL,
                predicted_class VARCHAR(100) NOT NULL,
                confidence FLOAT NOT NULL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_predicted_class (predicted_class),
                INDEX idx_image_path (image_path),
                INDEX idx_patient_name (patient_name),
                INDEX idx_timestamp (timestamp)
            ) ENGINE=InnoDB
            """
            cursor.execute(predictions_table_query)
            
            # Create probabilities table
            probabilities_table_query = """
            CREATE TABLE IF NOT EXISTS class_probabilities (
                id INT AUTO_INCREMENT PRIMARY KEY,
                prediction_id INT NOT NULL,
                class_name VARCHAR(100) NOT NULL,
                probability FLOAT NOT NULL,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE,
                INDEX idx_prediction_id (prediction_id)
            ) ENGINE=InnoDB
            """
            cursor.execute(probabilities_table_query)
            
            # Create the stored procedure
            self._create_stored_procedures(cursor)
            
        except Error as e:
            print(f"Error creating tables: {e}")
            
    def _create_stored_procedures(self, cursor):
        """Create stored procedures for database operations."""
        try:
            # Drop the procedure if it exists
            cursor.execute("DROP PROCEDURE IF EXISTS SavePrediction")
            
            # Create the stored procedure for saving prediction results
            save_prediction_proc = """
            CREATE PROCEDURE SavePrediction(
                IN p_image_path VARCHAR(255),
                IN p_patient_name VARCHAR(100),
                IN p_predicted_class_index INT,
                IN p_predicted_class VARCHAR(100),
                IN p_confidence FLOAT,
                IN p_timestamp DATETIME,
                IN p_class_names JSON,
                IN p_probabilities JSON
            )
            BEGIN
                DECLARE v_prediction_id INT;
                DECLARE v_index INT DEFAULT 0;
                DECLARE v_class_name VARCHAR(100);
                DECLARE v_probability FLOAT;
                DECLARE v_max_index INT;
                DECLARE v_patient_name VARCHAR(100);
                
                -- Debug output to see received values
                SELECT 'Received patient_name:', p_patient_name;
                
                -- Handle patient_name properly
                SET v_patient_name = CASE 
                    WHEN p_patient_name = '' THEN NULL
                    ELSE p_patient_name
                END;
                
                -- Start transaction
                START TRANSACTION;
                
                -- Insert into predictions table
                INSERT INTO predictions (
                    image_path, 
                    patient_name,
                    predicted_class_index, 
                    predicted_class, 
                    confidence, 
                    timestamp
                ) VALUES (
                    p_image_path,
                    v_patient_name,  -- Use our processed patient name
                    p_predicted_class_index,
                    p_predicted_class,
                    p_confidence,
                    p_timestamp
                );
                
                -- Get the inserted ID
                SET v_prediction_id = LAST_INSERT_ID();
                
                -- Get the length of the probabilities array
                SET v_max_index = JSON_LENGTH(p_probabilities) - 1;
                
                -- Insert class probabilities
                WHILE v_index <= v_max_index DO
                    SET v_class_name = JSON_UNQUOTE(JSON_EXTRACT(p_class_names, CONCAT('$[', v_index, ']')));
                    SET v_probability = JSON_EXTRACT(p_probabilities, CONCAT('$[', v_index, ']'));
                    
                    INSERT INTO class_probabilities (
                        prediction_id,
                        class_name,
                        probability
                    ) VALUES (
                        v_prediction_id,
                        v_class_name,
                        v_probability
                    );
                    
                    SET v_index = v_index + 1;
                END WHILE;
                
                -- Commit the transaction
                COMMIT;
                
                -- Return the prediction ID
                SELECT v_prediction_id AS prediction_id;
            END
            """
            cursor.execute(save_prediction_proc)
            
        except Error as e:
            print(f"Error creating stored procedures: {e}")
    
    def save_prediction_to_db(self, prediction):
        """
        Save prediction results to database using stored procedure.
        
        Args:
            prediction (dict): Prediction result dictionary
            
        Returns:
            int: ID of the saved prediction record, or None if failed
        """
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()
                
            cursor = self.connection.cursor(dictionary=True)
            
            # Convert timestamp string to datetime if needed
            timestamp = prediction.get('timestamp')
            if isinstance(timestamp, str):
                timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            else:
                timestamp = datetime.now()
                
            # Handle patient_name: ensure it's None if empty string
            patient_name = prediction.get('patient_name')
            print(f"Initial patient_name in save_prediction_to_db: {patient_name!r} (type: {type(patient_name).__name__ if patient_name is not None else 'None'})")
            
            if patient_name is None:
                print("Patient name is already None")
            elif patient_name == '':
                print("Empty string patient_name detected, setting to None")
                patient_name = None
            elif isinstance(patient_name, str) and patient_name.strip() == '':
                print("Whitespace-only patient_name detected, setting to None")
                patient_name = None
            elif isinstance(patient_name, str):
                # Ensure the patient name is properly trimmed
                patient_name = patient_name.strip()
                print(f"Using trimmed patient name: '{patient_name}'")
            else:
                # Try to convert to string if it's some other type
                try:
                    converted = str(patient_name).strip()
                    if not converted:
                        print("Converted patient_name is empty, setting to None")
                        patient_name = None
                    else:
                        patient_name = converted
                        print(f"Using converted patient name: '{patient_name}'")
                except:
                    print("Error converting patient_name to string, setting to None")
                    patient_name = None
            
            # Extra safety check - force to None if empty string
            if patient_name == '':
                print("Empty string still detected, forcing to None")
                patient_name = None
                
            print(f"Final patient_name in save_prediction_to_db: {patient_name!r}")
            
            print(f"\n==== DATABASE SAVE OPERATION ====")
            print(f"Patient name from prediction: {patient_name} (type: {type(patient_name).__name__})")
            print(f"Image path: {prediction['image_path']} (type: {type(prediction['image_path']).__name__})")
            print(f"Class index: {prediction['predicted_class_index']} (type: {type(prediction['predicted_class_index']).__name__})")
            print(f"Class name: {prediction['predicted_class']} (type: {type(prediction['predicted_class']).__name__})")
            print(f"Confidence: {prediction['confidence']} (type: {type(prediction['confidence']).__name__})")
            print(f"Timestamp: {timestamp} (type: {type(timestamp).__name__})")
            
            args = [
                prediction['image_path'],
                patient_name,  # Use None if patient_name is not provided
                int(prediction['predicted_class_index']),  # Ensure this is an integer
                prediction['predicted_class'],
                float(prediction['confidence']),  # Ensure this is a float
                timestamp,
                json.dumps(prediction['class_names']),
                json.dumps(prediction['probabilities'])
            ]
            
            print(f"Arguments prepared for database procedure:")
            print(f"1. Image path: {args[0]!r}")
            print(f"2. Patient name: {args[1]!r} (type: {type(args[1]).__name__ if args[1] is not None else 'None'})")
            print(f"3. Class index: {args[2]!r}")
            print(f"4. Class name: {args[3]!r}")
            print(f"5. Confidence: {args[4]!r}")
            print(f"6. Timestamp: {args[5]!r}")
            print(f"7. Class names: (JSON)")
            print(f"8. Probabilities: (JSON)")
            
            try:
                cursor.callproc('SavePrediction', args)
                
                # Get the result (prediction ID)
                for result in cursor.stored_results():
                    result_row = result.fetchone()
                    if result_row and 'prediction_id' in result_row:
                        print(f"Successfully saved prediction with ID: {result_row['prediction_id']}")
                        print(f"==== END DATABASE SAVE OPERATION ====")
                        return result_row['prediction_id']
            except Error as e:
                print(f"Error calling SavePrediction procedure: {e}")
                # Print the arguments for debugging
                print(f"Arguments passed to procedure: {args}")
                print(f"==== END DATABASE SAVE OPERATION ====")
                raise e
            
            self.connection.commit()
            print(f"==== END DATABASE SAVE OPERATION ====")
            return None
            
        except Error as e:
            print(f"Error saving prediction to database: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def close(self):
        """Close the database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()

def save_prediction(prediction, db_config=None):
    """
    Save prediction to MySQL database.
    
    Args:
        prediction (dict): Prediction result with optional patient_name field
        db_config (dict, optional): Database configuration. 
                                   Defaults to localhost, user=suba, password=Suba@123.
    
    Returns:
        int: ID of the saved prediction or None if failed
    """
    # Debug info about the prediction being passed
    patient_name = prediction.get('patient_name')
    print(f"\n=== SAVE_PREDICTION DEBUG ===")
    print(f"Initial patient_name: {patient_name!r} (type: {type(patient_name).__name__})")
    
    # Make sure patient_name is None if it's an empty string or only whitespace
    if patient_name is None:
        print("Patient name is already None")
    elif patient_name == '':
        print("Empty string patient_name detected, setting to None")
        prediction['patient_name'] = None
    elif isinstance(patient_name, str) and patient_name.strip() == '':
        print("Whitespace-only patient_name detected, setting to None")
        prediction['patient_name'] = None
    
    print(f"Final patient_name: {prediction.get('patient_name')!r}")
    print(f"=== END SAVE_PREDICTION DEBUG ===\n")
    
    if db_config is None:
        db_config = {
            'host': 'localhost',
            'user': 'suba',
            'password': 'Suba@123',
            'database': 'embryo_predictions'
        }
    
    db = DatabaseConnection(**db_config)
    if db.connect():
        prediction_id = db.save_prediction_to_db(prediction)
        db.close()
        return prediction_id
    
    return None 