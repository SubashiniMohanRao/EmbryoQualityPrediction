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
                patient_name VARCHAR(100) NULL,
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
                    p_patient_name,
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
                
            # Call the stored procedure
            patient_name = prediction.get('patient_name', None)
            print(f"\n==== DATABASE SAVE OPERATION ====")
            print(f"Patient name from prediction: {patient_name} (type: {type(patient_name)})")
            print(f"Image path: {prediction['image_path']}")
            print(f"Class index: {prediction['predicted_class_index']} (type: {type(prediction['predicted_class_index'])})")
            print(f"Class name: {prediction['predicted_class']}")
            print(f"Timestamp: {timestamp}")
            
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
            print(f"1. Image path: {args[0]}")
            print(f"2. Patient name: {args[1]}")
            print(f"3. Class index: {args[2]}")
            print(f"4. Class name: {args[3]}")
            print(f"5. Confidence: {args[4]}")
            print(f"6. Timestamp: {args[5]}")
            print(f"7. Class names: (JSON)")
            print(f"8. Probabilities: (JSON)")
            print(f"==== END DATABASE SAVE OPERATION ====")
            
            try:
                cursor.callproc('SavePrediction', args)
                
                # Get the result (prediction ID)
                for result in cursor.stored_results():
                    result_row = result.fetchone()
                    if result_row and 'prediction_id' in result_row:
                        print(f"Successfully saved prediction with ID: {result_row['prediction_id']}")
                        return result_row['prediction_id']
            except Error as e:
                print(f"Error calling SavePrediction procedure: {e}")
                # Print the arguments for debugging
                print(f"Arguments passed to procedure: {args}")
                raise e
            
            self.connection.commit()
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


def get_predictions_by_patient(patient_name=None, db_config=None):
    """Get predictions by patient name or all predictions if patient_name is None.
    
    Args:
        patient_name (str, optional): Patient name to filter by. If None, returns all predictions.
        db_config (dict): Database configuration.
        
    Returns:
        list: List of prediction dictionaries
    """
    if not db_config:
        db_config = {
            'host': 'localhost',
            'user': 'suba',
            'password': 'Suba@123',
            'database': 'embryo_predictions'
        }
    
    try:
        # Connect to the database
        db = DatabaseConnection(**db_config)
        if not db.connect():
            raise ValueError("Could not connect to database")
        
        cursor = db.connection.cursor(dictionary=True)
        
        # First get the predictions
        if patient_name:
            # Get predictions for a specific patient
            query = """
            SELECT id, image_path, patient_name, predicted_class_index, predicted_class, 
                   confidence, timestamp
            FROM predictions
            WHERE patient_name = %s
            ORDER BY timestamp DESC
            """
            cursor.execute(query, (patient_name,))
        else:
            # Get all predictions
            query = """
            SELECT id, image_path, patient_name, predicted_class_index, predicted_class, 
                   confidence, timestamp
            FROM predictions
            ORDER BY timestamp DESC
            """
            cursor.execute(query)
        
        predictions = cursor.fetchall()
        
        # Process the results
        results = []
        for pred in predictions:
            # Get class probabilities for this prediction
            prob_query = """
            SELECT class_name, probability
            FROM class_probabilities
            WHERE prediction_id = %s
            ORDER BY probability DESC
            """
            cursor.execute(prob_query, (pred['id'],))
            probabilities_data = cursor.fetchall()
            
            # Extract class names and probabilities
            class_names = [p['class_name'] for p in probabilities_data]
            probabilities = [p['probability'] for p in probabilities_data]
            
            # Create a prediction dictionary
            result = {
                'id': pred['id'],
                'image_path': pred['image_path'],
                'patient_name': pred['patient_name'],
                'predicted_class_index': pred['predicted_class_index'],
                'predicted_class': pred['predicted_class'],
                'confidence': pred['confidence'],
                'timestamp': pred['timestamp'],
                'class_names': class_names,
                'probabilities': probabilities
            }
            
            results.append(result)
        
        cursor.close()
        db.close()
        return results
    except Error as e:
        print(f"Error retrieving predictions: {e}")
        raise e


def get_unique_patients(db_config=None):
    """Get a list of unique patient names from the database.
    
    Args:
        db_config (dict): Database configuration.
        
    Returns:
        list: List of unique patient names
    """
    if not db_config:
        db_config = {
            'host': 'localhost',
            'user': 'suba',
            'password': 'Suba@123',
            'database': 'embryo_predictions'
        }
    
    try:
        # Connect to the database
        db = DatabaseConnection(**db_config)
        if not db.connect():
            raise ValueError("Could not connect to database")
        
        cursor = db.connection.cursor(dictionary=True)
        
        # Get unique patient names
        query = """
        SELECT DISTINCT patient_name 
        FROM predictions 
        WHERE patient_name IS NOT NULL 
        ORDER BY patient_name
        """
        cursor.execute(query)
        
        patients = cursor.fetchall()
        
        cursor.close()
        db.close()
        
        # Extract patient names from result
        return [p['patient_name'] for p in patients]
    except Error as e:
        print(f"Error retrieving unique patients: {e}")
        raise e