import argparse
import os
import sys
import mysql.connector
from mysql.connector import Error

# Get the absolute path to the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

def fix_database(host, user, password, database):
    """Fix the database schema and stored procedures to correctly handle patient name and predicted_class_index."""
    print("Fixing embryo prediction database...")
    
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
            cursor = connection.cursor()
            
            # Make sure patient_name column exists
            print("Ensuring patient_name column exists in predictions table...")
            try:
                cursor.execute("""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s
                AND TABLE_NAME = 'predictions' 
                AND COLUMN_NAME = 'patient_name'
                """, (database,))
                
                if not cursor.fetchone():
                    print("Adding patient_name column to predictions table...")
                    cursor.execute("""
                    ALTER TABLE predictions 
                    ADD COLUMN patient_name VARCHAR(255) NULL AFTER image_path,
                    ADD INDEX idx_patient_name (patient_name)
                    """)
                    print("Patient name column added successfully.")
                else:
                    print("Patient name column already exists.")
            except Error as e:
                print(f"Error checking/adding patient_name column: {e}")
                
            # Drop and recreate the SavePrediction stored procedure
            print("Recreating SavePrediction stored procedure...")
            cursor.execute("DROP PROCEDURE IF EXISTS SavePrediction")
            
            save_prediction_proc = """
            CREATE PROCEDURE SavePrediction(
                IN p_image_path VARCHAR(255),
                IN p_patient_name VARCHAR(255),
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
            print("SavePrediction procedure recreated successfully.")
            
            # Create GetPredictionsByPatient stored procedure if it doesn't exist
            print("Creating GetPredictionsByPatient stored procedure...")
            cursor.execute("DROP PROCEDURE IF EXISTS GetPredictionsByPatient")
            
            get_by_patient_proc = """
            CREATE PROCEDURE GetPredictionsByPatient(
                IN p_patient_name VARCHAR(255)
            )
            BEGIN
                SELECT 
                    p.id, 
                    p.image_path,
                    p.patient_name,
                    p.predicted_class, 
                    p.confidence, 
                    p.timestamp, 
                    p.created_at
                FROM 
                    predictions p
                WHERE 
                    p.patient_name = p_patient_name
                ORDER BY 
                    p.timestamp DESC;
            END
            """
            cursor.execute(get_by_patient_proc)
            print("GetPredictionsByPatient procedure created successfully.")
            
            print("Database fix completed successfully.")
            
    except Error as e:
        print(f"Error: {e}")
    
    finally:
        # Close the connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("Connection closed.")

def main():
    """Main function to fix the database."""
    parser = argparse.ArgumentParser(description='Fix the embryo prediction database')
    parser.add_argument('--host', type=str, default='localhost', help='Database host')
    parser.add_argument('--user', type=str, required=True, help='Database username')
    parser.add_argument('--password', type=str, required=True, help='Database password')
    parser.add_argument('--database', type=str, default='embryo_predictions', help='Database name')
    
    args = parser.parse_args()
    
    fix_database(args.host, args.user, args.password, args.database)
    print("Database fix process completed.")

if __name__ == "__main__":
    main()
