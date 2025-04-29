import argparse
import mysql.connector
from mysql.connector import Error

def update_database_schema(host, user, password, database):
    """Update the embryo prediction database schema to include patient name."""
    print("Updating embryo prediction database schema...")
    
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
            
            # Add patient_name column to predictions table if it doesn't exist
            print("Adding patient_name column to predictions table if it doesn't exist...")
            try:
                cursor.execute("""
                ALTER TABLE predictions 
                ADD COLUMN patient_name VARCHAR(255) NULL AFTER image_path,
                ADD INDEX idx_patient_name (patient_name)
                """)
                print("Patient name column added successfully.")
            except Error as e:
                if "Duplicate column name" in str(e):
                    print("Patient name column already exists.")
                else:
                    raise e
            
            # Update the SavePrediction stored procedure to include patient_name
            print("Updating SavePrediction stored procedure...")
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
            print("SavePrediction procedure updated successfully.")
            
            # Create a new stored procedure to get predictions by patient name
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
            
            print("Database schema update completed successfully.")
            
    except Error as e:
        print(f"Error: {e}")
    
    finally:
        # Close the connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("Connection closed.")

def main():
    """Main function to update the database schema."""
    parser = argparse.ArgumentParser(description='Update the embryo prediction database schema')
    parser.add_argument('--host', type=str, default='localhost', help='Database host')
    parser.add_argument('--user', type=str, required=True, help='Database username')
    parser.add_argument('--password', type=str, required=True, help='Database password')
    parser.add_argument('--database', type=str, default='embryo_predictions', help='Database name')
    
    args = parser.parse_args()
    
    update_database_schema(args.host, args.user, args.password, args.database)
    print("Database schema update process completed.")

if __name__ == "__main__":
    main()
