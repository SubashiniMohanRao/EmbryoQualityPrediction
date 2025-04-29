import os
import sys
import mysql.connector
from mysql.connector import Error

# Get the absolute path to the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

def setup_database(host="localhost", user="suba", password="Suba@123", database="embryo_predictions"):
    """Set up the MySQL database, tables, and stored procedures."""
    try:
        print(f"Connecting to MySQL server at {host}...")
        
        # Connect to MySQL server
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password
        )
        
        if connection.is_connected():
            print("Connected to MySQL server successfully.")
            
            # Create a cursor
            cursor = connection.cursor()
            
            # Create database if it doesn't exist
            print(f"Creating database '{database}' if it doesn't exist...")
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
            
            # Use the specified database
            cursor.execute(f"USE {database}")
            print(f"Using database '{database}'")
            
            # Read the SQL script file
            sql_script_path = os.path.join(SCRIPT_DIR, "db_setup.sql")
            
            if not os.path.exists(sql_script_path):
                print(f"SQL script file not found: {sql_script_path}")
                return False
            
            print(f"Reading SQL script from {sql_script_path}...")
            with open(sql_script_path, 'r') as f:
                sql_script = f.read()
            
            # Split the script into individual statements
            # This is a simple way to handle multiple statements
            # For more complex scripts, consider using a proper SQL parser
            statements = sql_script.split(';')
            
            # Execute each statement (skip empty ones)
            for statement in statements:
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    # Handle DELIMITER changes
                    if 'DELIMITER' in statement:
                        # For simplicity, we'll execute procedure creation directly
                        # In a real implementation, you might need a more robust parser
                        continue
                        
                    try:
                        # Replace any non-standard delimiters
                        statement = statement.replace('DELIMITER //', '')
                        statement = statement.replace('DELIMITER ;', '')
                        statement = statement.replace('//', ';')
                        
                        cursor.execute(statement)
                        print(f"Executed statement successfully.")
                    except Error as e:
                        print(f"Error executing statement: {e}")
                        print(f"Statement was: {statement[:100]}...")
            
            # Commit the changes
            connection.commit()
            print("Database setup completed successfully.")
            
            # Close the cursor and connection
            cursor.close()
            connection.close()
            print("Connection closed.")
            
            return True
            
    except Error as e:
        print(f"Error connecting to MySQL Database: {e}")
        return False

def setup_procedures(host="localhost", user="suba", password="Suba@123", database="embryo_predictions"):
    """Set up stored procedures in the database."""
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            
            # SavePrediction stored procedure
            print("Creating SavePrediction stored procedure...")
            cursor.execute("DROP PROCEDURE IF EXISTS SavePrediction")
            
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
            
            # GetPredictionsByDateRange stored procedure
            print("Creating GetPredictionsByDateRange stored procedure...")
            cursor.execute("DROP PROCEDURE IF EXISTS GetPredictionsByDateRange")
            
            get_by_date_proc = """
            CREATE PROCEDURE GetPredictionsByDateRange(
                IN p_start_date DATETIME,
                IN p_end_date DATETIME
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
                    p.timestamp BETWEEN p_start_date AND p_end_date
                ORDER BY 
                    p.timestamp DESC;
            END
            """
            cursor.execute(get_by_date_proc)
            
            # GetPredictionWithProbabilities stored procedure
            print("Creating GetPredictionWithProbabilities stored procedure...")
            cursor.execute("DROP PROCEDURE IF EXISTS GetPredictionWithProbabilities")
            
            get_with_probs_proc = """
            CREATE PROCEDURE GetPredictionWithProbabilities(
                IN p_prediction_id INT
            )
            BEGIN
                -- Get the prediction details
                SELECT 
                    p.id, 
                    p.image_path, 
                    p.patient_name,
                    p.predicted_class_index, 
                    p.predicted_class, 
                    p.confidence, 
                    p.timestamp, 
                    p.created_at
                FROM 
                    predictions p
                WHERE 
                    p.id = p_prediction_id;
                
                -- Get all class probabilities for this prediction
                SELECT 
                    cp.class_name, 
                    cp.probability
                FROM 
                    class_probabilities cp
                WHERE 
                    cp.prediction_id = p_prediction_id
                ORDER BY 
                    cp.probability DESC;
            END
            """
            cursor.execute(get_with_probs_proc)
            
            # GetPredictionsByClass stored procedure
            print("Creating GetPredictionsByClass stored procedure...")
            cursor.execute("DROP PROCEDURE IF EXISTS GetPredictionsByClass")
            
            get_by_class_proc = """
            CREATE PROCEDURE GetPredictionsByClass(
                IN p_class_name VARCHAR(100)
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
                    p.predicted_class = p_class_name
                ORDER BY 
                    p.confidence DESC, 
                    p.timestamp DESC;
            END
            """
            cursor.execute(get_by_class_proc)
            
            # GetPredictionsByPatient stored procedure
            print("Creating GetPredictionsByPatient stored procedure...")
            cursor.execute("DROP PROCEDURE IF EXISTS GetPredictionsByPatient")
            
            get_by_patient_proc = """
            CREATE PROCEDURE GetPredictionsByPatient(
                IN p_patient_name VARCHAR(100)
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
            
            connection.commit()
            print("All stored procedures created successfully.")
            
            cursor.close()
            connection.close()
            
            return True
            
    except Error as e:
        print(f"Error setting up stored procedures: {e}")
        return False

def main():
    """Main function to set up the database."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Set up the embryo prediction database')
    parser.add_argument('--host', type=str, default='localhost', help='Database host')
    parser.add_argument('--user', type=str, default='suba', help='Database username')
    parser.add_argument('--password', type=str, default='Suba@123', help='Database password')
    parser.add_argument('--database', type=str, default='embryo_predictions', help='Database name')
    args = parser.parse_args()
    
    print("Setting up embryo prediction database...")
    
    # Set up database schema
    schema_result = setup_database(args.host, args.user, args.password, args.database)
    
    if not schema_result:
        print("Failed to set up database schema.")
        
        # Try setting up just the stored procedures
        print("Attempting to set up stored procedures...")
        proc_result = setup_procedures(args.host, args.user, args.password, args.database)
        
        if proc_result:
            print("Stored procedures set up successfully.")
        else:
            print("Failed to set up stored procedures.")
    
    print("Database setup process completed.")

if __name__ == "__main__":
    main() 