import argparse
import os
import mysql.connector
from mysql.connector import Error

def setup_database(host, user, password, database):
    """Set up the embryo prediction database with required tables and stored procedures."""
    print("Setting up embryo prediction database...")
    
    try:
        # Connect to MySQL server
        print(f"Connecting to MySQL server at {host}...")
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
            
            # Use the database
            print(f"Using database '{database}'")
            cursor.execute(f"USE {database}")
            
            # Create tables
            print("Creating tables if they don't exist...")
            
            # Create predictions table
            predictions_table_query = """
            CREATE TABLE IF NOT EXISTS predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image_path VARCHAR(255) NOT NULL,
                predicted_class_index INT NOT NULL,
                predicted_class VARCHAR(100) NOT NULL,
                confidence FLOAT NOT NULL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_predicted_class (predicted_class),
                INDEX idx_image_path (image_path),
                INDEX idx_timestamp (timestamp)
            ) ENGINE=InnoDB
            """
            cursor.execute(predictions_table_query)
            print("Predictions table created successfully.")
            
            # Create class_probabilities table
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
            print("Class probabilities table created successfully.")
            
            # Create stored procedures
            print("Creating stored procedures...")
            
            # Drop existing procedures if they exist
            cursor.execute("DROP PROCEDURE IF EXISTS SavePrediction")
            cursor.execute("DROP PROCEDURE IF EXISTS GetPredictionsByDateRange")
            cursor.execute("DROP PROCEDURE IF EXISTS GetPredictionWithProbabilities")
            cursor.execute("DROP PROCEDURE IF EXISTS GetPredictionsByClass")
            
            # Create SavePrediction stored procedure
            save_prediction_proc = """
            CREATE PROCEDURE SavePrediction(
                IN p_image_path VARCHAR(255),
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
                    predicted_class_index, 
                    predicted_class, 
                    confidence, 
                    timestamp
                ) VALUES (
                    p_image_path,
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
            print("SavePrediction procedure created successfully.")
            
            # Create GetPredictionsByDateRange stored procedure
            get_by_date_proc = """
            CREATE PROCEDURE GetPredictionsByDateRange(
                IN p_start_date DATETIME,
                IN p_end_date DATETIME
            )
            BEGIN
                SELECT 
                    p.id, 
                    p.image_path, 
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
            print("GetPredictionsByDateRange procedure created successfully.")
            
            # Create GetPredictionWithProbabilities stored procedure
            get_with_probs_proc = """
            CREATE PROCEDURE GetPredictionWithProbabilities(
                IN p_prediction_id INT
            )
            BEGIN
                -- Get the prediction details
                SELECT 
                    p.id, 
                    p.image_path, 
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
            print("GetPredictionWithProbabilities procedure created successfully.")
            
            # Create GetPredictionsByClass stored procedure
            get_by_class_proc = """
            CREATE PROCEDURE GetPredictionsByClass(
                IN p_class_name VARCHAR(100)
            )
            BEGIN
                SELECT 
                    p.id, 
                    p.image_path, 
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
            print("GetPredictionsByClass procedure created successfully.")
            
            print("Database setup completed successfully.")
            
    except Error as e:
        print(f"Error: {e}")
    
    finally:
        # Close the connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("Connection closed.")

def main():
    """Main function to set up the database."""
    parser = argparse.ArgumentParser(description='Set up the embryo prediction database')
    parser.add_argument('--host', type=str, default='localhost', help='Database host')
    parser.add_argument('--user', type=str, required=True, help='Database username')
    parser.add_argument('--password', type=str, required=True, help='Database password')
    parser.add_argument('--database', type=str, default='embryo_predictions', help='Database name')
    
    args = parser.parse_args()
    
    setup_database(args.host, args.user, args.password, args.database)
    print("Database setup process completed.")

if __name__ == "__main__":
    main()
