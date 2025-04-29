-- Create the database if it doesn't exist
CREATE DATABASE IF NOT EXISTS embryo_predictions;

-- Use the embryo_predictions database
USE embryo_predictions;

-- Create the predictions table
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
) ENGINE=InnoDB;

-- Create the class probabilities table
CREATE TABLE IF NOT EXISTS class_probabilities (
    id INT AUTO_INCREMENT PRIMARY KEY,
    prediction_id INT NOT NULL,
    class_name VARCHAR(100) NOT NULL,
    probability FLOAT NOT NULL,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE,
    INDEX idx_prediction_id (prediction_id)
) ENGINE=InnoDB;

-- Drop the stored procedure if it exists
DROP PROCEDURE IF EXISTS SavePrediction;

-- Create the stored procedure for saving prediction results
DELIMITER //
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
END //
DELIMITER ;

-- Add a stored procedure to retrieve predictions by date range
DELIMITER //
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
END //
DELIMITER ;

-- Add a stored procedure to retrieve a single prediction with all class probabilities
DELIMITER //
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
END //
DELIMITER ;

-- Add a stored procedure to get predictions by class
DELIMITER //
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
END //
DELIMITER ;

-- Add a stored procedure to get predictions by patient name
DELIMITER //
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
END //
DELIMITER ; 