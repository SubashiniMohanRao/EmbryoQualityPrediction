-- Setup the users table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME NULL,
    INDEX idx_email (email)
) ENGINE=InnoDB;

-- Check if email exists procedure
DELIMITER $$
DROP PROCEDURE IF EXISTS CheckEmailExists $$
CREATE PROCEDURE CheckEmailExists(
    IN p_email VARCHAR(255)
)
BEGIN
    SELECT COUNT(*) AS email_count
    FROM users
    WHERE email = p_email;
END$$

-- Create user procedure
DROP PROCEDURE IF EXISTS CreateUser $$
CREATE PROCEDURE CreateUser(
    IN p_email VARCHAR(255),
    IN p_password_hash VARCHAR(255)
)
BEGIN
    INSERT INTO users (email, password_hash)
    VALUES (p_email, p_password_hash);
    
    SELECT LAST_INSERT_ID() AS user_id;
END$$

-- Authenticate user procedure
DROP PROCEDURE IF EXISTS AuthenticateUser $$
CREATE PROCEDURE AuthenticateUser(
    IN p_email VARCHAR(255),
    IN p_password_hash VARCHAR(255)
)
BEGIN
    DECLARE v_user_id INT;
    DECLARE v_found BOOLEAN DEFAULT FALSE;
    
    SELECT id INTO v_user_id
    FROM users
    WHERE email = p_email AND password_hash = p_password_hash
    LIMIT 1;
    
    IF v_user_id IS NOT NULL THEN
        -- Update last login timestamp
        UPDATE users SET last_login = NOW() WHERE id = v_user_id;
        
        -- Return user details
        SELECT id, email, created_at, last_login
        FROM users
        WHERE id = v_user_id;
        
        SET v_found = TRUE;
    END IF;
    
    IF NOT v_found THEN
        -- Return empty result to indicate authentication failure
        SELECT NULL AS id, NULL AS email, NULL AS created_at, NULL AS last_login;
    END IF;
END$$

DELIMITER ; 