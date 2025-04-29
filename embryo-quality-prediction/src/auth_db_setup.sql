-- Authentication System Setup

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME NULL,
    INDEX idx_email (email)
) ENGINE=InnoDB;

-- Create stored procedure for user registration
DROP PROCEDURE IF EXISTS RegisterUser;
CREATE PROCEDURE RegisterUser(
    IN p_email VARCHAR(255),
    IN p_password VARCHAR(255)
)
BEGIN
    -- Check if user already exists
    DECLARE user_exists INT DEFAULT 0;
    
    SELECT COUNT(*) INTO user_exists 
    FROM users 
    WHERE email = p_email;
    
    IF user_exists = 0 THEN
        -- Insert new user
        INSERT INTO users (email, password) 
        VALUES (p_email, p_password);
        
        -- Return success with user id
        SELECT 
            id, 
            email,
            created_at
        FROM users
        WHERE id = LAST_INSERT_ID();
    ELSE
        -- Return error message
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'User with this email already exists';
    END IF;
END;

-- Create stored procedure for user authentication
DROP PROCEDURE IF EXISTS AuthenticateUser;
CREATE PROCEDURE AuthenticateUser(
    IN p_email VARCHAR(255),
    IN p_password VARCHAR(255)
)
BEGIN
    DECLARE v_user_id INT;
    DECLARE v_found BOOLEAN DEFAULT FALSE;
    
    SELECT id INTO v_user_id
    FROM users
    WHERE email = p_email AND password = p_password
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
END; 