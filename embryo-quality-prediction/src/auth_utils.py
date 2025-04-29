import os
import sys
import mysql.connector
from mysql.connector import Error
import bcrypt
from datetime import datetime

# Get the absolute path to the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

class AuthManager:
    def __init__(self, host="localhost", user="suba", password="Suba@123", database="embryo_predictions"):
        """
        Initialize authentication manager with database connection.
        
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
                
                # Create users table if it doesn't exist
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_login DATETIME NULL,
                    INDEX idx_email (email)
                ) ENGINE=InnoDB;
                """)
                
                return True
                
        except Error as e:
            print(f"Error connecting to MySQL Database: {e}")
            return False
            
    def register_user(self, email, password):
        """
        Register a new user with email and hashed password.
        
        Args:
            email (str): User's email address
            password (str): Plain text password to be hashed
            
        Returns:
            dict: User information if registration successful, None if failed
        """
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()
                
            cursor = self.connection.cursor(dictionary=True)
            
            # First check if email already exists
            try:
                cursor.callproc('CheckEmailExists', [email])
                for result in cursor.stored_results():
                    email_check = result.fetchone()
                    if email_check and email_check['email_count'] > 0:
                        return {"error": "User with this email already exists"}
            except Error as e:
                print(f"Error calling CheckEmailExists procedure: {e}")
                return {"error": str(e)}
                
            # Hash the password with bcrypt
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            hashed_password_str = hashed_password.decode('utf-8')
            
            # Call the stored procedure for user registration
            try:
                cursor.callproc('CreateUser', [email, hashed_password_str])
                
                # Get the result (user information)
                user_id = None
                for result in cursor.stored_results():
                    id_result = result.fetchone()
                    if id_result and 'user_id' in id_result:
                        user_id = id_result['user_id']
                
                if user_id:
                    self.connection.commit()
                    return {"id": user_id, "email": email}
                else:
                    return {"error": "Registration failed"}
                    
            except Error as e:
                print(f"Error calling CreateUser procedure: {e}")
                return {"error": str(e)}
            
            self.connection.commit()
            return None
            
        except Error as e:
            print(f"Error in register_user: {e}")
            return {"error": str(e)}
    
    def authenticate_user(self, email, password):
        """
        Authenticate a user with email and password.
        
        Args:
            email (str): User's email address
            password (str): Plain text password to verify
            
        Returns:
            dict: User information if authentication successful, None if failed
        """
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()
                
            cursor = self.connection.cursor(dictionary=True)
            
            # First get the stored hash for this user
            query = "SELECT id, password_hash FROM users WHERE email = %s LIMIT 1"
            cursor.execute(query, [email])
            user_data = cursor.fetchone()
            
            if not user_data:
                return {"error": "User not found"}
            
            # Verify the password using bcrypt
            stored_password = user_data['password_hash'].encode('utf-8')
            if bcrypt.checkpw(password.encode('utf-8'), stored_password):
                # Now call the authentication procedure with the stored hash
                try:
                    cursor.callproc('AuthenticateUser', [email, user_data['password_hash']])
                    
                    # Get the result (user information)
                    for result in cursor.stored_results():
                        user = result.fetchone()
                        if user and user['id'] is not None:
                            return user
                        else:
                            return {"error": "Authentication failed"}
                except Error as e:
                    print(f"Error calling AuthenticateUser procedure: {e}")
                    return {"error": str(e)}
            else:
                return {"error": "Invalid password"}
            
        except Error as e:
            print(f"Error in authenticate_user: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close the database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()

# Function to set up authentication tables
def setup_auth_database(host="localhost", user="suba", password="Suba@123", database="embryo_predictions"):
    """Set up authentication tables in the database."""
    auth_manager = AuthManager(host, user, password, database)
    return auth_manager.connect() 