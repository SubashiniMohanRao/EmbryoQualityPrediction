#!/usr/bin/env python
"""
Setup Authentication Procedures Script

This script sets up the necessary MySQL stored procedures for user authentication
in the Embryo Quality Prediction application.
"""

import os
import sys
import mysql.connector
from mysql.connector import Error

# Get the absolute path to the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

def setup_auth_procedures(host="localhost", user="suba", password="Suba@123", database="embryo_predictions"):
    """Set up authentication stored procedures in the database."""
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
            sql_script_path = os.path.join(SCRIPT_DIR, "src", "setup_auth_procedures.sql")
            
            if not os.path.exists(sql_script_path):
                print(f"SQL script file not found: {sql_script_path}")
                return False
            
            print(f"Reading SQL script from {sql_script_path}...")
            with open(sql_script_path, 'r') as f:
                sql_script = f.read()
            
            # Replace DELIMITER statements for proper execution
            sql_script = sql_script.replace('DELIMITER $$', '')
            sql_script = sql_script.replace('DELIMITER ;', '')
            sql_script = sql_script.replace('END$$', 'END;')
            
            # Split the script into individual statements
            statements = sql_script.split(';')
            
            # Execute each statement (skip empty ones)
            for statement in statements:
                statement = statement.strip()
                if statement:
                    try:
                        cursor.execute(statement)
                        print(f"Executed statement successfully.")
                    except Error as e:
                        print(f"Error executing statement: {e}")
                        print(f"Statement was: {statement[:100]}...")
            
            # Commit the changes
            connection.commit()
            print("Authentication procedures setup completed successfully.")
            
            # Close the cursor and connection
            cursor.close()
            connection.close()
            print("Connection closed.")
            
            return True
            
    except Error as e:
        print(f"Error connecting to MySQL Database: {e}")
        return False

def main():
    """Main function to setup authentication procedures."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup Authentication Procedures')
    parser.add_argument('--host', default='localhost', help='MySQL host')
    parser.add_argument('--user', default='suba', help='MySQL username')
    parser.add_argument('--password', default='Suba@123', help='MySQL password')
    parser.add_argument('--database', default='embryo_predictions', help='MySQL database name')
    
    args = parser.parse_args()
    
    print("Setting up authentication procedures...")
    success = setup_auth_procedures(
        host=args.host,
        user=args.user,
        password=args.password,
        database=args.database
    )
    
    if success:
        print("Authentication procedures setup completed successfully!")
        print("\nYou can now start the application with:")
        print(f"cd {SCRIPT_DIR}")
        print("python run_workflow.py web")
    else:
        print("Authentication procedures setup failed. Check logs for details.")
        sys.exit(1)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 