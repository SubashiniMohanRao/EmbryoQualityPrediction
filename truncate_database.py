import mysql.connector
from mysql.connector import Error

def truncate_tables():
    """Safely truncate the predictions and class_probabilities tables."""
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="suba",
            password="Suba@123",
            database="embryo_predictions"
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            
            print("Connected to MySQL database")
            
            # Disable foreign key checks
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
            
            # Truncate tables
            cursor.execute("TRUNCATE TABLE class_probabilities")
            print("Truncated class_probabilities table")
            
            cursor.execute("TRUNCATE TABLE predictions")
            print("Truncated predictions table")
            
            # Re-enable foreign key checks
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
            
            connection.commit()
            print("Tables truncated successfully")
            
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

if __name__ == "__main__":
    print("Truncating prediction tables...")
    truncate_tables() 