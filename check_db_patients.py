import mysql.connector
from mysql.connector import Error

def check_patient_names():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="suba",
            password="Suba@123",
            database="embryo_predictions"
        )
        
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            
            # Query the last 10 records
            cursor.execute("""
            SELECT id, image_path, patient_name, predicted_class, confidence, timestamp
            FROM predictions
            ORDER BY id DESC
            LIMIT 10
            """)
            
            records = cursor.fetchall()
            
            print("\n--- Latest 10 Records in Database ---")
            for record in records:
                patient_name = record['patient_name']
                print(f"ID: {record['id']}, Patient: {patient_name!r} (type: {type(patient_name).__name__ if patient_name is not None else 'None'}), Class: {record['predicted_class']}")
            
            # Count records by patient name status
            cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN patient_name IS NULL THEN 1 ELSE 0 END) as null_names,
                SUM(CASE WHEN patient_name IS NOT NULL THEN 1 ELSE 0 END) as non_null_names
            FROM predictions
            """)
            
            stats = cursor.fetchone()
            print("\n--- Patient Name Statistics ---")
            print(f"Total records: {stats['total']}")
            print(f"Records with NULL patient_name: {stats['null_names']} ({stats['null_names']/stats['total']*100:.1f}%)")
            print(f"Records with non-NULL patient_name: {stats['non_null_names']} ({stats['non_null_names']/stats['total']*100:.1f}%)")
            
    except Error as e:
        print(f"Error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("\nMySQL connection is closed")

if __name__ == "__main__":
    check_patient_names() 