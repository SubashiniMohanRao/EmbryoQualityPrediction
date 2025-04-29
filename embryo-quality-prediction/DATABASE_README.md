# MySQL Database Integration for Embryo Quality Prediction

This document explains how to set up and use the MySQL database integration for saving embryo quality prediction results.

## Prerequisites

- MySQL Server 5.7+ installed and running
- Python 3.7+ with mysql-connector-python package installed

## Database Setup

1. Ensure you have MySQL server installed and running
2. You need a MySQL user with appropriate permissions (creation of databases, tables, and stored procedures)
3. Run the database setup script using the setup-db command:

```bash
cd embryo-quality-prediction
python run_workflow.py setup-db --host localhost --user suba --password Suba@123 --database embryo_predictions
```

Alternatively, you can use the dedicated setup script:

```bash
cd embryo-quality-prediction
python src/setup_database.py --host localhost --user suba --password Suba@123 --database embryo_predictions
```

This script will:
- Create the `embryo_predictions` database if it doesn't exist
- Create the required tables (`predictions` and `class_probabilities`)
- Set up the stored procedures for saving and retrieving prediction results

## Database Schema

The database consists of two main tables:

### predictions
- `id`: Auto-increment primary key
- `image_path`: Path to the original image file
- `patient_name`: Name of the patient (optional)
- `predicted_class_index`: Numeric index of the predicted class
- `predicted_class`: Name of the predicted class
- `confidence`: Confidence score of the prediction (0-1)
- `timestamp`: When the prediction was made
- `created_at`: When the record was inserted into the database

### class_probabilities
- `id`: Auto-increment primary key
- `prediction_id`: Foreign key to the predictions table
- `class_name`: Name of the class
- `probability`: Probability score for this class (0-1)

## Stored Procedures

The following stored procedures are available:

1. `SavePrediction`: Saves a prediction result and its class probabilities
2. `GetPredictionsByDateRange`: Retrieves predictions within a date range
3. `GetPredictionWithProbabilities`: Retrieves a single prediction with all its class probabilities
4. `GetPredictionsByClass`: Retrieves all predictions for a specific class
5. `GetPredictionsByPatient`: Retrieves all predictions for a specific patient

## Usage

### Saving Predictions to Database

You can run predictions and save results to the database using the command-line interface:

```bash
# Predict and save to database (default)
python run_workflow.py predict --image path/to/image.jpg

# Include patient name with prediction
python run_workflow.py predict --image path/to/image.jpg --patient "John Doe"

# Specify a specific model to use
python run_workflow.py predict --image path/to/image.jpg --model path/to/model.pth --patient "John Doe"

# Save prediction results to a specific directory
python run_workflow.py predict --image path/to/image.jpg --output path/to/output/dir --patient "John Doe"
```

To disable database saving:

```bash
python run_workflow.py predict --image path/to/image.jpg --patient "John Doe" --no-db
```

### Database Configuration

You can configure the database connection using command-line arguments:

```bash
python run_workflow.py predict --image path/to/image.jpg --patient "John Doe" --db-host localhost --db-user suba --db-password Suba@123 --db-name embryo_predictions
```

### Programmatically Accessing the Database

```python
from src.db_utils import DatabaseConnection

# Connect to database
db = DatabaseConnection(
    host="localhost", 
    user="suba", 
    password="Suba@123", 
    database="embryo_predictions"
)
db.connect()

# Example: Call a stored procedure to get predictions by class
cursor = db.connection.cursor(dictionary=True)
cursor.callproc('GetPredictionsByClass', ["blastocyst_grade_A"])

# Process results
for result in cursor.stored_results():
    predictions = result.fetchall()
    for prediction in predictions:
        print(f"ID: {prediction['id']}, Patient: {prediction.get('patient_name', 'N/A')}, Confidence: {prediction['confidence']}")

# Example: Call a stored procedure to get predictions for a specific patient
cursor.callproc('GetPredictionsByPatient', ["John Doe"])

# Process results
for result in cursor.stored_results():
    predictions = result.fetchall()
    print(f"Found {len(predictions)} predictions for patient John Doe")
    for prediction in predictions:
        print(f"ID: {prediction['id']}, Class: {prediction['predicted_class']}, Confidence: {prediction['confidence']}")

# Close the connection
cursor.close()
db.close()
```

## Querying the Database

### Using Stored Procedures

You can use the provided stored procedures to query the database in Python:

```python
from src.db_utils import DatabaseConnection

# Connect to database
db = DatabaseConnection()
db.connect()

# Call the GetPredictionsByDateRange procedure
cursor = db.connection.cursor(dictionary=True)
cursor.callproc('GetPredictionsByDateRange', ["2023-01-01 00:00:00", "2023-12-31 23:59:59"])

# Process results
for result in cursor.stored_results():
    predictions = result.fetchall()
    print(f"Found {len(predictions)} predictions in date range")
    
# Get details of a specific prediction including class probabilities
cursor.callproc('GetPredictionWithProbabilities', [1])  # prediction_id = 1

# Process results - first result set is the prediction details
prediction_details = None
class_probabilities = []

for result in cursor.stored_results():
    if prediction_details is None:
        prediction_details = result.fetchone()
    else:
        class_probabilities = result.fetchall()

print(f"Prediction details: {prediction_details}")
print(f"Patient: {prediction_details.get('patient_name', 'N/A')}")
print(f"Class probabilities: {class_probabilities}")

# Call stored procedures
cursor.callproc('GetPredictionsByClass', ['blastocyst_grade_A'])
cursor.callproc('GetPredictionsByPatient', ['John Doe'])

# Process results
for result in cursor.stored_results():
    predictions = result.fetchall()
    print(f"Found {len(predictions)} predictions for patient John Doe")
    for prediction in predictions:
        print(f"ID: {prediction['id']}, Class: {prediction['predicted_class']}, Confidence: {prediction['confidence']}")

# Close connection
cursor.close()
db.close()
```

## Manual Database Operations

You can also connect to the database directly with MySQL client tools:

```bash
mysql -h localhost -u suba -p embryo_predictions
```

Example queries:

```sql
-- Get all predictions
SELECT * FROM predictions ORDER BY timestamp DESC;

-- Get predictions for a specific patient
SELECT * FROM predictions WHERE patient_name = 'John Doe' ORDER BY timestamp DESC;

-- Get probabilities for a specific prediction
SELECT * FROM class_probabilities WHERE prediction_id = 1;

-- Get high-confidence predictions
SELECT * FROM predictions WHERE confidence > 0.9;

-- Call a stored procedure
CALL GetPredictionsByClass('blastocyst_grade_A');
CALL GetPredictionsByPatient('John Doe');
``` 