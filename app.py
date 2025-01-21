from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from datetime import datetime

app = FastAPI()

# Global variable for the model and encoder
model = None
one_hot_encoder = None
data_path = "uploaded_data.csv"
model_path = "model.pkl"


@app.get("/")
def root():
    """
    Root endpoint to provide basic API information.
    """
    return {
        "message": "Welcome to the Predictive Analysis API! Use the following endpoints:",
        "endpoints": {
            "POST /upload": "Upload a CSV file containing manufacturing data.",
            "POST /train": "Train the machine learning model using the uploaded data.",
            "POST /predict": "Make a prediction using input parameters.",
            "Example": {
                "Upload File": 'curl -X POST "http://127.0.0.1:8000/upload" -F "file=@machine_downtime.csv"',
                "Train Model": 'curl -X POST "http://127.0.0.1:8000/train"',
                "Predict": 'curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"Machine_ID\":\"Makino-L1-Unit1-2013\",\"Date\":\"31-12-2021\",\"Hydraulic_Pressure(bar)\":125.33,\"Coolant_Pressure(bar)\":4.93,\"Air_System_Pressure(bar)\":6.19,\"Coolant_Temperature\":35.3,\"Hydraulic_Oil_Temperature(?C)\":47.4,\"Spindle_Bearing_Temperature(?C)\":34.6,\"Spindle_Vibration(?m)\":1.38,\"Tool_Vibration(?m)\":25.27,\"Spindle_Speed(RPM)\":19856,\"Voltage(volts)\":368,\"Torque(Nm)\":14.2,\"Cutting(kN)\":2.68}"'
            }
        }
    }



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload CSV file and save it locally.
    """
    try:
        # Read the uploaded file into a DataFrame
        data = pd.read_csv(file.file)
        # Save the DataFrame to a local file
        data.to_csv(data_path, index=False)
        return {"message": "File uploaded successfully", "columns": list(data.columns)}
    except Exception as e:
        return {"error": str(e)}


@app.post("/train")
async def train_model():
    """
    Train a Random Forest classifier on the uploaded data.
    """
    global model, one_hot_encoder
    try:
        # Load the uploaded data
        data = pd.read_csv(data_path)

        # Convert 'Date' to datetime
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

        # Add a feature for days since the earliest date in the dataset
        data['Days_Since_Start'] = (data['Date'] - data['Date'].min()).dt.days

        # Drop the original 'Date' column
        data = data.drop(columns=['Date'])

        # One-hot encode 'Machine_ID'
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoder.set_output(transform="pandas")
        machine_id_encoded = one_hot_encoder.fit_transform(data[['Machine_ID']])

        # Combine the dataset with encoded Machine_ID
        data = pd.concat([data.drop(columns=['Machine_ID', 'Assembly_Line_No']), machine_id_encoded], axis=1)

        # Encode 'Downtime' (target variable)
        data['Downtime'] = data['Downtime'].apply(lambda x: 1 if x == "Machine_Failure" else 0)

        # Prepare features and target
        X = data.drop(columns=['Downtime'])
        y = data['Downtime']

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Define and train the pipeline
        model = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(max_depth=6, random_state=0)
        )
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Save the trained model
        joblib.dump(model, model_path)

        return {
            "message": "Model trained successfully",
            "accuracy": accuracy,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
    except Exception as e:
        return {"error": str(e)}



@app.post("/predict")
async def predict(input_data: dict):
    """
    Make a prediction based on input data using the trained model.
    """
    global model, one_hot_encoder
    try:
        if model is None or one_hot_encoder is None:
            return {"error": "Model not trained yet. Train the model using the /train endpoint first."}

        # Extract Machine_ID and other features
        machine_id = input_data.get("Machine_ID")
        required_features = [
            "Hydraulic_Pressure(bar)", "Coolant_Pressure(bar)", "Air_System_Pressure(bar)",
            "Coolant_Temperature", "Hydraulic_Oil_Temperature(?C)", "Spindle_Bearing_Temperature(?C)",
            "Spindle_Vibration(?m)", "Tool_Vibration(?m)", "Spindle_Speed(RPM)", "Voltage(volts)",
            "Torque(Nm)", "Cutting(kN)"
        ]

        # Validate input
        if machine_id is None or not all(feature in input_data for feature in required_features):
            return {"error": f"Input must contain 'Machine_ID' and all required features: {required_features}"}

        # Calculate Days_Since_Start
        reference_date = datetime.strptime("01-01-2013", "%d-%m-%Y")
        input_date = datetime.strptime(input_data.get("Date", "31-12-2021"), "%d-%m-%Y")
        days_since_start = (input_date - reference_date).days

        # One-hot encode Machine_ID
        machine_id_encoded = one_hot_encoder.transform([[machine_id]])

        # Combine all features
        feature_values = [input_data[feature] for feature in required_features]
        input_array = np.concatenate([[days_since_start], feature_values, machine_id_encoded.values.flatten()])

        # Predict using the trained model
        prediction = model.predict([input_array])[0]
        confidence = max(model.predict_proba([input_array])[0])

        return {
            "Downtime": "Machine_Failure" if prediction == 1 else "No_Failure",
            "Confidence": round(confidence, 2)
        }
    except Exception as e:
        return {"error": str(e)}

