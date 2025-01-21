# Predictive Analysis for Manufacturing Operations

This API allows you to predict machine downtime or production defects in a manufacturing environment. It uses machine learning to analyze manufacturing data and make predictions based on the input features.

## Table of Contents

1. [Requirements](#requirements)
2. [Setup Guide](#setup-guide)
3. [API Endpoints](#api-endpoints)
4. [Example Requests](#example-requests)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Testing](#testing)
7. [License](#license)

---

## Requirements

To run the API, ensure you have the following installed:

- Python 3.7+
- FastAPI
- Uvicorn (for running the server)
- Pandas
- Scikit-learn
- Joblib
- Pydantic
- Any other dependencies listed in `requirements.txt`

To install the dependencies, run the following command:
- `pip install -r requirements.txt`

---

## Setup Guide

1. **Clone the Repository**  
   Clone the repository to your local machine by running:
   - `git clone https://github.com/yourusername/predictive-analysis-manufacturing.git`
   - `cd predictive-analysis-manufacturing`

2. **Install Dependencies**  
   Install the required dependencies using pip:
   - `pip install -r requirements.txt`

3. **Start the FastAPI Server**  
   Run the FastAPI server using Uvicorn:
   - `uvicorn main:app --reload`  
   This will start the API server locally at `http://127.0.0.1:8000`.

---

## API Endpoints

The API has three main endpoints:

1. **/upload (POST)**  
   Uploads a CSV file containing manufacturing data.

   Request:
   - `curl -X POST "http://127.0.0.1:8000/upload" -F "file=@machine_downtime.csv"`

   Response:
   - `{"message": "File uploaded successfully", "columns": ["Date", "Machine_ID", "Assembly_Line_No", "Hydraulic_Pressure(bar)", "Coolant_Pressure(bar)", ...]}`

2. **/train (POST)**  
   Trains the machine learning model using the uploaded dataset.

   Request:
   - `curl -X POST "http://127.0.0.1:8000/train"`

   Response:
   - `{"message": "Model trained successfully", "accuracy": 0.95, "confusion_matrix": [[50, 5], [10, 35]]}`

3. **/predict (POST)**  
   Make a prediction based on input data.

   Request:
   - `curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"Machine_ID": "Makino-L1-Unit1-2013", "Date": "31-12-2021", "Hydraulic_Pressure(bar)": 125.33, "Coolant_Pressure(bar)": 4.93, "Air_System_Pressure(bar)": 6.19, "Coolant_Temperature": 35.3, "Hydraulic_Oil_Temperature(?C)": 47.4, "Spindle_Bearing_Temperature(?C)": 34.6, "Spindle_Vibration(?m)": 1.38, "Tool_Vibration(?m)": 25.27, "Spindle_Speed(RPM)": 19856, "Voltage(volts)": 368, "Torque(Nm)": 14.2, "Cutting(kN)": 2.68}'`

   Response:
   - `{"Downtime": "Machine_Failure", "Confidence": 0.85}`

---

## Example Requests

1. **Upload File**  
   To upload a file, use the following request:
   - `curl -X POST "http://127.0.0.1:8000/upload" -F "file=@machine_downtime.csv"`

2. **Train Model**  
   After uploading the file, you can train the model using:
   - `curl -X POST "http://127.0.0.1:8000/train"`

3. **Make a Prediction**  
   To make predictions, provide the required input data using:
   - `curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"Machine_ID": "Makino-L1-Unit1-2013", "Date": "31-12-2021", "Hydraulic_Pressure(bar)": 125.33, "Coolant_Pressure(bar)": 4.93, "Air_System_Pressure(bar)": 6.19, "Coolant_Temperature": 35.3, "Hydraulic_Oil_Temperature(?C)": 47.4, "Spindle_Bearing_Temperature(?C)": 34.6, "Spindle_Vibration(?m)": 1.38, "Tool_Vibration(?m)": 25.27, "Spindle_Speed(RPM)": 19856, "Voltage(volts)": 368, "Torque(Nm)": 14.2, "Cutting(kN)": 2.68}'`

---

## Evaluation Metrics

- **Accuracy**: Proportion of correctly predicted instances.
- **Confusion Matrix**: Shows the true vs predicted classifications.

---

## Testing

To test the API, use tools like Postman or cURL to interact with the endpoints. 
You can run the server locally as described and use the examples above to upload data, train the model, and make predictions.
