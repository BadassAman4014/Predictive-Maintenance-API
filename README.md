# Predictive Analysis for Manufacturing Operations

This API allows you to predict machine downtime or production defects in a manufacturing environment. It uses machine learning to analyze manufacturing data and make predictions based on the input features.

## Table of Contents

1. [Requirements](#requirements)
2. [Setup Guide](#setup-guide)
3. [API Endpoints](#api-endpoints)
4. [Example Requests](#example-requests)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Testing](#testing)

## Requirements

To run the API, ensure you have the following installed:

- Python 3.7+ (currently using 3.9.11)
- FastAPI
- Uvicorn (for running the server)
- Pandas
- Scikit-learn
- Joblib
- Pydantic
- Any other dependencies listed in `requirements.txt`

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Alternatively you can use this command

```bash
pip install fastapi uvicorn pandas scikit-learn joblib numpy
```

## Setup Guide

1. **Clone the Repository**

```bash
git clone https://github.com/BadassAman4014/Predictive-Maintenance-API.git
cd Predictive-Maintenance-API
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Start the FastAPI Server**

```bash
uvicorn main:app --reload
```

This will start the API server locally at `http://127.0.0.1:8000`.

## API Endpoints

The API has three main endpoints:

### 1. Upload Data (/upload)

Upload a CSV file containing manufacturing data.

```bash
curl -X POST "http://127.0.0.1:8000/upload" -F "file=@machine_downtime.csv"
```

Response:
```json
{
  "message":"File uploaded successfully","columns":["Date","Machine_ID","Assembly_Line_No","Hydraulic_Pressure(bar)","Coolant_Pressure(bar)","Air_System_Pressure(bar)","Coolant_Temperature","Hydraulic_Oil_Temperature(?C)","Spindle_Bearing_Temperature(?C)","Spindle_Vibration(?m)","Tool_Vibration(?m)","Spindle_Speed(RPM)","Voltage(volts)","Torque(Nm)","Cutting(kN)","Downtime"]
}
```

### 2. Train Model (/train) 
This endpoint trains the machine learning model using the uploaded dataset. Please note that the training process may take several minutes to complete due to the grid search cross-validation method being used to optimize model hyperparameters.

```bash
curl -X POST "http://127.0.0.1:8000/train"
```

Response:
```json
{
  "message":"Model trained successfully","accuracy":0.992,
  "confusion_matrix":[[245,2],[2,251]]
}
```

### 3. Make Prediction (/predict)

Make a prediction based on input data.

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"Machine_ID\":\"Makino-L1-Unit1-2013\",\"Date\":\"31-12-2021\",\"Hydraulic_Pressure(bar)\":125.33,\"Coolant_Pressure(bar)\":4.93,\"Air_System_Pressure(bar)\":6.19,\"Coolant_Temperature\":35.3,\"Hydraulic_Oil_Temperature(?C)\":47.4,\"Spindle_Bearing_Temperature(?C)\":34.6,\"Spindle_Vibration(?m)\":1.38,\"Tool_Vibration(?m)\":25.27,\"Spindle_Speed(RPM)\":19856,\"Voltage(volts)\":368,\"Torque(Nm)\":14.2,\"Cutting(kN)\":2.68}"
```

Response:
```json
{
  "Downtime":"Machine_Failure",
  "Confidence":0.53
}
```

## Evaluation Metrics

- **Accuracy**: Proportion of correctly predicted instances.
- **Confusion Matrix**: Shows the true vs predicted classifications.

## Testing

To test the API, use tools like Postman or cURL to interact with the endpoints. You can run the server locally as described in the setup section and use the examples above to:
- Upload data
- Train the model
- Make predictions