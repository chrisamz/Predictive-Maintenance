pip install pymongo

import pymongo
from datetime import datetime

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["predictive_maintenance"]

# Collections
components_col = db["components"]
sensor_readings_col = db["sensor_readings"]
maintenance_logs_col = db["maintenance_logs"]
rul_predictions_col = db["rul_predictions"]

# Example Data
component = {
    "_id": "comp_001",
    "name": "Compressor A",
    "type": "Compressor",
    "installation_date": "2021-01-01",
    "specifications": {
        "manufacturer": "ABC Corp",
        "model": "X123",
        "details": "Max pressure 150 psi"
    }
}

sensor_reading = {
    "_id": "read_001",
    "component_id": "comp_001",
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "readings": {
        "temperature": 75.5,
        "vibration": 0.02,
        "pressure": 101.3
    }
}

maintenance_log = {
    "_id": "log_001",
    "component_id": "comp_001",
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "action": "Routine Check",
    "details": "Checked all connections and cleaned filters",
    "cost": 200
}

rul_prediction = {
    "_id": "pred_001",
    "component_id": "comp_001",
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "predicted_rul": 150,
    "model_version": "v1.0"
}

# Insert Data
components_col.insert_one(component)
sensor_readings_col.insert_one(sensor_reading)
maintenance_logs_col.insert_one(maintenance_log)
rul_predictions_col.insert_one(rul_prediction)

def get_component(component_id):
    return components_col.find_one({"_id": component_id})

def get_sensor_readings(component_id, start_time, end_time):
    return sensor_readings_col.find({
        "component_id": component_id,
        "timestamp": {"$gte": start_time, "$lte": end_time}
    })

def get_maintenance_logs(component_id):
    return maintenance_logs_col.find({"component_id": component_id})

def get_rul_predictions(component_id):
    return rul_predictions_col.find({"component_id": component_id})

# Example Usage
component = get_component("comp_001")
sensor_readings = get_sensor_readings("comp_001", "2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z")
maintenance_logs = get_maintenance_logs("comp_001")
rul_predictions = get_rul_predictions("comp_001")

print(component)
print(list(sensor_readings))
print(list(maintenance_logs))
print(list(rul_predictions))

