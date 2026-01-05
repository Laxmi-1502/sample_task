import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time


def generate_data(rows=300):
    np.random.seed(0)

    base_salary = np.random.normal(50000, 6000, rows)
    overtime_hours = np.random.normal(8, 3, rows)
    overtime_pay = overtime_hours * 250
    dept_avg_salary = base_salary + np.random.normal(0, 3000, rows)
    total_salary = base_salary + overtime_pay


    base_salary[:5] *= 2        
    overtime_hours[5:10] *= 6         
    overtime_pay[5:10] = overtime_hours[5:10] * 250

    return pd.DataFrame({
        "base_salary": base_salary,
        "overtime_hours": overtime_hours,
        "overtime_pay": overtime_pay,
        "total_salary": total_salary,
        "dept_avg_salary": dept_avg_salary
    })


data = generate_data()

scaler = StandardScaler()
X = scaler.fit_transform(data)


model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)
model.fit(X)

data["anomaly_score"] = model.decision_function(X)
data["anomaly_flag"] = model.predict(X)
data["anomaly_flag"] = data["anomaly_flag"].map({1: "Normal", -1: "Anomaly"})

print("\n🔹 Batch Detection Output:")
print(data.head(10))

print("\n Detected Anomalies:")
print(data[data["anomaly_flag"] == "Anomaly"].head())

def realtime_detection(employee_record):
    df = pd.DataFrame([employee_record])
    scaled = scaler.transform(df)
    score = model.decision_function(scaled)[0]
    label = model.predict(scaled)[0]

    return {
        "timestamp": datetime.now(),
        "anomaly_score": round(score, 4),
        "status": "Anomaly" if label == -1 else "Normal"
    }



new_record = {
    "base_salary": 120000,
    "overtime_hours": 90,
    "overtime_pay": 22500,
    "total_salary": 142500,
    "dept_avg_salary": 52000
}

print("\n Real-Time Detection Output:")
print(realtime_detection(new_record))

def retrain_on_recent_data(full_data, window_size=150):
    recent = full_data.tail(window_size)
 
    features = ["base_salary", "overtime_hours", "overtime_pay", "total_salary", "dept_avg_salary"]
    X_recent = scaler.fit_transform(recent[features])
    model.fit(X_recent)
    print("\n♻ Model retrained due to concept drift")



retrain_on_recent_data(data)

def simulate_real_time_stream():

    np.random.seed(None)
    print(f"\n Starting Real-Time Data Stream Simulation (Press Ctrl+C to stop)...")
    print("-" * 70)
    
    try:
        while True:
            
            if np.random.random() < 0.2:
            
                base = np.random.normal(100000, 10000)
                hours = np.random.normal(60, 10)
            else:
            
                base = np.random.normal(50000, 6000)
                hours = np.random.normal(8, 3)
                
            record = {
                "base_salary": base,
                "overtime_hours": hours,
                "overtime_pay": hours * 250,
                "total_salary": base + (hours * 250),
                "dept_avg_salary": 50000 + np.random.normal(0, 3000)
            }
            
        
            result = realtime_detection(record)
            
        
            icon = "🚨" if result["status"] == "Anomaly" else "✅"
            print(f"{icon} [{result['timestamp'].strftime('%H:%M:%S')}] "
                f"Score: {result['anomaly_score']:>7.4f} | "
                f"Status: {result['status']:<7} | "
                f"Salary: ${record['total_salary']:,.0f}")
            
            time.sleep(0.5) 
    except KeyboardInterrupt:
        print("\n🛑 Stream stopped by user.")

simulate_real_time_stream()
