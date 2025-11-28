# documents-

# Smart Traffic Congestion & Prediction System

The Smart Traffic Congestion and Prediction System is an AI-powered solution that uses Machine Learning, Deep Learning, IoT, Cloud, and Big Data to monitor real-time traffic, predict congestion, and control traffic signals automatically.


## 🚦 Project Overview

1. Detects vehicles using YOLO.
2. Predicts upcoming traffic using ML/DL models.
3. Dynamically controls green/red signals.
4. Provides a live Streamlit dashboard.
5. Integrates APIs (like Google Maps) for external traffic data.

## 🧠 Key Features

1. Real-Time Vehicle Detection
- Uses YOLOv8 deep learning model.
- Detects and counts: cars, bikes, buses, trucks.
- Works with CCTV/IP cameras or video streams.

📈 2. Traffic Prediction

ML Models: Random Forest, XGBoost

DL Model: LSTM (Time-series forecasting)

Predicts congestion 5–15 minutes ahead.

🚦 3. Smart Adaptive Signal Control

Automatically adjusts:

Green light duration

Red light duration
Based on:

Traffic density

Peak hour patterns

Prediction values

📊 4. Streamlit Live Dashboard

Shows:

4 real-time camera feeds

Vehicle count

Predicted congestion level

Adaptive signal time

Traffic graphs

Map-based traffic visualization

☁️ 5. IoT + Cloud + Big Data Support

IoT Sensors

Google Maps real-time API

AWS/GCP cloud storage

Kafka/Spark for large-scale traffic data (optional)

## 📂 Project Structure

```
Smart-Traffic-Congestion-Prediction
│
├── data/
│   ├── raw_data.csv
│   ├── processed_data.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── model_training.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│
├── dashboard/
│   ├── app.py
│   ├── static/
│   ├── templates/
│
├── models/
│   ├── best_model.pkl
│
├── README.md
└── requirements.txt
```

---

## 🔧 Technologies Used

* **Python** (NumPy, Pandas, Sklearn, Matplotlib)
* **Machine Learning Models** (Random Forest, XGBoost, LSTM)
* **Flask / FastAPI** for API deployment
* **HTML, CSS, JS** for dashboard (optional)
* **Jupyter Notebook** for EDA

---

## 📊 Workflow / Methodology

### 1️⃣ Data Collection

Traffic datasets can include:

* Vehicle count
* Average speed
* Weather data
* Time & date
* Special events
* Road conditions

### 2️⃣ Data Preprocessing

* Remove noise, duplicates
* Handle missing values
* Normalize feature scales
* Train-test split

### 3️⃣ Feature Engineering

* Time-based features (hour, day, peak/off-peak)
* Road ID encoding
* Lag values for time-series modeling

### 4️⃣ Model Training

Test various ML algorithms and choose the best-performing model based on:

* RMSE
* MAE
* R² score

### 5️⃣ Prediction Engine

Model predicts congestion level:

* Low
* Medium
* High

### 6️⃣ Dashboard Visualization

Displays:

* Live congestion heatmaps
* Predicted vs actual traffic
* Time-series trend graphs

---

## ▶️ How to Run the Project

### Step 1: Install Dependencies

```
pip install -r requirements.txt
```

### Step 2: Run Model Training

```
python src/train.py
```

### Step 3: Start Prediction API

```
python src/predict.py
```

### Step 4: Run Dashboard (Optional)

```
python dashboard/app.py
```

---

## ⚙️ Example Prediction Code

```python
from src.model import load_model
from src.preprocess import preprocess_input

model = load_model('models/best_model.pkl')
input_data = preprocess_input({
    "vehicle_count": 120,
    "avg_speed": 35,
    "weather": "clear",
    "hour": 18,
    "day": "Monday"
})

prediction = model.predict([input_data])
print("Predicted Congestion Level:", prediction)
```

---

## 📈 Future Enhancements

* Integration with **IoT road sensors**.
* Use of **CNN + LSTM** for video-based congestion analysis.
* Deployment on cloud (AWS/GCP/Azure).
* Mobile app for live congestion alerts.

---

## 👩‍💻 Author

**Sakshi Arbad**
MCA Student | Smart Traffic Solutions Developer

---

## 📜 License

This project is open-source for academic and learning purposes.
