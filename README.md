# Smart Traffic Congestion & Prediction System

The Smart Traffic Congestion and Prediction System is an AI-powered solution that uses Machine Learning, Deep Learning, IoT, Cloud, and Big Data to monitor real-time traffic, predict congestion, and control traffic signals automatically.

## Project Overview
1. Detects vehicles using YOLO.
2. Predicts upcoming traffic using ML/DL models.
3. Dynamically controls green/red signals.
4. Provides a live Streamlit dashboard.
5. Integrates APIs (like Google Maps) for external traffic data.

##  Key Features
1. Real-Time Vehicle Detection
- Uses YOLOv8 deep learning model.
- Detects and counts: cars, bikes, buses, trucks.
- Works with CCTV/IP cameras or video streams.

 2. Traffic Prediction
- ML Models: Random Forest, XGBoost
- DL Model: LSTM (Time-series forecasting)
- Predicts congestion 5–15 minutes ahead.

 3. Smart Adaptive Signal Control
- Automatically adjusts:
- Green light duration
- Red light duration
Based on:
- Traffic density
- Peak hour patterns
- Prediction values

4. Streamlit Live Dashboard
Shows:
- 4 real-time camera feeds
- Vehicle count
- Predicted congestion level
- Adaptive signal time
- Traffic graphs
- Map-based traffic visualization

5. IoT + Cloud + Big Data Support
- IoT Sensors
- Google Maps real-time API
- AWS/GCP cloud storage
- Kafka/Spark for large-scale traffic data (optional)

## **Project Structure (Corrected & Clean)**
```text
Smart-Traffic-Congestion-Prediction/
│
├── data/
│
├── model/
│
├── src/
│   ├── controller.py
│   ├── dashboard.py
│   ├── detection.py
│   ├── main.py
│   ├── timer.py
│   ├── static/
│   └── __pycache__/
│
├── traffic_env/
│
├── venv/
│
├── README.md
├── requirements.txt
├── yolov8n.pt
└── yolov8s.pt
```
## **Technologies Used**
- AI/ML Algorithms - Random Forest, XGBoost,opencv-python
- Libraries - ultralytics, numpy, pandas, streamlit, matplotlib
- Language - Python
- Deep Learning - YOLOv8, LSTM, PyTorch
- Web Dashboard - Streamlit
- Backend API - FastAPI
- IoT - IP Cameras / Sensors
- Cloud -	AWS / GCP / Azure
- Big Data - Spark, Kafka (optional)

##  Workflow / Methodology
1. Data Collection
- CCTV cameras, IoT sensors, Google Maps API collect vehicle count, speed, time, weather.
2. Vehicle Detection (YOLOv8)
- AI model detects vehicles from live video and calculates real-time traffic density.
3. Preprocessing
- Clean data, extract frames, remove noise, convert video to numerical features.
4. Feature Engineering
- Create features like vehicle count, peak hour, density level, weather, historical traffic.
5. Model Training (RF + XGBoost + LSTM)
- Random Forest → classify congestion
6. XGBoost → predict delay
- LSTM → forecast future traffic
7. Traffic Prediction
- Output congestion level (Low/Medium/High) and expected delay.
8. Intelligent Signal Control
- Auto-adjust green/red timing based on traffic density.
9. Dashboard (Streamlit)
Shows live feed, vehicle count, predictions, Google Maps traffic, signal timers.
10. Cloud + Big Data
- Store and analyze data using AWS/GCP or Spark/Kafka.

## How to Run the Project
## Step 1: Create env
```bash
python -m venv venv
```

## Step 1: Activate venv
source venv/Scripts/activate

## Step 1: Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

### Step 2: Run Model Training
python src/controller.py
python src/detection.py
python src/main.py
python src/timer.py

### Step 4: Run Dashboard
streamlit run src/dashboard.py

##  Example Prediction Code
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

##  Future Enhancements
- Integration with IoT road sensors.
- Use of CNN + LSTM for video-based congestion analysis.
-  Deployment on cloud (AWS/GCP/Azure).
- Mobile app for live congestion alerts.

##  Author
1. Sakshi Arbad
2. Saloni Dhanvij
3. Shravani Chandodkar
4. Shivakanya Ladekar 
- Maharashtra Institute of technology Chhatrapati Sambhajinagar
- Fial Year B.Tech-CSE
