# 🍔 Food Delivery Time Prediction

A Machine Learning-based web application that predicts the estimated delivery time of food orders based on various order and delivery-related features. The project uses regression models to analyze delivery data and provides real-time predictions through an interactive Flask web interface.

---

## 📌 Project Overview

Timely food delivery plays a crucial role in customer satisfaction for online food delivery platforms. This project predicts the expected delivery time by analyzing factors such as distance, weather conditions, traffic, vehicle type, and order characteristics.

The trained machine learning model is deployed as a Flask web application, allowing users to enter order details and instantly receive an estimated delivery time.

---

## 🚀 Features

- Predicts food delivery time in real time
- User-friendly Flask web interface
- Machine Learning regression model
- Model comparison for selecting the best-performing algorithm
- Easy deployment and reproducibility

---

## 🛠️ Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS
- Pickle

---

## 📂 Project Structure

```
Food_Delivery_Time_Prediction/
│
├── static/                 # CSS and static resources
├── templates/              # HTML templates
├── app.py                  # Flask application
├── train_model.py          # Model training script
├── model.pkl               # Trained machine learning model
├── Food_Delivery_Times.csv # Dataset
├── model_comparison.csv    # Model evaluation results
├── requirements.txt        # Project dependencies
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/Food_Delivery_Time_Prediction.git
```

### Navigate to the project

```bash
cd Food_Delivery_Time_Prediction
```

### Create a virtual environment (Optional)

```bash
python -m venv venv
```

### Activate the virtual environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📊 Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Model Serialization using Pickle
7. Flask Deployment
8. Real-Time Prediction

---

## 📈 Model Evaluation

Multiple regression algorithms were trained and compared to select the best-performing model.

Evaluation metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The comparison results are available in:

```
model_comparison.csv
```

---

## 📷 Screenshots

Add screenshots here after uploading them to the `assets` folder.

### Home Page

```
assets/homepage.png
```

### Prediction Result

```
assets/result.png
```

---

## 🔮 Future Improvements

- Deploy on Streamlit or Render
- Add Google Maps API integration
- Include live weather and traffic data
- Improve prediction accuracy with advanced ensemble models
- Build a responsive UI

---

## 👩‍💻 Author

**Arpita**