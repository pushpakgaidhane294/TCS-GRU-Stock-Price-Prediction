````markdown
# 📈 TCS Stock Price Prediction Using GRU

A machine learning web application that predicts the **next trading day's TCS stock closing price** using a **Gated Recurrent Unit (GRU)** deep learning model.

The project uses historical TCS stock data from **Yahoo Finance**, trains a GRU-based neural network, and provides a Flask web dashboard where users can log in, view stock information, generate predictions, and analyze model performance.

---

## 🚀 Project Overview

Stock prices are time-series data because their values depend on previous observations.

In this project, a **GRU (Gated Recurrent Unit)** neural network is used to learn patterns from historical TCS stock closing prices and predict the expected closing price for the next trading day.

The project includes:

- Historical TCS stock data
- GRU deep learning model
- Data preprocessing and normalization
- Next-day stock price prediction
- UP/DOWN movement prediction
- Expected percentage movement
- MAE and RMSE evaluation
- Historical stock price visualization
- Actual vs predicted visualization
- User registration and login
- SQLite database
- Prediction history
- Flask web application
- Yahoo Finance data refresh
- CSV download
- Render deployment

---

# 🎯 Objectives

The main objectives of this project are:

1. To collect historical TCS stock price data.
2. To preprocess and normalize the stock price data.
3. To use a GRU neural network for time-series prediction.
4. To predict the next trading day's TCS closing price.
5. To determine whether the expected movement is UP or DOWN.
6. To calculate the expected percentage movement.
7. To evaluate the performance of the GRU model.
8. To provide an interactive web interface using Flask.
9. To maintain user login and prediction history.
10. To deploy the application online.

---

# 🧠 Why GRU?

GRU stands for **Gated Recurrent Unit**.

GRU is a type of Recurrent Neural Network (RNN) designed for sequential and time-series data.

Stock prices form a sequence:

```text
Day 1 → Day 2 → Day 3 → Day 4 → ... → Day N
````

The previous stock prices can contain information useful for predicting future prices.

GRU uses gates to control how much previous information should be remembered or forgotten.

Compared with a traditional RNN, GRU can handle long-term dependencies more effectively and is generally simpler than LSTM because it uses fewer gates.

---

# 🔄 Project Workflow

```text
Yahoo Finance
      ↓
Historical TCS Stock Data
      ↓
Data Cleaning
      ↓
Data Normalization
      ↓
Create 60-Day Sequences
      ↓
GRU Model
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Save Trained Model
      ↓
Flask Web Application
      ↓
Latest TCS Data
      ↓
Next Trading Day Prediction
      ↓
UP / DOWN Movement
      ↓
Dashboard
```

---

# 📊 Dataset

The project uses TCS stock data with the Yahoo Finance ticker:

```text
TCS.NS
```

The historical dataset contains:

```text
Date
Close
```

Example:

```text
Date        Close
2018-01-01  1063.44
2018-01-02  1072.31
2018-01-03  1085.27
...
```

The dataset is stored in:

```text
data/TCS_stock.csv
```

The application can also refresh recent market data using Yahoo Finance.

---

# 🔧 Technologies Used

## Programming Language

* Python 3.11

## Machine Learning

* TensorFlow
* Keras
* GRU
* NumPy
* Pandas
* Scikit-learn

## Data Source

* Yahoo Finance
* yfinance

## Web Development

* Flask
* HTML
* CSS
* JavaScript

## Authentication

* Flask-Login
* Werkzeug password hashing

## Database

* SQLite

## Visualization

* Matplotlib
* Chart.js

## Deployment

* Gunicorn
* Render

## Version Control

* Git
* GitHub

---

# 📁 Project Structure

```text
GRU-Stock-Prediction/
│
├── venv/
│
├── data/
│   ├── TCS_stock.csv
│   ├── tcs_historical_price.png
│   ├── actual_vs_predicted.png
│   ├── model_evaluation.json
│   └── latest_prediction.json
│
├── model/
│   ├── gru_model.keras
│   └── scaler.pkl
│
├── templates/
│   ├── login.html
│   ├── register.html
│   └── dashboard.html
│
├── static/
│   ├── style.css
│   ├── tcs_historical_price.png
│   └── actual_vs_predicted.png
│
├── download_data.py
├── explore_data.py
├── preprocess_data.py
├── train.py
├── generate_graphs.py
├── generate_evaluation_cache.py
├── generate_latest_prediction.py
├── app.py
├── database.db
├── requirements.txt
├── README.md
└── .python-version
```

---

# 📌 Important Files

## `download_data.py`

Downloads historical TCS stock data from Yahoo Finance.

---

## `explore_data.py`

Used to inspect the downloaded dataset and understand:

* Number of records
* Columns
* Date range
* Missing values
* Basic statistics

---

## `preprocess_data.py`

Prepares the stock data for GRU training.

Main preprocessing steps include:

* Selecting closing prices
* Normalizing values
* Creating time-series sequences
* Preparing training and testing data

---

## `train.py`

Builds and trains the GRU model.

The model architecture is:

```text
Input
  ↓
GRU(64)
  ↓
Dropout(0.2)
  ↓
GRU(32)
  ↓
Dropout(0.2)
  ↓
Dense(1)
  ↓
Predicted Closing Price
```

The trained model is saved as:

```text
model/gru_model.keras
```

The scaler is saved as:

```text
model/scaler.pkl
```

---

# 🧮 Sequence Length

The model uses the previous:

```text
60 trading days
```

to predict the next trading day's closing price.

For example:

```text
Previous 60 closing prices
          ↓
      GRU Model
          ↓
Next Trading Day Price
```

This gives the model a sequence of historical information instead of using only one previous price.

---

# 📈 Model Evaluation

The model is evaluated using:

## MAE

Mean Absolute Error measures the average absolute difference between the actual and predicted values.

```text
MAE = Average(|Actual - Predicted|)
```

A lower MAE generally indicates better prediction performance.

---

## RMSE

Root Mean Squared Error gives more importance to larger prediction errors.

```text
RMSE = √(Average((Actual - Predicted)²))
```

A lower RMSE generally indicates better performance.

---

## Prediction Accuracy

The project also calculates an accuracy-style value based on the mean absolute percentage error.

The evaluation results are stored in:

```text
data/model_evaluation.json
```

This allows the Flask application to display the model evaluation results without recalculating the entire test set every time the website starts.

---

# ⚡ Next Trading Day Prediction

The application predicts the expected closing price for the next trading day.

The prediction uses the latest 60 available closing prices.

Example:

```text
Current Price       : ₹3,089.89
Predicted Price     : ₹3,142.76
Expected Movement   : +1.71%
Direction            : UP
```

The actual values are generated by the trained GRU model.

---

# 📊 UP / DOWN Prediction

The application compares:

```text
Predicted Price
      vs
Current Price
```

### If:

```text
Predicted Price > Current Price
```

the direction is:

```text
📈 UP
```

### If:

```text
Predicted Price < Current Price
```

the direction is:

```text
📉 DOWN
```

### If:

```text
Predicted Price = Current Price
```

the direction is:

```text
➖ NEUTRAL
```

---

# 📐 Expected Movement

The expected percentage movement is calculated as:

```text
Percentage Change =
((Predicted Price - Current Price) / Current Price) × 100
```

For example:

```text
Current Price = ₹3000
Predicted Price = ₹3060

Percentage Change =
((3060 - 3000) / 3000) × 100

= 2%
```

Therefore:

```text
Expected Movement = +2.00%
Direction = UP
```

---

# 🔐 Login and Registration

The Flask application provides:

* User registration
* User login
* Password hashing
* Session management
* Logout

Authentication is implemented using:

```text
Flask-Login
```

Passwords are securely stored using Werkzeug password hashing rather than storing plain-text passwords.

---

# 🗄️ SQLite Database

SQLite is used to store application data.

The database file is:

```text
database.db
```

The application stores:

## Users

```text
id
username
password
name
```

## Prediction History

```text
user_id
prediction_date
latest_price
predicted_price
difference
percentage_change
direction
created_at
```

This allows users to see their previous predictions.

---

# 🌐 Flask Dashboard

The dashboard provides information such as:

* Current TCS price
* Latest available date
* Predicted next-day price
* Expected movement
* UP/DOWN direction
* Model accuracy
* MAE
* RMSE
* Historical price chart
* Actual vs predicted chart
* Prediction history
* Market data refresh
* Historical CSV download

---

# 🔄 Refresh Market Data

The application provides a market refresh option.

When the user selects:

```text
Refresh Market Data
```

the Flask application retrieves recent TCS data from Yahoo Finance.

The data is merged with the existing historical dataset and duplicate dates are removed.

The updated dataset is stored in:

```text
data/TCS_stock.csv
```

---

# ⚙️ Offline Prediction Architecture

To make the deployed application more stable on a limited server environment, the heavy GRU inference is performed separately using:

```text
generate_latest_prediction.py
```

The generated prediction is stored in:

```text
data/latest_prediction.json
```

The Flask web application reads this lightweight JSON result instead of running TensorFlow inference during the web request.

The architecture is:

```text
TCS Data
   ↓
GRU Model
   ↓
generate_latest_prediction.py
   ↓
latest_prediction.json
   ↓
GitHub
   ↓
Render
   ↓
Flask Dashboard
   ↓
Prediction Result
```

This prevents the production Flask worker from performing heavy TensorFlow inference for every prediction request.

---

# 🧪 Generate Latest Prediction

Activate the virtual environment:

## Windows

```powershell
venv\Scripts\activate
```

Run:

```powershell
python generate_latest_prediction.py
```

This script:

1. Downloads recent TCS data.
2. Updates the historical dataset.
3. Loads the trained GRU model.
4. Loads the scaler.
5. Takes the latest 60 closing prices.
6. Generates the next trading day prediction.
7. Calculates price difference.
8. Calculates percentage movement.
9. Determines UP/DOWN/NEUTRAL.
10. Saves the result to:

```text
data/latest_prediction.json
```

---

# 🖥️ Run the Project Locally

## Step 1: Open the project

```powershell
cd C:\Users\pushp\GRU-Stock-Prediction
```

---

## Step 2: Activate virtual environment

```powershell
venv\Scripts\activate
```

---

## Step 3: Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Step 4: Generate latest prediction

```powershell
python generate_latest_prediction.py
```

---

## Step 5: Start Flask

```powershell
python app.py
```

---

## Step 6: Open the website

Open:

```text
http://127.0.0.1:5000
```

---

# 📦 Requirements

The main dependencies are:

```text
Flask
Flask-Login
Werkzeug
pandas
numpy
scikit-learn
tensorflow
yfinance
joblib
gunicorn
matplotlib
```

Install them using:

```powershell
pip install -r requirements.txt
```

---

# ☁️ Deployment on Render

The application is deployed using:

```text
Render
```

Gunicorn is used as the production WSGI server.

The start command is:

```text
gunicorn app:app
```

The Python version is specified using:

```text
.python-version
```

Current Python version:

```text
3.11.8
```

---

# 🚀 Deployment Workflow

```text
Local Project
     ↓
Git
     ↓
GitHub
     ↓
Render
     ↓
Build
     ↓
Gunicorn
     ↓
Flask Application
     ↓
Online Dashboard
```

After generating a new prediction:

```powershell
python generate_latest_prediction.py
```

Commit the updated prediction file:

```powershell
git add data/latest_prediction.json
git commit -m "Update latest TCS prediction"
git push origin main
```

Render automatically deploys the latest GitHub version if automatic deployment is enabled.

---

# ❤️ Health Check

The Flask application provides:

```text
/health
```

For example:

```text
https://your-render-app.onrender.com/health
```

A successful response is:

```text
OK
```

This endpoint can be used to check whether the Flask server is running.

---

# 📊 Visualizations

The project includes:

## Historical Price Chart

```text
data/tcs_historical_price.png
```

This shows the historical movement of TCS closing prices.

## Actual vs Predicted Chart

```text
data/actual_vs_predicted.png
```

This compares actual stock prices with prices predicted by the GRU model during evaluation.

---

# 🔒 Security

The project includes basic application security features:

* Login authentication
* Password hashing
* Session-based authentication
* Protected dashboard routes
* Protected prediction routes
* Protected CSV download
* HTTP-only session cookies
* Secure session cookies when deployed with Render

---

# 💡 Key Features

| Feature           | Description                         |
| ----------------- | ----------------------------------- |
| GRU Model         | Predicts future TCS closing price   |
| 60-Day Sequence   | Uses previous 60 trading days       |
| Yahoo Finance     | Provides stock market data          |
| Flask             | Web application framework           |
| Login/Register    | User authentication                 |
| SQLite            | Stores users and prediction history |
| UP/DOWN           | Predicts expected direction         |
| Expected Movement | Calculates percentage change        |
| MAE               | Measures average prediction error   |
| RMSE              | Measures larger prediction errors   |
| Charts            | Visualizes stock data               |
| CSV Download      | Downloads historical data           |
| Market Refresh    | Retrieves recent TCS data           |
| Render            | Online deployment                   |

---

# 🧠 Advantages of the Project

1. Uses deep learning for time-series prediction.
2. GRU is suitable for sequential stock data.
3. Provides an easy-to-use web interface.
4. Includes user authentication.
5. Stores prediction history.
6. Provides graphical analysis.
7. Uses real stock-market data.
8. Provides UP/DOWN movement information.
9. Separates heavy ML processing from the production web server.
10. Can be accessed through an online deployment.

---

# ⚠️ Limitations

Stock prices are affected by many external factors.

The model mainly learns patterns from historical price data.

It does not directly consider factors such as:

* Company announcements
* News
* Global economic conditions
* Market sentiment
* Political events
* Unexpected market events
* Trading volume
* Technical indicators not included in the model

Therefore, the prediction should be considered a **machine learning estimate**, not a guaranteed future stock price.

---

# 🔮 Future Scope

The project can be improved by adding:

* Multiple stock support
* Technical indicators
* Moving averages
* RSI
* MACD
* Trading volume
* News sentiment analysis
* Financial-news integration
* More advanced GRU/LSTM architectures
* Transformer-based time-series models
* Real-time market APIs
* Automatic scheduled prediction generation
* Cloud database
* Portfolio tracking
* Prediction confidence score
* Mobile-friendly dashboard
* Advanced model comparison

---

# 🎓 Academic Project

## Project Title

**Stock Price Prediction Using GRU**

## Domain

**Artificial Intelligence and Machine Learning**

## Problem Type

**Time-Series Forecasting**

## Model

**Gated Recurrent Unit (GRU)**

## Dataset

**TCS Stock Market Data**

## Web Framework

**Flask**

## Deployment

**Render**

---

# 👨‍💻 Project Role

My contribution to the project includes:

* Collecting and preprocessing TCS stock data
* Implementing the GRU model
* Training the deep learning model
* Evaluating model performance
* Integrating the trained model with Flask
* Developing the login and registration system
* Developing the prediction dashboard
* Implementing UP/DOWN movement calculation
* Implementing prediction history
* Creating data visualizations
* Integrating Yahoo Finance data refresh
* Implementing offline prediction generation for deployment
* Deploying the Flask application using Render

---

# 📌 Disclaimer

This project is developed for **educational and academic purposes**.

The predicted stock price is generated by a machine learning model and should not be considered financial advice or a guaranteed future market price.

Always perform independent research before making any financial decision.

---

# ⭐ Project Summary

This project demonstrates how a **GRU deep learning model** can be applied to stock-market time-series data to predict the next trading day's TCS closing price.

The trained model is integrated with a **Flask web application** that provides authentication, market-data visualization, prediction results, UP/DOWN movement, prediction history, and model evaluation.

The application is deployed online using **Render**, while heavy GRU prediction generation is handled separately to keep the production web application lightweight and stable.

```
