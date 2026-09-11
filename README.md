# 📈 TCS Stock Price Prediction Using GRU

A machine learning web application that predicts the **next trading day's TCS stock closing price** using a **Gated Recurrent Unit (GRU)** deep learning model.

The project uses historical TCS stock data from **Yahoo Finance**, trains a GRU-based neural network, and provides a **Flask web dashboard** where users can log in, view stock information, generate predictions, and analyze model performance.

---

## 🚀 Project Overview

Stock prices are time-series data because their values depend on previous observations.

In this project, a **GRU (Gated Recurrent Unit)** neural network is used to learn patterns from historical TCS stock closing prices and predict the expected closing price for the next trading day.

### Main Features

- Historical TCS stock data
- GRU deep learning model
- Data preprocessing and normalization
- Next trading day stock price prediction
- UP / DOWN / NEUTRAL movement prediction
- Expected percentage movement
- MAE and RMSE model evaluation
- Historical stock price visualization
- Actual vs Predicted visualization
- User registration and login
- SQLite database
- Prediction history
- Flask web dashboard
- Yahoo Finance market-data refresh
- Historical CSV download
- Render deployment

---

# 🎯 Objectives

1. Collect historical TCS stock price data.
2. Preprocess and normalize the stock price data.
3. Create time-series sequences for the GRU model.
4. Train a GRU neural network.
5. Predict the next trading day's TCS closing price.
6. Determine the expected UP / DOWN movement.
7. Calculate the expected percentage movement.
8. Evaluate the model using MAE and RMSE.
9. Provide an interactive Flask web application.
10. Store user accounts and prediction history.
11. Deploy the application online using Render.

---

# 🧠 Why GRU?

**GRU (Gated Recurrent Unit)** is a type of Recurrent Neural Network (RNN) designed for sequential and time-series data.

Stock prices form a sequence:

```text
Day 1 → Day 2 → Day 3 → Day 4 → ... → Day N
```

The previous stock prices contain information that can be useful for predicting future prices.

GRU uses gates to control how much previous information should be remembered or forgotten.

Compared with a traditional RNN, GRU can handle long-term dependencies more effectively and has a simpler architecture than LSTM.

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
Generate Latest Prediction
      ↓
latest_prediction.json
      ↓
Flask Web Application
      ↓
Dashboard
      ↓
UP / DOWN Prediction
```

---

# 📊 Dataset

The project uses TCS stock data from Yahoo Finance.

### Yahoo Finance Ticker

```text
TCS.NS
```

### Dataset Columns

```text
Date
Close
```

The dataset is stored at:

```text
data/TCS_stock.csv
```

Example:

```text
Date        Close
2018-01-01  1063.44
2018-01-02  1072.31
2018-01-03  1085.27
...
```

Recent market data can also be refreshed using Yahoo Finance.

---

# 🔧 Technologies Used

### Programming Language

- Python 3.11

### Machine Learning

- TensorFlow
- Keras
- GRU
- NumPy
- Pandas
- Scikit-learn

### Data Source

- Yahoo Finance
- yfinance

### Web Development

- Flask
- HTML
- CSS
- JavaScript
- Chart.js

### Authentication

- Flask-Login
- Werkzeug password hashing

### Database

- SQLite

### Visualization

- Matplotlib
- Chart.js

### Deployment

- Gunicorn
- Render

### Version Control

- Git
- GitHub

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

### `download_data.py`

Downloads historical TCS stock data from Yahoo Finance.

### `explore_data.py`

Used to inspect the dataset, including:

- Number of records
- Columns
- Date range
- Missing values
- Basic statistics

### `preprocess_data.py`

Prepares the stock data for GRU training by:

- Selecting closing prices
- Normalizing values
- Creating time-series sequences
- Preparing training and testing data

### `train.py`

Builds and trains the GRU model.

### `generate_graphs.py`

Generates the historical price and actual-vs-predicted graphs.

### `generate_evaluation_cache.py`

Stores model evaluation results in JSON format so that the Flask application does not need to recalculate the complete evaluation every time.

### `generate_latest_prediction.py`

Runs the trained GRU model locally and generates the latest prediction.

### `app.py`

Runs the Flask web application and provides:

- Login
- Registration
- Dashboard
- Prediction results
- Prediction history
- Market refresh
- CSV download
- Health check

---

# 🧠 GRU Model Architecture

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

```text
Previous 60 Closing Prices
            ↓
        GRU Model
            ↓
Next Trading Day Price
```

Using multiple previous observations allows the model to learn patterns from a sequence rather than relying on only one previous price.

---

# 📈 Model Evaluation

The model is evaluated using **MAE** and **RMSE**.

## MAE

Mean Absolute Error measures the average absolute difference between actual and predicted values.

```text
MAE = Average(|Actual - Predicted|)
```

A lower MAE generally indicates better prediction performance.

## RMSE

Root Mean Squared Error gives more importance to larger prediction errors.

```text
RMSE = √(Average((Actual - Predicted)²))
```

A lower RMSE generally indicates better performance.

The evaluation results are stored in:

```text
data/model_evaluation.json
```

---

# ⚡ Next Trading Day Prediction

The application predicts the expected closing price for the next trading day using the latest 60 available closing prices.

The prediction process is:

```text
Latest TCS Data
      ↓
Last 60 Closing Prices
      ↓
Data Scaling
      ↓
GRU Model
      ↓
Predicted Price
      ↓
UP / DOWN / NEUTRAL
```

---

# 📊 UP / DOWN Prediction

The application compares the predicted price with the latest available price.

```text
Predicted Price > Current Price
        ↓
       📈 UP
```

```text
Predicted Price < Current Price
        ↓
      📉 DOWN
```

```text
Predicted Price = Current Price
        ↓
     ➖ NEUTRAL
```

---

# 📐 Expected Movement

The expected percentage movement is calculated using:

```text
Percentage Change =
((Predicted Price - Current Price) / Current Price) × 100
```

Example:

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

- User registration
- User login
- Password hashing
- Session management
- Logout
- Protected dashboard routes

Authentication is implemented using:

```text
Flask-Login
```

Passwords are stored using Werkzeug password hashing instead of plain-text passwords.

---

# 🗄️ SQLite Database

SQLite is used to store application data.

Database file:

```text
database.db
```

### Users

```text
id
username
password
name
```

### Prediction History

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

This allows users to view their previous prediction results.

---

# 🌐 Flask Dashboard

The dashboard provides:

- Current TCS price
- Latest available date
- Predicted next-day price
- Expected percentage movement
- UP / DOWN / NEUTRAL direction
- Model accuracy
- MAE
- RMSE
- Historical price chart
- Actual vs Predicted chart
- Prediction history
- Market data refresh
- Historical CSV download

---

# 🔄 Refresh Market Data

The application can retrieve recent TCS data from Yahoo Finance.

The refreshed data is merged with the existing dataset and duplicate dates are removed.

The updated dataset is stored in:

```text
data/TCS_stock.csv
```

---

# ⚙️ Offline Prediction Architecture

To make the deployed Flask application lightweight and stable, the heavy TensorFlow GRU inference is performed separately using:

```text
generate_latest_prediction.py
```

The prediction result is saved as:

```text
data/latest_prediction.json
```

The Flask application reads this lightweight JSON file instead of running TensorFlow inference during every web request.

### Architecture

```text
TCS Market Data
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
Flask Application
      ↓
Dashboard
      ↓
Prediction Result
```

This separates the heavy machine-learning prediction process from the production web server.

---

# 🧪 Generate Latest Prediction

Activate the virtual environment:

```powershell
venv\Scripts\activate
```

Run:

```powershell
python generate_latest_prediction.py
```

The script:

1. Downloads recent TCS data.
2. Updates the historical dataset.
3. Loads the trained GRU model.
4. Loads the scaler.
5. Takes the latest 60 closing prices.
6. Generates the next trading day prediction.
7. Calculates the price difference.
8. Calculates percentage movement.
9. Determines UP / DOWN / NEUTRAL.
10. Saves the result to `data/latest_prediction.json`.

---

# 🖥️ Run the Project Locally

## Step 1: Open the project

```powershell
cd C:\Users\pushp\GRU-Stock-Prediction
```

## Step 2: Activate the virtual environment

```powershell
venv\Scripts\activate
```

## Step 3: Install dependencies

```powershell
pip install -r requirements.txt
```

## Step 4: Generate the latest prediction

```powershell
python generate_latest_prediction.py
```

## Step 5: Start Flask

```powershell
python app.py
```

## Step 6: Open the website

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

The Flask application is deployed using **Render**.

Gunicorn is used as the production WSGI server.

### Start Command

```text
gunicorn app:app
```

### Python Version

The project uses:

```text
3.11.8
```

The version is specified in:

```text
.python-version
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

After generating a new prediction locally:

```powershell
python generate_latest_prediction.py
```

Commit the updated files:

```powershell
git add data/latest_prediction.json
git add data/TCS_stock.csv
git commit -m "Update latest TCS prediction"
git push origin main
```

Render can automatically deploy the latest GitHub version when automatic deployment is enabled.

---

# ❤️ Health Check

The Flask application provides a health-check endpoint:

```text
/health
```

Example:

```text
https://tcs-gru-stock-price-prediction-3.onrender.com/health
```

A successful response is:

```text
OK
```

This endpoint can be used to check whether the Flask server is running.

---

# 📊 Visualizations

## Historical Price Chart

```text
data/tcs_historical_price.png
```

Shows the historical movement of TCS closing prices.

## Actual vs Predicted Chart

```text
data/actual_vs_predicted.png
```

Compares actual stock prices with prices predicted by the GRU model during evaluation.

---

# 🔒 Security

The application includes:

- Login authentication
- Password hashing
- Session-based authentication
- Protected dashboard routes
- Protected prediction routes
- Protected CSV download
- HTTP-only session cookies
- Secure session cookies when deployed with Render

---

# 💡 Key Features

| Feature | Description |
|---|---|
| GRU Model | Predicts future TCS closing price |
| 60-Day Sequence | Uses previous 60 trading days |
| Yahoo Finance | Provides stock market data |
| Flask | Web application framework |
| Login/Register | User authentication |
| SQLite | Stores users and prediction history |
| UP/DOWN | Predicts expected direction |
| Expected Movement | Calculates percentage change |
| MAE | Measures average prediction error |
| RMSE | Measures larger prediction errors |
| Charts | Visualizes stock data |
| CSV Download | Downloads historical data |
| Market Refresh | Retrieves recent TCS data |
| Render | Online deployment |

---

# 🧠 Advantages

1. Uses deep learning for time-series prediction.
2. GRU is suitable for sequential stock data.
3. Provides an easy-to-use web interface.
4. Includes user authentication.
5. Stores prediction history.
6. Provides graphical analysis.
7. Uses real stock-market data.
8. Provides UP/DOWN movement information.
9. Separates heavy ML processing from the production web server.
10. Can be deployed online.

---

# ⚠️ Limitations

Stock prices are affected by many external factors.

The model mainly learns patterns from historical price data.

It does not directly consider:

- Company announcements
- News
- Global economic conditions
- Market sentiment
- Political events
- Unexpected market events
- Trading volume
- Technical indicators not included in the model

Therefore, the prediction should be considered a **machine learning estimate**, not a guaranteed future stock price.

---

# 🔮 Future Scope

The project can be improved by adding:

- Multiple stock support
- Technical indicators
- Moving averages
- RSI
- MACD
- Trading volume
- News sentiment analysis
- Financial-news integration
- Advanced GRU/LSTM architectures
- Transformer-based time-series models
- Real-time market APIs
- Automatic scheduled prediction generation
- Cloud database
- Portfolio tracking
- Prediction confidence score
- Mobile-friendly dashboard
- Advanced model comparison

---

# 🎓 Academic Project

### Project Title

**Stock Price Prediction Using GRU**

### Domain

**Artificial Intelligence and Machine Learning**

### Problem Type

**Time-Series Forecasting**

### Model

**Gated Recurrent Unit (GRU)**

### Dataset

**TCS Stock Market Data**

### Web Framework

**Flask**

### Deployment

**Render**

---

# 👨‍💻 Project Role

My contribution to the project includes:

- Collecting and preprocessing TCS stock data
- Implementing the GRU model
- Training the deep learning model
- Evaluating model performance
- Integrating the model with Flask
- Developing the login and registration system
- Developing the prediction dashboard
- Implementing UP/DOWN movement calculation
- Implementing prediction history
- Creating data visualizations
- Integrating Yahoo Finance data refresh
- Implementing offline prediction generation
- Deploying the Flask application using Render

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
