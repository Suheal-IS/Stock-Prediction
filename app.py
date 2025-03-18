
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from pickle import load
from nltk.sentiment import SentimentIntensityAnalyzer
import io
import base64



from flask import Flask, jsonify, render_template, request, send_file, redirect, url_for, flash

import sqlite3 as sql




app = Flask(__name__)
app.secret_key = 'your_secret_key_here'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/userlogin', methods=['GET', 'POST'])
def userlogin():
    msg = None
    if request.method == "POST":
        email = request.form['email']
        password = request.form['pwd']

        with sql.connect("data.db") as con:
            c = con.cursor()
            c.execute("SELECT email, password FROM users WHERE email = ? AND password = ?", (email, password))
            r = c.fetchall()
            if r:
                return render_template("UserHome.html")
            else:
                msg = "Invalid email or password. Please try again."

    return render_template("userlogin.html", msg=msg)


@app.route('/adminlogin', methods=['GET', 'POST'])
def adminlogin():
    msg = None
    if request.method == "POST":
        email = request.form['email']
        password = request.form['pwd']

        with sql.connect("data.db") as con:
            c = con.cursor()
            c.execute("SELECT email, password FROM admin WHERE email = ? AND password = ?", (email, password))
            r = c.fetchall()
            if r:
                return render_template("AdminHome.html")
            else:
                msg = "Invalid email or password. Please try again."

    return render_template("adminlogin.html", msg=msg)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    msg = None
    if request.method == "POST":
        username = request.form['name']
        email = request.form['email']
        mobile = request.form['mobile']
        upassword = request.form['pwd']
        
        try:
            with sql.connect("data.db") as con:
                c = con.cursor()
                c.execute("INSERT INTO users (username, email, mobile, password) VALUES (?, ?, ?, ?)", 
                          (username, email, mobile, upassword))
                con.commit()
                success_msg = "Signup Successful. Please login."
        except sql.Error as e:
            con.rollback()
            msg = f"Error occurred: {e}"
        
        if 'success_msg' in locals():
            return render_template("signup.html", success_msg=success_msg)
        else:
            return render_template("signup.html", msg=msg)
    
    return render_template("signup.html", msg=msg)


@app.route('/userhome')
def userhome():
    return render_template('UserHome.html')


@app.route('/userlogout')
def user_logout():
    return render_template('index.html')


@app.route('/userfaqlist')
def user_faq_list():
    conn = sql.connect('data.db')
    faqs = conn.execute('SELECT * FROM faq').fetchall()
    conn.close()
    return render_template('UserFaqList.html', faqs=faqs)


@app.route('/adminhome')
def adminhome():
    return render_template('AdminHome.html')


def get_users():
    conn = sql.connect('data.db')
    c = conn.cursor()
    c.execute("SELECT id, username, email, mobile FROM users")
    users = c.fetchall()
    conn.close()
    return users


@app.route('/adminuserslist')
def admin_users_list():
    users = get_users()
    return render_template('AdminUsersList.html', users=users)


@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    conn = sql.connect('data.db')
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('admin_users_list'))


@app.route('/adminfaqlist')
def faq_list():
    conn = sql.connect('data.db')
    faqs = conn.execute('SELECT * FROM faq').fetchall()
    conn.close()
    return render_template('AdminFaqList.html', faqs=faqs)


@app.route('/delete_faq/<int:faq_id>', methods=['POST'])
def delete_faq(faq_id):
    conn = sql.connect('data.db')
    conn.execute('DELETE FROM faq WHERE id = ?', (faq_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('faq_list'))


@app.route('/add_faq', methods=['GET', 'POST'])
def add_faq():
    if request.method == 'POST':
        subject = request.form['subject']
        answer = request.form['answer']
        conn = sql.connect('data.db')
        conn.execute('INSERT INTO faq (subject, answer) VALUES (?, ?)', (subject, answer))
        conn.commit()
        conn.close()
        return redirect(url_for('faq_list'))
    return render_template('AdminAddFAQ.html')


@app.route('/adminlogout')
def admin_logout():
    return redirect(url_for('adminlogin'))










# Load the sentiment analysis model (NLTK)
analyzer = SentimentIntensityAnalyzer()

# Load stock and tweet datasets
stock_name = 'AMZN'
df = pd.read_csv('Dataset/stock_tweets.csv')
df = df[df['Stock Name'] == stock_name]
df['Date'] = pd.to_datetime(df['Date']).dt.date

# Sentiment analysis
df['sentiment_score'] = df['Tweet'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'] if isinstance(x, str) else 0)
df = df[['Date', 'sentiment_score']]
df = df.groupby('Date').mean().reset_index()

# Load stock price data
stock_data = pd.read_csv('Dataset/stock_yfinance_data.csv')
stock_data = stock_data[stock_data['Stock Name'] == stock_name]
stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date

# Merge sentiment data with stock data
final_df = stock_data.merge(df, on='Date', how='left').fillna(0)

# Load the saved scaler
scaler = load(open('scaler.pkl', 'rb'))

# Normalize the data
scaled_data = scaler.transform(final_df.drop(columns=['Date', 'Stock Name']))

# Prepare sequences for LSTM (as done in training)
def create_sequences(data, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length][0])  # Predict Close Price
    return np.array(X), np.array(y)

X_data, y_data = create_sequences(scaled_data)

# Load the trained model
model = load_model('stock_lstm_model.h5')

# Prediction function for future days
def predict_future_prices(model, scaled_data, seq_length, num_days):
    latest_sequence = scaled_data[-seq_length:]
    latest_sequence = np.copy(latest_sequence)
    future_predictions = []

    for _ in range(num_days):
        prediction = model.predict(np.expand_dims(latest_sequence, axis=0))
        predicted_price = scaler.inverse_transform(np.concatenate([prediction, np.zeros((prediction.shape[0], scaler.n_features_in_ - 1))], axis=1))[:, 0]
        future_predictions.append(predicted_price[0])
        latest_sequence = np.roll(latest_sequence, shift=-1, axis=0)
        latest_sequence[-1, 0] = predicted_price[0]

    return future_predictions


# Route for displaying the prediction results
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'GET':
        return render_template('predict.html')  # When the page is accessed via GET (link click)
  
    if request.method == 'POST':
        option = request.form.get('option')
        print(f"Selected option: {option}")  # This will help you debug

        if request.form.get('option') == 'historical':
            # Historical prediction: Plot actual vs predicted stock prices
            predictions = model.predict(X_data)
            predictions_rescaled = scaler.inverse_transform(np.concatenate([predictions, np.zeros((predictions.shape[0], scaler.n_features_in_ - 1))], axis=1))[:, 0]
            y_data_rescaled = scaler.inverse_transform(np.concatenate([y_data.reshape(-1, 1), np.zeros((y_data.shape[0], scaler.n_features_in_ - 1))], axis=1))[:, 0]
            
            # Plot the results
            plt.figure(figsize=(14, 6))
            plt.plot(y_data_rescaled, label='Actual Prices', color='blue')
            plt.plot(predictions_rescaled, label='Predicted Prices', color='red')
            plt.xlabel('Time')
            plt.ylabel('Stock Price')
            plt.title(f'{stock_name} Stock Price Prediction')
            plt.legend()

            # Save the plot as a base64 image
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            img_base64 = base64.b64encode(img.getvalue()).decode()

            return render_template('result.html', img_base64=img_base64, message="Historical Prediction")

        elif request.form.get('option') == 'future':
            # Future prediction
            num_days = int(request.form['num_days'])
            future_prices = predict_future_prices(model, scaled_data, seq_length=5, num_days=num_days)

            # Plot the predictions
            plt.figure(figsize=(14, 6))
            plt.plot(range(1, num_days + 1), future_prices, label=f'Predicted Stock Prices for {num_days} days', color='red')
            plt.xlabel('Days in the Future')
            plt.ylabel('Stock Price')
            plt.title(f'{stock_name} Stock Price Prediction for the Next {num_days} Days')
            plt.legend()

            # Save the plot as a base64 image
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            img_base64 = base64.b64encode(img.getvalue()).decode()

            return render_template('result.html', img_base64=img_base64, message=f"Predicted Stock Prices for {num_days} Days")
    
    return render_template('predict.html')  # When the page is accessed via GET (link click)



if __name__ == '__main__':
    app.run(debug=True)
