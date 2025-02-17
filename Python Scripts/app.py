from flask import Flask, render_template, request, send_from_directory, url_for
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt  # Make sure matplotlib is installed
import os

app = Flask(__name__)

# ... (rest of your code)

app = Flask(__name__)

model = None
scaler = None
features = None
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return """
    <h1> Upload Online Retail Data as "Online Retail.xlsx" file. </h1>
    <form action = '/train' method = 'post' enctype = 'multipart/form-data'>
    <input type = 'file' name = 'file' required>
    <input type='submit' value='Proceed'>
    </form>
    """

@app.route('/train', methods=['POST'])
def train():
    global model, scaler, features
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        ml_retail_df = pd.read_excel(filepath)

        ml_retail_df = ml_retail_df.dropna()
        ml_retail_df.drop_duplicates(inplace=True)
        filtered_df = ml_retail_df[~ml_retail_df['InvoiceNo'].astype(str).str.startswith('C')]
        filtered_df = ml_retail_df[(ml_retail_df['Quantity'] > 0) & (ml_retail_df['UnitPrice'] > 0)]

        filtered_df["TotalPrice"] = filtered_df["Quantity"] * filtered_df["UnitPrice"]
        filtered_df["InvoiceDate"] = pd.to_datetime(filtered_df["InvoiceDate"])

        filtered_df["Year"] = filtered_df["InvoiceDate"].dt.year
        filtered_df["Month"] = filtered_df["InvoiceDate"].dt.month
        filtered_df["Day"] = filtered_df["InvoiceDate"].dt.day
        filtered_df["Hour"] = filtered_df["InvoiceDate"].dt.hour
        filtered_df["Weekday"] = filtered_df["InvoiceDate"].dt.weekday

        customer_df = filtered_df.groupby('CustomerID').agg(
            TotalSpent=('TotalPrice', 'sum'),
            NumPurchases=('InvoiceNo', 'nunique'),
            AvgBasketSize=('Quantity', 'mean'),
            AvgOrderValue=('TotalPrice', 'mean'),
            UniqueProducts=('StockCode', 'nunique')
        ).reset_index()

        X = customer_df.drop(columns=["CustomerID", "TotalSpent"])
        y = pd.qcut(customer_df["TotalSpent"], q=3, labels=[0, 1, 2])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=100, random_state=42) # Only Random Forest
        model.fit(X_train_scaled, y_train)

        joblib.dump(model, "../best_model.pkl")  # Save the trained model
        joblib.dump(scaler, "../scaler.pkl") # Save the scaler
        features = list(X.columns) # Save the features

        np.save("../X_test_scaled.npy", X_test_scaled) # Save the test data
        np.save("../y_test.npy", y_test)

        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        cm_filename = "confusion_matrix.png"  # Filename for the plot
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], cm_filename)) # Save the confusion matrix plot
        plt.close() # Close the plot to release memory

        return f"""
        <h1>Model Training Complete</h1>
        <p>Accuracy: {accuracy:.4f}</p>
        <p>Classification Report: <pre>{report}</pre></p>
        <img src="{url_for('static', filename='uploads/' + cm_filename)}", alt="Confusion Matrix"> <br>  </a> <br>
        <a href='/predict'>Go to test prediction</a>
        """

    except Exception as e:
        return f"An error occurred during training: {e}"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global model, scaler, features

    if model is None or scaler is None or features is None:
        try:
            model = joblib.load("../best_model.pkl")
            scaler = joblib.load("../scaler.pkl")
            # Load features from a file (if you saved them) or define them directly
            features = list(pd.read_csv("../cleaned_customer_data.csv").drop(columns=["CustomerID", "TotalSpent"]).columns)

        except Exception as e:
            return f"<h1>Error loading model or scaler: {e}</h1><a href='/'>Back to Home</a>"

    if request.method == 'POST':
        try:
            input_values = [float(request.form[feature]) for feature in features]
            input_array = np.array(input_values).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            predicted_segment = model.predict(input_scaled)[0]

            return f"""
            <h1>Predicted Customer Segment</h1>
            <p><strong>Segment:</strong> {predicted_segment}</p>
            <a href='/predict'>Predict another test case</a> | <a href='/'>Back to home</a>
            """
        except Exception as e:
            return f"<h1>Error during prediction: {str(e)}</h1><a href='/predict'>Try again</a>"

    form_html = "<h1>Enter Feature Values</h1><form method='post'>"
    for feature in features:
        form_html += f"<label>{feature}: <input type='text' name='{feature}' required></label><br>"
    form_html += "<input type='submit' value='Predict'></form>"

    return form_html

from flask import send_from_directory, url_for
import os

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(app.config['UPLOAD_FOLDER'], path)

if __name__ == '__main__':
    app.run(debug=True)
