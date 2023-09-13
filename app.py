import pickle
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open("regmodel.pkl", "rb"))

# Define and load the scaler
with open("scaling.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = model.predict(new_data)
    print(output)
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)
