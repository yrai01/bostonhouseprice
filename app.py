import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open("regmodel.pkl", "rb"))

# Define and load the scaler
scaler = pickle.load(open("scaling.pkl", "rb"))

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json
    print(data)
    input_data = np.array(list(data.values())).reshape(1, -1)
    new_data = scaler.transform(input_data)
    output = model.predict(new_data)
    return jsonify({"prediction": output[0]})

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    output = model.predict(final_input)[0]
    return render_template("home.html", prediction_text="The predicted house price is {:.2f}".format(output)) 

if __name__ == "__main__":
    app.run(debug=True)
