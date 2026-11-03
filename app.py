import pickle
from flask import Flask, request, jsonify   
from prometheus_flask_exporter import PrometheusMetrics
app = Flask(__name__)
metrics = PrometheusMetrics(app)

with open("iris_model.pkl" , "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return jsonify({"message": "API is running "})

@app.route("/health")
def health():
    return jsonify({"message": "App is healthy "})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    s_l = data['sepal_length']
    s_w = data['sepal_width']
    p_l = data['petal_length']
    p_w = data['petal_width']
    pred = model.predict([[s_l , s_w , p_l , p_w]])
    return jsonify({"prediction": pred[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

#s_w = float(input("Enter Sepal Width:"))
#s_l = float(input("Enter Sepal Length:"))
#p_w = float(input("Enter Petal Width:"))
#p_l = float(input("Enter Petal Length:"))
#samples = [[1.5 , 2.6 , 4.4 , 2.8]]
#prediction = model.predict([[s_w , s_l , p_w , p_l]])
#print(samples)