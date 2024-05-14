from flask import Flask, request
import pandas as pd
from joblib import load

app = Flask(__name__)

# Load the saved models
ca_model = load('G:\\Intern\\final_predict\\RandomForest_ModelCa.joblib')
hb_model = load('G:\\Intern\\final_predict\\RandomForest_ModelHb.joblib')
gl_model = load('G:\\Intern\\final_predict\\RandomForest_ModelGl.joblib')  # Assuming the same model for both hb and gl

# Load the dataset
df_ca = pd.read_csv("G:\\Intern\\final_predict\\interpolatedca.csv")
df_hb = pd.read_csv("G:\\Intern\\final_predict\\interpolatedHb.csv")
df_gl = pd.read_csv("G:\\Intern\\final_predict\\interpolatedgl.csv")  # Assuming the same dataset for hb and gl

# Function to predict concentration value given values from the 2nd to 4th columns
def predict_concentration(model, df, input_value):
    # Predict concentration value using the loaded Random Forest model
    concentration_prediction = model.predict([[input_value] * 3])
    return concentration_prediction[0]

@app.route("/")
def root():
    with open("G:\\Intern\\final_predict\\result.html") as file:
        return file.read()

@app.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form['model']
    input_values_str = request.form['input_value']
    input_values = [float(value.strip()) for value in input_values_str.split(',')]
    results = "<table border='1'><tr><th>Input Value</th><th>Predicted Concentration</th></tr>"

    if model_choice == "ca":
        for value in input_values:
            predicted_concentration = format(predict_concentration(ca_model, df_ca, value), ".2f")
            results += f"<tr><td>{value}</td><td>{predicted_concentration}</td></tr>"
    elif model_choice == "hb":
        for value in input_values:
            predicted_concentration = format(predict_concentration(hb_model, df_hb, value),".2f")
            results += f"<tr><td>{value}</td><td>{predicted_concentration}</td></tr>"
    elif model_choice == "gl":
        for value in input_values:
            predicted_concentration = format(predict_concentration(gl_model, df_gl, value),".2f")
            results += f"<tr><td>{value}</td><td>{predicted_concentration}</td></tr>"
    else:
        return "Invalid model choice. Please choose 'ca', 'hb', or 'gl'."

    results += "</table>"
    return results

if __name__ == '__main__':
    app.run(debug=True)
