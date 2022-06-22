# Importing the libraries
from flask import Flask, render_template, request
import pickle
import numpy as np


# Global variables
app = Flask(__name__)
loaded_model = pickle.load(open('Model.pkl', 'rb'))


# User defined routes
@app.route("/")
def home():
    return render_template("form.html")

@app.route("/prediction", methods=['POST'])
def predict():
    Pclass = request.form['Pclass']
    Age = request.form['Age']
    SibSp = request.form['SibSp']
    Parch = request.form['Parch']
    Fare = request.form['Fare']
    male = request.form['male']
    Q = request.form['Q']
    S = request.form['S']

    prediction = loaded_model.predict([[Pclass,Age,SibSp,Parch,Fare,male,Q,S]])
    probability = loaded_model.predict_proba([[Pclass,Age,SibSp,Parch,Fare,male,Q,S]])
    probability = f"{np.round((np.max(probability) * 100), 2)}%"
    prediction = ""
    probability = f"{probability}%"
    
    if prediction == 0:
        prediction = "Not Died"
    else:
        prediction = "Died"

    print(prediction, probability)

    return render_template("form.html", output_prediction=prediction, output_proba=probability)


# Main function
if __name__ == '__main__':
    app.run(debug=True)