import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    try:
        float_features = [float(x) for x in request.form.values()]
        test_vector = np.reshape(np.asarray(float_features), (1, 7))
        p = np.array(model.predict(test_vector)[0])
        print(p)
        label = ['Robbery', 'Rape', 'Accident', 'Violence', 'Kidnapping', 'Murder']
        if p[0] == 1:
            print("Robbery")
            pred = "Robbery"
        if p[1] == 1:
            print("Rape")
            pred = "Rape"
        if p[2] == 1:
            print("Accident")
            pred = "Accident"
        if p[3] == 1:
            print("Violence")
            pred = "Violence"
        if p[4] == 1:
            print("Kidnapping")
            pred = "Kidnapping"
        if p[5] == 1:
            print("Murder")
            pred = "Murder"
        output = pred

        return render_template('index.html', prediction_text='Crime: {}'.format(output))
    except:
        return render_template('index.html', prediction_text='Error: Please enter valid inputs')


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
