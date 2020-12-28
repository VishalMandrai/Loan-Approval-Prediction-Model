import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model_gb = pickle.load(open('model_gb.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index_main.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]    
    final_features = [np.array(int_features)]
    temp = scaler.transform(final_features)
    prediction = model_gb.predict(temp)

    if prediction == 1:
            return render_template('index_main.html', prediction_text='CONGRATS! Your Loan Will be Approved!')
    else:
        return render_template('index_main.html', prediction_text='Oops! Sorry Your Loan Will not be Approved!')



if __name__ == "__main__":
    app.run(debug=True)
