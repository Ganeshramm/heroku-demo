# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 23:54:15 2020

@author: Gowtham Muruganandam
"""

import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

model = pickle.load(open("model1.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    feat = [int(x) for x in request.form.values()]
    final = [np.array(feat)]
    pred = model.predict(final)
    
    output = round(pred[0],2)
    return render_template('index.html', prediction_text = 'Employee salary should be $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug = True)