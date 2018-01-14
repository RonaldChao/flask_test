#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 17:34:40 2018

@author: ronaldchao
"""

from flask import Flask, jsonify, request
import numpy as np

app = Flask(__name__)

# Load stored model
from sklearn.externals import joblib
model = joblib.load('./model.pkl')

@app.route('/', methods=["GET"])
def hello_world():
    return "Hello World!"

@app.route('/predict', methods=["POST"])
def predict():
    # extract data from POST request
    X_in = request.data
    X_in_str = X_in.decode("utf-8")
    X_in_array = X_in_str.split(',')
    X_in = [[float(i) for i in X_in_array]]
    
    # run classification
    result = model.predict(X_in)
    
    # send response
    if(result.item() == 1):
        return "Let's do further testing"
    else:
        return "You are good"

if __name__ == "__main__":
        app.run()