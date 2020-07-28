import argparse
from flask import Flask, jsonify, request
from flask import render_template, send_from_directory
import os
import re
import joblib
import socket
import json
import numpy as np
import pandas as pd


## import model specific functions and variables
from model import train, loadModel, predict
from model import MODEL_VERSION, MODEL_VERSION_NOTE

app = Flask(__name__)

@app.route('/predict', methods=['GET','POST'])
def predict():
    """
    predict API
    """
    
    ## mandatory validations
    if not request.json:
        print("ERROR: No request data received")
        return jsonify([])

    if 'query' not in request.json:
        print("ERROR: Missing 'query' attribute")
        return jsonify([])

    ## assign default value if not populated
    if 'type' not in request.json:
        query_type = 'numpy'

       
    ## load model
    model = loadModel()
    
    if not model:
        print("ERROR: model not loaded")
        return jsonify([])

    ## process the query
    query = request.json['query']
    _result = predict(query, model, test=test)
    result = {}
    
    for key,item in _result.items():
        if isinstance(item,np.ndarray):
            result[key] = item.tolist()
        else:
            result[key] = item
    
    return(jsonify(result))

@app.route('/train', methods=['GET','POST'])
def trainAPI():
    """
    train API
    """
    
    ## mandatory validations
    if not request.json:
        print("ERROR: No request data received")
        return jsonify(False)

    ## set the test flag
    test = False
    if 'mode' in request.json and request.json['mode'] == 'test':
        test = True

    print("... model training begin")
    train(test=test)
    print("... model training done")

    return(jsonify(True))
        
if __name__ == '__main__':

    ## parse arguments for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = vars(ap.parse_args())

    if args["debug"]:
        app.run(debug=True, port=8080)
    else:
        app.run(host='0.0.0.0', threaded=True ,port=8080)

