import numpy as np
from flask import Flask, request, jsonify, render_template 
import pickle

app = Flask(__name__)
model = pickle.load(open('taxi.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

app.run(debug=True)