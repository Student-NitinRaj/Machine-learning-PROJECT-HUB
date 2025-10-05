from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

model=pickle.load(open('model.pkl','rb'))

# flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form['features']
    features_list = features.split(',')
    np_features = np.array(features_list, dtype=np.float32)
    pred= model.predict(np_features.reshape(1, -1))
    output= ["cancrous" if pred[0]==1 else "non-cancrous"]
    return render_template('index.html', message= output)
    
# python main
if __name__ == '__main__':
    app.run(debug=True)