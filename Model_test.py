import AVM_Model
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import joblib
import pickle
import xgboost as xgb

# file_name = r"D:\python folder\PycharmProjects\pythonProject2\AVM_new_Model.pkl"
#
# #load saved model
# regressor = joblib.load(r"D:\python folder\PycharmProjects\pythonProject2\xgb_reg.pkl")
#
#
# print("LOAD ZHALA")
# # app=Flask(__name__)
#
# search = [2000,3,1,0,"alandi","sun"]
# lc=AVM_Model.get_loc(search)
# pr=AVM_Model.get_proj(search)
#
# test1=[[search[0],search[1],search[2],search[3],lc,pr]]
# test2= pd.DataFrame(test1,columns=
# ['Area','BHK','Covered_Parking','Open_Parking','encoded_location','encoded_project_name'],dtype=float)
#
# y = regressor.predict(test2)
# print("Our prediction",y)
# print("Range", round(float(y*0.85)), " - ", round(float(y*1.15)))
#
# print("ZHALAAAAAAAAAAAAAAa")

app = Flask(__name__)
regressor = joblib.load(r"C:\Users\Yogesh.Tiwari\PycharmProjects\flaskTest\venv\xgb_reg.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = [x for x in request.form.values()]
    print(features)
    lc=AVM_Model.get_loc(features)
    pr=AVM_Model.get_proj(features)
    test1=[features[0],features[1],features[2],features[3],lc,pr]
    test2= pd.DataFrame([test1],columns=
    ['Area','BHK','Covered_Parking','Open_Parking','encoded_location','encoded_project_name'],dtype=float)
    prediction = regressor.predict(test2)
    print(prediction)
    range_val=f"{round(float(prediction)*.85)} - {round(float(prediction)*1.15)}"
    # output = round(float(prediction))
    print("Outputtttt",range_val)
    # return render_template('index.html', prediction_text='House price should be Rs {}'.format(range_val))
    return(range_val)
if __name__ == "__main__":
    app.run(debug=True)
