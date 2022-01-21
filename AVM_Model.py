import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from scipy.stats import norm
import xgboost

df = pd.read_csv(r'C:\Users\Yogesh.Tiwari\PycharmProjects\flaskTest\venv\SHS.csv')

################################ project name handling #######################################

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
le = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
df['encoded_project_name']= le.fit_transform(df['Project_Name'])

le_project_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

def get_proj(list1):
  ss = list1
  for k,v in le_project_mapping.items():
    if ss[-1] in str(f"{k}"):
      return le_project_mapping[k]

################################ project name handling #######################################

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
df['encoded_location']= label_encoder.fit_transform(df['Location'])

le_location_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

def get_loc(list1):
  ss = list1
  for k,v in le_location_mapping.items():
    if ss[4] in str(f"{k}"):
      return le_location_mapping[k]

#############################################################################################

ss = []
for i in df['Area']:
  try:
    ss.append(float(i))
  except:
    ss.append(float(1789))

df['Area']=ss

############outlier
maxy=df['Area'].quantile(0.99)

miny=df['Area'].quantile(0.01)

df2=df[(df.Area<maxy)&(df.Area>miny)]

df2 = df2[['Area','BHK','Covered_Parking','Open_Parking','encoded_location','encoded_project_name','final_price']]

print(df2.shape)

#################### Train Test split (80:20) ###########################
dff_Train = df2.iloc[:4095,:]
dff_Test = df2.iloc[4095:,:]
################# From Train dataset we create x_train and y_train #####################
X_train=dff_Train.drop(['final_price'],axis=1)
y_train=dff_Train['final_price']
print(y_train.shape)
dff_Test.drop(['final_price'],axis=1,inplace=True)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=0.33, random_state=7)

import xgboost
regressor = xgboost.XGBRegressor(max_depth=5, min_child_weight=2, n_estimators=500)
regressor.fit(X_train1,y_train1)

import joblib
import pickle
import xgboost as xgb
file_name = "xgb_reg.pkl"
#save model
joblib.dump(regressor, file_name)

#load saved model
xgb = joblib.load(file_name)
