import pandas as pd
playing_path ='C:/Users/embibe/Documents/MyGit/PCode/Flask_API/Playing.csv'
playing_data = pd.read_csv(playing_path)
# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
temperature_en=le.fit_transform(playing_data['Temperature'])
wind_en=le.fit_transform(playing_data['Wind'])
sunshine_en=le.fit_transform(playing_data['Sunshine'])
play_en=le.fit_transform(playing_data['Play'])
features=list(zip(temperature_en,wind_en,sunshine_en))
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(features,play_en)
from sklearn.externals import joblib
joblib.dump(model, 'model.pkl')
