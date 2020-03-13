from flask import Flask, jsonify
from flask import request
import pandas as pd
from sklearn.externals import joblib
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
     json_ = request.json
     prediction = model.predict(json_)
     if prediction == 1:
         prediction = "Hurray! Game will be played"
     else: prediction = "No Game Today!"
     return jsonify({'prediction': str(prediction)})
if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    model = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')

    app.run(port=port, debug=True)
