# Authors: CS-World Domination Summer19 - JG
from joblib import dump, load
import numpy as np

def predict(mystr):
    # Format string input to list of floats
    mystr = mystr.replace(" ", "")
    print(mystr)
    numList = []
    currnum = ""
    for s in mystr:
        if s != ",":
            currnum+=s
        else:
            numList.append(float(currnum))
            currnum = ""
    numList.append(float(currnum))
    
    # Make list numpy array
    numList = np.array(numList)
    numList = numList.reshape(1, -1)
    # Load the scaler and scale the input
    if False:
        scaler = load('RegressionScaler.sav')
        numList[:4] = scaler.transform(numList[:4])
    # Load the model and make a prediction
    model = load('RegressionModel.sav')
    prediction = model.predict(numList)
    return str(round(prediction[0], 1))