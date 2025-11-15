import requests
url = "http://127.0.0.1:8000/score"
client = {
    "V1": -1.359807134,
    "V2": -0.072781173,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462387,
    "V7": 0.239599,    
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": 	-0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount":0.244964
}
resp = requests.post(url, json=client) 
resp.raise_for_status() 
data = resp.json() 
prediction = data.get("prediction", 0)  # get the value of "prediction" or default to 0 if it doesn't exist
if prediction == 0:
    print(f"Predicted Class: not fraud")
else:
    print(f"Predicted Class: fraud")
print(f"Predicted probability: {data}")