import requests
url = "http://127.0.0.1:8000/score"
client = {
    "V1":-4.3979744417,
    "V2":1.3583670284,
    "V3":-2.5928442183,
    "V4":2.6797869669,
    "V5":-1.1281309421,
    "V6":-1.7065363877,
    "V7":-3.496197293,
    "V8":-0.248777743,
    "V9":-0.2477678995,
    "V10":-4.801637406,
    "V11":4.8958442235,
    "V12":-10.9128193194,
    "V13":0.1843716858,
    "V14":-6.7710967247,
    "V15":-0.0073261826,
    "V16":-7.3580832213,
    "V17":-12.5984185406,
    "V18":-5.1315486284,
    "V19":0.3083339458,
    "V20":-0.1716078786,
    "V21":0.5735740684,
    "V22":0.176967718,
    "V23":-0.4362068836,
    "V24":-0.0535018649,
    "V25":0.252405262,
    "V26":-0.6574877548,
    "V27":-0.8271357146,
    "V28":0.84957338,
    "Amount":-0.1173423075
}
resp = requests.post(url, json=client) 
resp.raise_for_status() 
data = resp.json() 
prediction = data.get("prediction")  # get the value of "prediction" or default to 0 if it doesn't exist
if prediction == 0:
    print(f"Predicted Class: not fraud")
else:
    print(f"Predicted Class: fraud")
print(f"Predicted probability: {data}")