import pickle
import uvicorn
from fastapi.exceptions import HTTPException 
from fastapi.responses import JSONResponse   
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi import FastAPI

count = 0

app = FastAPI(title="Decision_Tree_Model")

# Load the trained model from the pickle file
with open('credit_card_fraud_detection.pkl', 'rb') as f:
    dv = pickle.load(f)
    rf = pickle.load(f) 

class ClientData(BaseModel):
        V1: float 
        V2: float 
        V3: float 
        V4: float 
        V5: float 
        V6: float 
        V7: float 
        V8: float 
        V9: float 
        V10: float 
        V11: float 
        V12: float 
        V13: float 
        V14: float 
        V15: float 
        V16: float 
        V17: float 
        V18: float 
        V19: float 
        V20: float 
        V21: float 
        V22: float 
        V23: float 
        V24: float 
        V25: float 
        V26: float 
        V27: float 
        V28: float 
        Amount: float

@app.post("/score")
async def score(client: ClientData): 
    payload = client.model_dump() # or client.dict() for pydantic v1 
    X = dv.transform([payload]) 
    pred = rf.predict(X)[0] 
    return {"prediction": int(pred)}

    # prediction = client.dict()
    # X = dv.transform([client.dict()])
    # pred = dtc.predict(X)[0]
    # return {"prediction": int(pred)}

@app.get("/")
async def root():
    return{"message": "Welcome to the Decision Tree Model API. Use the /score endpoint to get predictions."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
 