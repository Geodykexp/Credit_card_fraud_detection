import pickle
from urllib import response
import uvicorn
from fastapi.exceptions import HTTPException    
from fastapi import FastAPI, requests
from pydantic import BaseModel, Field

count = 0

app = FastAPI(title="Decision_Tree_Model")

# Load the trained model from the pickle file
pickle_in = open("credit_card_fraud_detection.pkl", "rb")
model = pickle.load(pickle_in)
pickle_in.close()

@app.get("/")
async def root():
    return{"message": "Welcome to the Daecision Tree Model API. Use the /score endpoint to get predictions."}
# return {"requests.post(url, json=client).json()"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
 