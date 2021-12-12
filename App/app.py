from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pickle

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def get_model():
    with open('Models/pipe.pkl', 'rb') as f:
        return pickle.load(f)

@app.get('/')
def hello_world():
    return {"message": "Hello World"}

@app.post('/predict')
async def predict(request: Request):
    data = await request.json()

    # return {"dados": data.get('bedrooms')}
    model = get_model()
    array = [data.get('bedrooms'), data.get('bathrooms'), data.get('sqft_living'),
             data.get('sqft_lot'), data.get('floors'), data.get('waterfront'),
             data.get('view'),
             data.get('condition'), data.get('grade'), data.get('sqft_above'),
             data.get('sqft_basement'),
             data.get('yr_built'), data.get('yr_renovated'),
             data.get('sqft_living15'), data.get('sqft_lot15')]
    pred = model.predict([array])

    return {"Price": pred[0]}
