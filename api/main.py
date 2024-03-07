from fastapi import FastAPI, Request,Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import json
import pickle
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

with open(r'C:\Users\Dell\House_prediction\columns.json','r')as file:
    columns=json.load(file)
    locations=sorted(columns['data_columns'][3:])

model=pickle.load(open(r"C:\Users\Dell\House_prediction\Bengalore_house_prices_model.pickle",'rb'))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("homepage.html", {"request": request,'locations':locations})

columns=columns['data_columns']
@app.post('/predict/')
def predict(location:str=Form(...),bhk:int=Form(...),bath:int=Form(...),sqft:float=Form(...)):
    location=location
    bhk=bhk
    bath=bath
    sqft=sqft
    x=[0]*len(columns)
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if location in columns:
        x[columns.index(location)]=1
    
    return round(model.predict([x])[0],2)


if __name__=='__main__':
    uvicorn.run('main:app',host='localhost',port=8000)