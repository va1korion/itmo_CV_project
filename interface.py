import json
from typing import Union
from time import perf_counter

import psutil
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect, UploadFile, Form, File
from fastapi.exceptions import RequestValidationError
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import cv2
import os
from random import shuffle
from fastapi import FastAPI
from pydantic import BaseModel


class Image(BaseModel):
    name: str


print(tf.test.gpu_device_name())
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = load_model('./my_model')
print(model.summary())
with open('my_model/labels.json') as f:
    labels = json.load(f)

SIZE = (180, 180)
COLORSPACE = 1
if COLORSPACE == 1:
    MODE = cv2.COLOR_BGR2GRAY
else:
    MODE = cv2.COLOR_BGR2RGB


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    print("Request: index")
    global err

    context = {
        'request': request
    }
    err = None
    return templates.TemplateResponse("index.html", context)

"""
@app.post("/card", response_class=HTMLResponse)
async def process(my_file: UploadFile = File(...)):
    print("Request: card")
    global err
    pic = np.array(my_file)
    print(pic)

    pic = cv2.imdecode(pic, cv2.IMREAD_COLOR)
    pic = cv2.cvtColor(pic, MODE)
    img = cv2.resize(pic, SIZE)
    s = img.reshape((img.shape[0], img.shape[1], COLORSPACE))

    ans = model.predict(s)

    return json.dumps(pic[1])
"""


@app.post("/card")
async def create_upload_file(file: Union[UploadFile, None] = None):
    print("Request: card")
    global err
    start = perf_counter()

    arr = np.asarray(bytearray(file.file.read()), dtype=np.uint8)
    pic = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    pic = cv2.cvtColor(pic, MODE)
    img = cv2.resize(pic, SIZE)
    s = img.reshape((img.shape[0], img.shape[1], COLORSPACE))
    sample = np.array([s,])
    ans = model.predict(sample).argmax()
    print(ans)
    end = perf_counter()
    print("frame time: " + str(end - start))
    print("mem: " + str(psutil.Process().memory_info().rss / (1024 * 1024)))
    return {"result": labels[str(ans)]}
