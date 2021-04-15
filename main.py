from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ner_predictor import *
from sentiment_predictor import *


ner = NERPredictor()
ner.load_generator()
nlp = SentimentPredictor()
nlp.load_generator()


class Hansard_Predictor(BaseModel):
    hansard_text: str
    output: dict = None


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Singapore Parliament Hansard NLP": "API"}


@app.post("/ner/")
async def ner_inference(hansard: Hansard_Predictor):
    hansard.output = ner.generate_entities(hansard.hansard_text)
    return {"output": hansard.output}


@app.post("/sentiment/")
async def ner_inference(hansard: Hansard_Predictor):
    hansard.output = nlp.predict_sentiment(hansard.hansard_text)
    return {"output": hansard.output}
