from transformers import pipeline, set_seed
from transformers.pipelines import TextClassificationPipeline
from transformers import *


class SentimentPredictor:
    def __init__(self):
        self.generator: TextClassificationPipeline
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained("./finetuned_models/xlm-roberta-base-sst-2-sh-sentiment")
        self.model = XLMRobertaForSequenceClassification.from_pretrained("./finetuned_models/xlm-roberta-base-sst-2-sh-sentiment")   

    def load_generator(self) -> None:
        self.generator = pipeline('sentiment-analysis', model= self.model, tokenizer = self.tokenizer)

    def predict_sentiment(self, sequence: str)->dict:
        result = self.generator(sequence)
        sentiment = {'LABEL_0': "Negative", "LABEL_1": "Positive"}
        print(result)
        result = {"label": sentiment[result[0]['label']], "score": round(result[0]['score'], 4)}
        return result
