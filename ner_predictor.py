from transformers import pipeline, set_seed
from transformers.pipelines import TokenClassificationPipeline
from transformers import *


class NERPredictor:
    def __init__(self):
        self.generator: TokenClassificationPipeline
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained("./finetuned_models/xlm-roberta-base-ontonotes5-sh-ner")
        self.model = XLMRobertaForTokenClassification.from_pretrained("./finetuned_models/xlm-roberta-base-ontonotes5-sh-ner")   

    def load_generator(self) -> None:
        self.generator = pipeline('ner', model= self.model, tokenizer = self.tokenizer, grouped_entities=True)

    def generate_entities(self, sequence: str)->dict:
        entity_list = self.generator(sequence)
        for idx in entity_list:
            idx['label'] = idx.pop('entity_group')
            idx['start'] = int(idx['start'])
            idx['end'] = int(idx['end'])
            idx['score'] = float(idx['score'])
        doc = {"text": sequence, "ents": entity_list}
        return doc
