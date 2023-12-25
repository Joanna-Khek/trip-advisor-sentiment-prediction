import torch
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from transformers import TextClassificationPipeline

from src.utils_dataset import TripAdvisorData
from src.utils_helper import load_model, set_up_device
from src.config import settings


tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
optim = torch.optim.Adam(model.parameters(), lr=settings.config['learning_rate'])

load_model(model=model, 
           target_dir=settings.TRAINED_MODEL_DIR, 
           model_name=settings.config['model_filename'])

load_model(model=optim, 
           target_dir=settings.TRAINED_MODEL_DIR, 
           model_name=settings.config['model_optim_filename'])


def make_prediction(input_text: str) -> dict:

    pipe = TextClassificationPipeline(model=model, 
                                  tokenizer=tokenizer, 
                                  top_k=1,
                                  device=set_up_device())


    trip_advisor = TripAdvisorData()
    clean_text = trip_advisor.clean_text(input_text)

    pred_class = pipe(clean_text)[0][0]['label']
    pred_prob = pipe(clean_text)[0][0]['score']

    sentiments = {'LABEL_0': 'Negative', 
                  'LABEL_1': 'Positive'}
    

    result = {"prediction": sentiments[pred_class],
              "probability": pred_prob}
    
    return result


