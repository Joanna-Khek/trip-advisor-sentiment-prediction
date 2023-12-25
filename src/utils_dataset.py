import string
import pandas as pd
from pathlib import Path
import random
from datasets import load_dataset
import torch
import spacy

class TripAdvisorData:
    """Reads and preprocess data"""
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def read_data(self, data_dir: Path, data_name: str):
        data_path = str(Path(data_dir, data_name))
        df = load_dataset("csv", data_files=data_path)
        df['train'].set_format(type='pandas')
        return df['train'][:]
    
    def create_splits(self, 
                      data: pd.DataFrame,
                      target_var: str,
                      train_prop=0.7, 
                      valid_prop=0.2, 
                      test_prop=0.1):
        train_idx = []
        valid_idx = []
        test_idx = []

        for label in data[f"{target_var}"].unique():
            # filter only the label
            df_class = data.query(f"{target_var} == {label}")
            list_of_indices = list(df_class.index)
            total_indices = len(list_of_indices)

            # randomly shuffle the list of indices
            random.shuffle(list_of_indices)

            # get the number of images needed
            num_train = int(total_indices * train_prop)
            num_valid = int(total_indices * valid_prop)
            
            # split
            train_idx.extend(list_of_indices[:num_train])
            valid_idx.extend(list_of_indices[num_train: num_train+num_valid])
            test_idx.extend(list_of_indices[num_train+num_valid:])

        train_data = data.iloc[train_idx]
        valid_data = data.iloc[valid_idx]
        test_data = data.iloc[test_idx]

        return train_data, valid_data, test_data
    
    def _remove_punct(self, doc):
        return [t for t in doc if t.text not in string.punctuation]
    
    def _remove_stopwords(self, doc):
        return [t for t in doc if not t.is_stop]
    
    def _lemmatize(self, doc):
        return ' '.join([t.lemma_ for t in doc])

    def _remove_numbers(self, doc):
        return [t for t in doc if not (t.like_num or t.is_currency)]
    
    def clean_text(self, text):
        clean_text = text.lower().strip()
        doc = self.nlp(clean_text)
        clean_text = self._remove_punct(doc)
        #clean_text = self._remove_stopwords(clean_text)
        clean_text = self._remove_numbers(clean_text)
        clean_text = self._lemmatize(clean_text)
        return clean_text
    
    def label_encoder(self, label):
        # Exclude rating = 3 as it is ambiguous
        # whether it is positive or negative
        if label == 1 or label == 2:
            return 0 # Negative
        
        elif label == 4 or label == 5:
            return 1 # Positive
        
        else:
            return 'Unclassified'
        
    def preprocess(self, data, feature_var):
    
        data = (data
                .assign(num_words=lambda df_: df_[f'{feature_var}'].str.split().apply(len),
                        label=lambda df_: df_.Rating.map(self.label_encoder),
                        clean_review=lambda df_: df_[f'{feature_var}'].apply(self.clean_text))
                .query("label != 'Unclassified'")
                .reset_index()
                .drop("index", axis=1)
                )

        return data
    

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
