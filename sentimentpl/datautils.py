from transformers import XLMTokenizer, RobertaModel
from tqdm.auto import tqdm, trange
import torch



mapping = {'plus': 1., 'minus': -1., 'zero': 0., 'amb': 0.}

def load_embed(filenames):
    lines = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            lines += [line for line in f]
    X, Y = [], []
    for line in lines:
        if '__label__' in line:
            text, sentiment = line.split('__label__')
            labels = [torch.tensor(v).unsqueeze(-1).float() for k, v in mapping.items() if k in sentiment][0]
            X.append(text)
            Y.append(labels)
    return X, Y


class SentimentPLDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.X, self.y = load_embed(filename)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
