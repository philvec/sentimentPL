from transformers import XLMTokenizer, RobertaModel
from tqdm.auto import tqdm, trange
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = XLMTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
embed_model = RobertaModel.from_pretrained("allegro/herbert-klej-cased-v1", return_dict=True).to(device)

mapping = {'plus': 1., 'minus': 0., 'zero': 0.5, 'amb': 0.5}
EMBED_BATCH_SIZE = 16

def embed(texts):
    with torch.no_grad():
        encoded = tokenizer(texts, return_tensors='pt', padding=True)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        embeddings = embed_model(**encoded)['pooler_output'].detach().float()
    return embeddings

def load_embed(filenames):
    lines = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            lines += [line for line in f]
    X, Y = [], []
    for i in trange(0, len(lines), EMBED_BATCH_SIZE):
        texts, sentiments = zip(*[line.split('__label__')
                                  for line in lines[i:i+EMBED_BATCH_SIZE] if ('__label__' in line)])
        embeddings = embed(texts)
        labels = [[torch.tensor(v).unsqueeze(-1).float() for k, v in mapping.items() if k in sentiment][0]
                  for sentiment in sentiments]
        X += embeddings
        Y += labels
    return X, Y


class SentimentPLDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.X, self.y = load_embed(filename)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]