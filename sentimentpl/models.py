from contextlib import ExitStack
import importlib
import torch
from torch import nn
from transformers import XLMTokenizer, RobertaModel

import io

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from sentimentpl import trained_models


class SentimentPLModel(nn.Module):
    def __init__(self,
                 from_pretrained=None,
                 tokenizer="allegro/herbert-klej-cased-tokenizer-v1",
                 embed_model="allegro/herbert-klej-cased-v1"):
        super().__init__()

        self.tokenizer = XLMTokenizer.from_pretrained(tokenizer)
        self.embed_model = RobertaModel.from_pretrained(embed_model, return_dict=True)

        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(768, 256), nn.LeakyReLU(),
                                nn.Linear(256, 16), nn.LeakyReLU(),
                                nn.Linear(16, 1), nn.Tanh())

        if from_pretrained is not None:
            f = io.BytesIO(importlib.resources.read_binary(trained_models, f'{from_pretrained}.pth'))
            self.fc.load_state_dict(torch.load(f))
            self.eval()

    def save(self, name='latest'):
        self.fc.to('cpu')
        torch.save(self.fc.state_dict(), f'./trained_models/{name}.pth')
        self.fc.to(next(self.embed_model.parameters()).device)

    def forward(self, x, tune_embedding=False):
        encoded = self.tokenizer(x, return_tensors='pt', padding=True)
        encoded = {k: v.to(next(self.parameters()).device) for k, v in encoded.items()}

        with ExitStack() as stack:
            if not tune_embedding:
                stack.enter_context(torch.no_grad())
            embeddings = self.embed_model(**encoded)['pooler_output'].float()

        output = embeddings
        for fc_i in self.fc:
            output = fc_i(output)
            # print(output)
        return output
