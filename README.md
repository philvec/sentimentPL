# sentimentPL
PyTorch models for polish language sentiment regression based on allegro/herbert and CLARIN-PL dataset

[![PyPI - License](https://img.shields.io/pypi/l/sentimentpl)](https://www.gnu.org/licenses/gpl-3.0)
[![PyPI](https://img.shields.io/pypi/v/sentimentpl)](https://pypi.org/project/sentimentpl/)
[![GitHub Repo stars](https://img.shields.io/github/stars/philvec/sentimentpl)](https://github.com/philvec/sentimentPL)
[![GitHub last commit](https://img.shields.io/github/last-commit/philvec/sentimentpl)](https://github.com/philvec/sentimentPL)

### Installation
sentimentPL is available on PyPI, so You can just run:
```
$ pip3 install sentimentpl
```

### Basic Usage
For a given sentence, the model produces output value from (-1;1) range (from most negative to most positive).
```python
from sentimentpl.models import SentimentPLModel

model = SentimentPLModel(from_pretrained='latest')
print(model('Jestem wesoły Romek').item())
```

**Note:** *The model uses transformers API to load pretrained embedding models from their repository. 
They should be downloaded and cached on Your machine.*

**Note:** *The model loads pretrained state dicts for final regression layers from a file included in the package files 
(as its size does not exceed 1MB). This will be changed in the future, so the model would be loaded from 
external repository.*

### Training
For training You would probably want to download the source code by cloning the repository:
```
$ git clone https://github.com/philvec/sentimentPL.git
```
Download training data from <br>
https://clarin-pl.eu/dspace/bitstream/handle/11321/710/dataset_conll.zip <br>
and unzip it to *sentimentpl/data*. <br><br>
In the main repository dir, run
```
$ python3 ./sentimentpl/train.py
```

### Version history

#### v.0.0.5 latest
Basic 3-layer MLP with ReLU and input Dropout.

### References:
- Kocoń, Jan; Zaśko-Zielińska, Monika and Miłkowski, Piotr, 2019, PolEmo 2.0 Sentiment Analysis Dataset for CoNLL, CLARIN-PL digital repository, http://hdl.handle.net/11321/710.
- T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi,P. Cistac, T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer,P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. L. Scao,S. Gugger, M. Drame, Q. Lhoest, and A. M. Rush, “Transformers:State-of-the-art natural language processing,” inProceedings of the2020 Conference on Empirical Methods in Natural LanguageProcessing: System Demonstrations, (Online), pp. 38–45, Associationfor Computational Linguistics, Oct. 2020.
- P. Rybak, R. Mroczkowski, J. Tracz, and I. Gawlik, “Klej:Comprehensive benchmark for polish language understanding,” 2020
