import torch
from torch import nn
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt

from sentimentpl.models import SentimentPLModel
from sentimentpl.datautils import SentimentPLDataset


if __name__ == '__main__':
    batch_size = 32
    train_files = [#'sentiment_data/all.text.train.txt',
                   'data/all.sentence.train.txt']
    test_files = [#'sentiment_data/all.text.test.txt',
                  'data/all.sentence.test.txt']

    train_loader = torch.utils.data.DataLoader(SentimentPLDataset(train_files), batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(SentimentPLDataset(test_files), batch_size)

    model = SentimentPLModel().cuda()#tokenizer='trained_models/politicalBERT', embed_model='trained_models/politicalBERT').cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters())
    n_epochs = 30
    train_embedding = [False]#*4 + [True]

    train_losses = []
    valid_losses = []

    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        print('---------------------------------------------------------')
        tune_embedding = train_embedding[min(epoch, len(train_embedding)-1)]
        print(f'epoch {epoch+1}/{n_epochs}\n  Tune embedding: {tune_embedding}')
        # training
        total_loss = 0
        model.train()
        for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            y = y.cuda()
            optimizer.zero_grad()
            pred = model(x, tune_embedding=tune_embedding)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / (i + 1))

        # validation
        total_loss = 0
        model.eval()
        for i, (x, y) in enumerate(test_loader):
            y = y.cuda()
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            total_loss += loss.item()
        valid_losses.append(total_loss / (i + 1))

        print(f'\tloss : {train_losses[-1]:.6f}, val_loss: {valid_losses[-1]:.6f}\n')

        # model saving
        if valid_losses[-1] <= best_val_loss:
            best_val_loss = valid_losses[-1]
            model.save()
            print('model saved!\n')

    print('Finished Training')

    plt.plot(train_losses)
    plt.plot(valid_losses)
    plt.legend(['train loss', 'valid loss'])
    plt.show()
