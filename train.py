import torch
from torch import nn
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt

from model import SentimentPLModel
from datautils import SentimentPLDataset


if __name__ == '__main__':
    batch_size = 64
    train_files = [#'sentiment_data/all.text.train.txt',
                   'data/all.sentence.train.txt']
    test_files = [#'sentiment_data/all.text.test.txt',
                  'data/all.sentence.test.txt']

    train_loader = torch.utils.data.DataLoader(SentimentPLDataset(train_files), batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(SentimentPLDataset(test_files), batch_size)

    model = SentimentPLModel().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters())

    train_losses = []
    valid_losses = []

    pbar = trange(40)

    for epoch in pbar:
        # training
        total_loss = 0
        model.train()
        for i, (x, y) in enumerate(train_loader, 0):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / (i + 1))

        # validation
        total_loss = 0
        model.eval()
        for i, (x, y) in enumerate(test_loader):
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            total_loss += loss.item()
        valid_losses.append(total_loss / (i + 1))

        pbar.set_description(f'loss : {train_losses[-1]:.6f}, val_loss: {valid_losses[-1]:.6f}')

    print('Finished Training')

    plt.plot(train_losses)
    plt.plot(valid_losses)
    plt.legend(['train loss', 'valid loss'])
    plt.show()

    model.cpu()
    torch.save(model, 'trained_models/sentimentPL_allegro-hubert_clarin-PL.pth')
