import torch
from torch import nn

from src.rnn.rnn import RNN


def train(self, model, epochs, x, y):
        model = RNN()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model.forward(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        return model
