import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class lstm_net(nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = nn.Embedding(65536, 512)
    self.lstm1 = nn.LSTM(512, 256, batch_first=True)
    self.lstm2 = nn.LSTM(256, 128, batch_first=True)
    self.lin1 = nn.Linear(128*4, 128)
    self.lin2 = nn.Linear(128, 6)

  def forward(self, x):
    out = self.emb(x)
    out, _ = self.lstm1(out)
    out = F.relu(out)
    out, _ = self.lstm2(out)
    out = F.relu(out)
    out = F.relu(self.lin1(out.reshape(-1, 128*4)))
    out = torch.log_softmax(self.lin2(out), dim=1)
    return out

class linear_net(nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = nn.Embedding(65536, 512)
    self.lin1 = nn.Linear(512, 256)
    self.lin2 = nn.Linear(256, 128)
    self.lin3 = nn.Linear(128, 64)
    self.lin4 = nn.Linear(64, 32)
    self.lin5 = nn.Linear(32, 16)
    self.lin6 = nn.Linear(16*4, 6)

  def forward(self, x):
    out = self.emb(x)
    out = F.relu(self.lin1(out))
    out = F.relu(self.lin2(out))
    out = F.relu(self.lin3(out))
    out = F.relu(self.lin4(out))
    out = F.relu(self.lin5(out))
    out = torch.log_softmax(self.lin6(out.reshape(-1, 16*4)), dim=1)
    return out

class cnn_net(nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = nn.Embedding(65536, 512)
    self.conv1 = nn.Conv1d(4, 3, 2)
    self.conv2 = nn.Conv1d(3, 2, 2)
    self.conv3 = nn.Conv1d(2, 1, 2)
    self.drop = nn.Dropout()
    self.pool = nn.MaxPool1d(2)
    self.lin = nn.Linear(63, 6)

  def forward(self, x):
    out = self.emb(x)
    out = F.relu(self.pool(self.drop(self.conv1(out))))
    out = F.relu(self.pool(self.drop(self.conv2(out))))
    out = F.relu(self.pool(self.drop(self.conv3(out))))
    out = torch.log_softmax(self.lin(out.reshape(-1, 63)), dim=1)
    return out
