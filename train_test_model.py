import models
import models_binary
from tqdm import tqdm
import torch
import torch.optim as optim

def train_network(model, model_name, loss_fn, n_epochs, dataloader, device):
  model = model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  model.train()
  losses, accuracies, n_epochs = list(), list(), n_epochs
  for epoch in range(n_epochs):
    for x, y in (loop := tqdm(dataloader, leave=False)):
      x = x.to(device)
      one_hot = torch.zeros(y.size(0), 2, dtype=torch.long)
      one_hot[range(y.size(0)), y.numpy()] = 1
      model.zero_grad()
      out = model(x)
      cat = out.argmax(dim=1).cpu()
      acc = (cat == y).float().mean().item()
      loss = loss_fn(out.cpu(), one_hot.float())
      loss.backward()
      optimizer.step()
      loop.set_description(f"Epochs: {epoch+1}/{n_epochs}")
      loop.set_postfix(loss=loss.item(), acc=acc)
  return model

def train_mul_net(model, model_name, loss_fn, n_epochs, dataloader, device):
  model = model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  model.train()
  losses, accuracies, n_epochs = list(), list(), n_epochs
  for epoch in range(n_epochs):
    for x, y in (loop := tqdm(dataloader, leave=False)):
      x = x.to(device)
      model.zero_grad()
      out = model(x)
      cat = out.argmax(dim=1).cpu()
      acc = (cat == y).float().mean().item()
      loss = loss_fn(out.cpu(), y.long())
      loss.backward()
      optimizer.step()
      loop.set_description(f"Epochs: {epoch+1}/{n_epochs}")
      loop.set_postfix(loss=loss.item(), acc=acc)
  return model

def test_network(model, model_name, dataloader, device, len_dataset):
  model.eval()
  correct = 0
  with torch.no_grad():
    for x, y in tqdm(dataloader, leave=False):
      x = x.to(device)
      out = model(x)
      cat = out.argmax(dim=1).to(device)
      correct += (cat == y.to(device)).float().sum().item()
  print(f'{model_name} accuracy: {correct/len_dataset} \n')
