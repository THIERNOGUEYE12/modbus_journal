import argparse
import data_loading
from torch.utils.data import DataLoader
import models, models_binary, train_test_model
import torch.nn as nn
import torch

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch', help='Number of Epochs', type=int, default=10)
  parser.add_argument('--train', help='Train?', type=int, default=1)
  parser.add_argument('--device', help='cuda or cpu', type=str, default='cpu')
  args = parser.parse_args()
  print('Loading the Dataset')
  x_train_bin, y_train_bin, x_test_bin, y_test_bin = data_loading.load_dataset('binary')
  x_train_mul, y_train_mul, x_test_mul, y_test_mul = data_loading.load_dataset('multi')
  train_dataloader_bin = DataLoader(list(zip(x_train_bin, y_train_bin)), batch_size=64, shuffle=True)
  print('.')
  test_dataloader_bin = DataLoader(list(zip(x_test_bin, y_test_bin)), batch_size=64, shuffle=False)
  print('..')
  train_dataloader_mul = DataLoader(list(zip(x_train_mul, y_train_mul)), batch_size=64, shuffle=True)
  print('...')
  test_dataloader_mul = DataLoader(list(zip(x_test_mul, y_test_mul)), batch_size=64, shuffle=False)
  print('Dataset Successfully Loaded')
  print()
  if args.device == 'cuda':
    print('Device Selected: GPU')
  else:
    print('Device Selected: CPU')
  print()
  if args.train:
    print('Training Binary Classification', '\n')
    loss_fn = nn.BCELoss()
    for model, name in [(models_binary.linear_net(), 'Linear Model'), (models_binary.lstm_net(), 'LSTM Model'), (models_binary.cnn_net(), 'CNN Model')]:
      print(f'Training {name}')
      model = model.to(args.device)
      model = train_test_model.train_network(model, name, loss_fn, args.epoch, train_dataloader_bin, args.device)
      print(f'Testing {name}')
      train_test_model.test_network(model, name, test_dataloader_bin, args.device, len(y_test_bin))

    print('Training Multi-Class Classification', '\n')
    loss_fn = nn.NLLLoss()
    for model, name in [(models.linear_net(), 'Linear Model'), (models.lstm_net(), 'LSTM Model'), (models.cnn_net(), 'CNN Model')]:
      print(f'Training {name}')
      model = model.to(args.device)
      model = train_test_model.train_mul_net(model, name, loss_fn, args.epoch, train_dataloader_mul, args.device)
      print(f'Testing {name}')
      train_test_model.test_network(model, name, test_dataloader_mul, args.device, len(y_test_mul))
  else:
    print('Testing on Pre-Trained Models', '\n')
    names = {'lin_model_bin': 'Linear Model', 'lstm_model_bin': "LSTM Model", 'cnn_model_bin': "CNN Model"}
    print('Testing Binary Classification Models \n ************************************')
    for model, name in [(models_binary.linear_net(), 'lin_model_bin'), (models_binary.lstm_net(), 'lstm_model_bin'), (models_binary.cnn_net(), 'cnn_model_bin')]:
      model = model.to(args.device)
      model.load_state_dict(torch.load(f"saved_models/{name}.pt"))
      train_test_model.test_network(model, names[name], test_dataloader_bin, args.device, len(y_test_bin))
    names = {'lin_model': 'Linear Model', 'lstm_model': "LSTM Model", 'cnn_model': "CNN Model"}
    print('Testing Multi-Class Classification Models \n ************************************')
    for model, name in [(models.linear_net(), 'lin_model'), (models.lstm_net(), 'lstm_model'), (models.cnn_net(), 'cnn_model')]:
      model = model.to(args.device)
      model.load_state_dict(torch.load(f'saved_models/{name}.pt'))
      train_test_model.test_network(model, names[name], test_dataloader_mul, args.device, len(y_test_mul))
