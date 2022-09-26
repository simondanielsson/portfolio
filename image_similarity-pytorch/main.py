import os
import numpy as np
import pandas as pd

from pprint import pprint

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import optimizer

from sklearn.model_selection import train_test_split

from dataset import TripletDataset
from load_data import load_images, load_triplets
from model import TripletNet
from predict import predict
from train import train

########## SETTINGS #############

reload_images = False            # TRUE: newly loading images from .jpg. FALSE: loading images from tensor file.
mode_choice = 'evaluation'             # possible: 'debugging' (5 Triplets), 'exploration' (5000 Triplets), 'evaluation' (All Triplets)

fine_tune = False                # Only train the fully connected layer
number_fc = 2                    # Number of fully connected layers attached to the ResNet

reload_checkpoint = True        # Reload model from a previously saved checkpoint
load_checkpoint_description = 'checkpoint_simon' # for filename, gets overwritten without unique name

save_checkpoints = True          # Save checkpoints to disk for this run
save_checkpoint_description = 'checkpoint_simon_secondtry' # for filename, gets overwritten without unique name

epoch_to_load = 'last_epoch'     # Specify epoch you want to load, can be set to most recent further down in the code
# NOTE: Epochs starting at index 0


if __name__ == "__main__":

  # Initialize hyperparameters
  random_state = 1
  batch_size_train = 128
  batch_size_val = 128

  out_dim = 1024
  num_epochs = 2

  margin = 5.0                            # for loss function
  lr = 1e-5
  weight_decay = 1e-2

  predictions_path = "/content/drive/MyDrive/task3-iml/submission.csv"

  # Tensorboard setup
  log_dir = "content/drive/MyDrive/task3-iml/tmp/logs"
  training_writer = SummaryWriter(log_dir + "/tensorboard/train")
  validation_writer = SummaryWriter(log_dir + "/tensorboard/valid")

  # Load data in feasible format
  print("Loading data...")

  device_number = torch.cuda.current_device()
  device = torch.device(device_number)

  # determine if we should reload the images
  if reload_images:
    images_path = "/content/drive/MyDrive/task3-iml/data/food.zip (Unzipped Files)"
    images = load_images(images_path)
    torch.save(images, '/content/drive/MyDrive/task3-iml/data/images.pt')
  else:
    try:
      isinstance(images, list)
    except NameError:
      images = torch.load('/content/drive/MyDrive/task3-iml/data/images.pt', map_location=torch.device('cpu'))
    print(f"Images loaded to device `{images[0].device}`")
  #base_path_images = "/content/drive/MyDrive/task3-iml/data/food.zip (Unzipped Files)/food/"

  if mode_choice == 'debugging':
    stopper = 5
  elif mode_choice == 'exploration':
    stopper = 5000
  elif mode_choice == 'evaluation':
    stopper = 'all'
  else:
    raise ValueError('Check mode_choice variable in Settings')

  triplets_train_path = "drive/MyDrive/task3-iml/data/train_triplets.txt"
  triplets_test_path = "drive/MyDrive/task3-iml/data/test_triplets.txt"
  triplets_train_, triplets_test = load_triplets(triplets_train_path, stopper), load_triplets(triplets_test_path, stopper)


  # Split into train and validation set
  triplets_train, triplets_val = train_test_split(triplets_train_, test_size=0.2, shuffle=True, random_state=random_state)
  # len_train = int(np.ceil(0.8*len(triplets_train_)))
  # triplets_train, triplets_val = torch.utils.data.random_split(triplets_train_, [len_train, len(triplets_train_) - len_train])

  # Convert data into torch.utils.data.Dataset
  triplets_train, triplets_val, triplets_test = (
      TripletDataset(triplets_train, images),
      TripletDataset(triplets_val, images),
      TripletDataset(triplets_test, images)
  )
  print(f"# training samples: {len(triplets_train)}")
  print(f"# validation samples: {len(triplets_val)}")
  print(f"# test samples: {len(triplets_test)}")


  # Prepare data for training (batching)
  train_loader = DataLoader(triplets_train, batch_size=batch_size_train, shuffle=True)
  val_loader = DataLoader(triplets_val, batch_size=batch_size_val, shuffle=True)
  test_loader = DataLoader(triplets_test, batch_size=batch_size_val, shuffle=False)

  # Initialize checkpoint dict for saving
  if save_checkpoints:
    checkpoint_dict = dict.fromkeys(list(range(num_epochs))+['last_epoch'])
    for epoch in range(num_epochs):
      checkpoint_dict[epoch] = dict.fromkeys(['epoch', 'model_state_dict', 'optimizer_state_dict', 'loss'])


  # Instantiate model
  model = TripletNet(out_dim=out_dim, number_fc=number_fc, fine_tune=fine_tune)

  use_cuda = torch.cuda.is_available()
  cuda_available = "" if use_cuda else " not"
  print(f"CUDA is{cuda_available} available.")

  if use_cuda:
    print("Enabling GPU parallelism...")
    model = torch.nn.DataParallel(model).cuda()         # important to do this before initializing optimizer


  # Instantiate loss function and optimizer
  loss = nn.TripletMarginLoss(margin=margin)
  optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)


  # Reload model from checkpoint if wanted
  if reload_checkpoint:
    print('Loading from checkpoint...')
    checkpoint = torch.load('/content/drive/MyDrive/task3-iml/checkpoints/' + load_checkpoint_description)

    if epoch_to_load == 'last_epoch':
      epoch_to_load = checkpoint['last_epoch']

    model.load_state_dict(checkpoint[epoch_to_load]['model_state_dict'])
    optimizer.load_state_dict(checkpoint[epoch_to_load]['optimizer_state_dict'])
    epoch = checkpoint[epoch_to_load]['epoch']
    loss = checkpoint[epoch_to_load]['loss']

    print('Loading from checkpoint successfull.')

  # Train model
  model = train(model, loss, optimizer, train_loader, val_loader, num_epochs, use_cuda, training_writer, validation_writer)

  print(f"\nUsing settings: \nmargin = {margin}\nlr = {lr}\nweight_decay = {weight_decay}\nout_dim = {out_dim}\nopimtizer={optimizer}")
  if mode_choice == "evaluation":
    # Model inference
    predictions = predict(model, loss, test_loader, use_cuda)


    # Save predictions
    print(f"Saving predictions to {predictions_path}...")
    predictions = pd.DataFrame(predictions.cpu().numpy()).applymap(lambda x: 1 if x else 0)
    predictions.to_csv(predictions_path, index=False, header=False)