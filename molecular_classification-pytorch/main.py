import torch
from torch import nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from typing import List, Tuple

import pandas as pd
import numpy as np
from datetime import datetime
import time

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import uuid
import gc

## MODEL

class MLP(nn.Module):
  """MLP to be used both as encoder and regressor. Usable for both pretraining and training task"""

  def __init__(self, dims: List[int], activation, use_batch_norm: bool, is_encoder: bool = False):
    super(MLP, self).__init__()

    self.dims = dims

    def create_layer(in_dim, out_dim) -> nn.Module:
      """Creates a single layer"""
      components = [
          nn.Linear(in_dim, out_dim),
          activation,
          nn.Dropout(p=0.5)
      ]
      if use_batch_norm:
        components.append(nn.BatchNorm1d(num_features=out_dim))

      return nn.Sequential(*components)

    # Create MLP with depth len(dims)-1
    if is_encoder:
      self.layers = nn.Sequential(
          *[create_layer(in_dim, out_dim) for in_dim, out_dim in zip(dims[:-1], dims[1:])]
      )
      self.linear = None
    else:
      self.layers = nn.Sequential(
          *[create_layer(in_dim, out_dim) for in_dim, out_dim in zip(dims[:-2], dims[1:-1])]
      ) if len(dims) >= 3 else None

      self.linear = nn.Linear(dims[-2], dims[-1])

  def __len__(self):
        return len(self.dims)-1

  def forward(self, x):
    if self.layers:
      x = self.layers(x)
    if self.linear:
      x = self.linear(x)

    return x


class Regressor(nn.Module):
  """Wrapper for standard regressor, to be attached to encoder"""
  pas


class FullNetwork(nn.Module):
  """Network consisting of a regressor attached to an encoder"""

  def __init__(self, encoder, dims_regressor, activation_regressor, use_batch_norm_regressor, activation_encoder=None, use_batch_norm_encoder=None, freeze=False):
    """args: encoder is either a nn.Module object or a list of dimensions (like dims_regressor). This is
    to be able to either create an encoder from scratch or use an existing (e.g. pretrained) encoder. """
    super(FullNetwork, self).__init__()

    if not activation_encoder:
      activation_encoder = activation_regressor
    if not use_batch_norm_encoder:
      use_batch_norm_encoder = use_batch_norm_regressor

    if isinstance(encoder, list):
      self.encoder = MLP(encoder, activation_encoder, use_batch_norm_encoder, is_encoder=True)
    else:
      if freeze:
        for _, param in encoder.named_parameters():
          param.requires_grad = False

      self.encoder = encoder

    self.regressor = MLP(dims_regressor, activation_regressor, use_batch_norm_regressor)


  def get_encoder(self, get_n_first=None) -> nn.Module:
    """Get the encoder of this network"""

    if get_n_first:

      # Remove len-n layer of original encoder
      self.layers = nn.Sequential(
        *list(self.encoder.children())[0][:-(len(self.encoder)-get_n_first)]
      )

      return self.layers

    return self.encoder

  def forward(self, x):

    x = self.encoder(x)
    x = self.regressor(x)

    return x

# CUSTOM LOSS

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


## PREPROCESS DATA

def preprocess_feature(pretrain_path, train_path, test_path, min=0, max=1):
  """
  For pre-processing. Delete some features that are almost constant in pretrain_feature.
  min<sum<max
  """

  raw_path = pretrain_path[:-15] + ".csv"
  raw = pd.read_csv(raw_path)
  df = raw.iloc[:,2:]
  df_sum = pd.DataFrame(df.sum())
  df_sum = df_sum/len(df)


  for processed_path in [pretrain_path, train_path, test_path]:
    raw_path = processed_path[:-15] + ".csv"
    raw = pd.read_csv(raw_path)
    df = raw.iloc[:,2:]
    feature_use = df.columns[df_sum[0]> min]
    processed = pd.merge(raw.iloc[:,:2],df[feature_use],left_index=True,right_index=True,how='outer')

    processed.to_csv(processed_path,index=False)

    num_feature_preprocessed = len(feature_use)
    print("Pre-process finished!", processed_path)
  return num_feature_preprocessed



## DATASET

class MoleculeDataset(Dataset):

    def __init__(self, x_data, y_data=None):

        # Initialize normalized x_data
        self.x_data = x_data

        # Initialize y_data
        self.y_data = y_data


    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):

        if self.y_data is None:
            return self.x_data[idx]
        else:
            return (self.x_data[idx], self.y_data[idx])


## TRAINING LOOP

# Train

def train(model, loss, optimizer, train_loader, val_loader, num_epochs, use_cuda, assess_pretraining=True, unsupervised=False, scheduler=None) -> torch.nn.Module:
  """Train model and return the trained version of it"""

  # Initialize statistics
  total_examples_seen = 0
  train_loss_cum = 0

  val_loss_cum = 0
  total_examples_seen_val = 0

  best_loss = 100
  best_epoch = 0

  best_loss_epoch = 100
  best_epoch_epoch = 0
  best_train_loss_epoch = 100

  num_batches_train = len(train_loader)

  for epoch in range(num_epochs):
    model.train()

    total_examples_seen_epoch = 0
    train_loss_cum_epoch = 0


    val_loss_cum_epoch = 0
    total_examples_seen_val_epoch = 0

    for index, (x, y) in enumerate(train_loader):
      if use_cuda:
         x, y = x.cuda(), y.cuda()

      prediction = model(x.float())

      optimizer.zero_grad()

      if unsupervised:
        train_loss = loss(prediction, x)
      else:
        train_loss = loss(prediction, y)

      train_loss.backward()
      optimizer.step()

      # Compute statistics
      batch_size = x.shape[0] # not necessarily constant
      total_examples_seen += batch_size
      total_examples_seen_epoch += batch_size

      train_loss_cum += train_loss.item() * batch_size
      train_loss_cum_epoch += train_loss.item() * batch_size


      # Free up GPU memory
      del x, y, prediction, train_loss
      torch.cuda.empty_cache()

    if assess_pretraining:
      # Validate
      model.eval()

      with torch.no_grad():
        for index, (x, y) in enumerate(val_loader):
          if use_cuda:
            x, y = x.cuda(), y.cuda()

          prediction = model(x.float())

          if unsupervised:
            val_loss = loss(prediction, x)
          else:
            val_loss = loss(prediction, y)

          # Compute statistics
          batch_size = x.shape[0]
          total_examples_seen_val += batch_size
          total_examples_seen_val_epoch += batch_size

          val_loss_cum += val_loss.item() * batch_size
          val_loss_cum_epoch += val_loss.item() * batch_size

          # Free up some GPU memory
          del x, y, prediction, val_loss
          torch.cuda.empty_cache()

    # Save if model improved
    val_loss = val_loss_cum / total_examples_seen_val
    val_loss_epoch = val_loss_cum_epoch / total_examples_seen_val_epoch

    if val_loss < best_loss:
      best_params = model.state_dict()
      best_epoch = epoch
      best_loss = val_loss

    if val_loss_epoch < best_loss_epoch:
      best_params_epoch = model.state_dict()
      best_loss_epoch = val_loss_epoch
      best_epoch_epoch = epoch+1

      best_train_loss_epoch = train_loss_cum_epoch / total_examples_seen_epoch


    # Log
    if epoch % 5 == 0:
      print(f"Epoch: {epoch+1} | Training loss: {train_loss_cum / total_examples_seen:.4f} | Validation loss: {val_loss_cum / total_examples_seen_val:.4f}"
      f" | Epoch val loss: {val_loss_epoch:.4f}")

    if scheduler:
      scheduler.step(val_loss)
      print("Decreasing learning rate...")

  print(f"Best performance: Epoch: {best_epoch_epoch} | Training loss: {train_loss_cum / total_examples_seen:.4f} | Validation loss: {best_loss_epoch}\n")

  model.load_state_dict(best_params_epoch)

  return model, best_train_loss_epoch, best_loss_epoch


# MANUAL ENCODE DATA
def data_through_encoder(encoder, loader, encoder_out_dim: int, use_cuda, test=False) -> np.array:

    """ Pipeline for Data through a given encoder."""

    if use_cuda:
        print("Enabling GPU parallelism on encoder...\n")
        encoder = torch.nn.DataParallel(encoder).cuda()

    encoder.eval()

    encoder_output = np.zeros((1, encoder_out_dim))
    encoder_output_y = np.zeros((1, 1))

    ids = np.zeros((1,))

    with torch.no_grad():

        if test:
            for index, xx in enumerate(loader):
                if use_cuda:
                    xx = xx.cuda()

                # Extract ids
                ids_tmp = xx[:, 0].cpu()
                ids_tmp = ids_tmp.numpy()
                ids = np.append(ids, ids_tmp, axis=0)


                xx = xx[:,1:]
                tmp = encoder(xx)
                tmp = tmp.cpu()
                tmp = tmp.numpy()
                encoder_output = np.append(encoder_output, tmp, axis=0)

                del xx, tmp, ids_tmp

            encoder_output = encoder_output[1:, :]
            ids = ids[1:]
            ids = np.expand_dims(ids, 1)

            torch.cuda.empty_cache()
            return encoder_output, ids


        else:
            for index, (xx, yy) in enumerate(loader):
                if use_cuda:
                    xx = xx.cuda()

                tmp = encoder(xx)
                tmp = tmp.cpu()
                tmp = tmp.numpy()

                yy = yy.numpy()

                encoder_output = np.append(encoder_output, tmp, axis=0)
                encoder_output_y = np.append(encoder_output_y, yy, axis=0)

                del xx, tmp, yy

            encoder_output = encoder_output[1:, :]
            encoder_output_y = encoder_output_y[1:, :]
            torch.cuda.empty_cache()

            return encoder_output, encoder_output_y

# PREDICT

def predict(model, test_loader, use_cuda):
  """Predict HOMO-LUMO gap values based on test set features"""

  num_batches = len(test_loader)
  predictions = []

  model.eval()

  with torch.no_grad():
    for index, x in enumerate(test_loader):
      if use_cuda:
         x = x.cuda()

      # Extract ids
      ids = x[:, 0]
      x = x[:, 1:]

      prediction = model(x)

      prediction = torch.column_stack((ids, prediction))
      predictions.append(prediction)

      if index % 25 == 0 and index != 0:
        print(f"{index / num_batches * 100:.1f}% of test batches processed")

      del x, prediction
      torch.cuda.empty_cache()

  return torch.cat(predictions)

## LOGGER

def logger(kwargs):
  """ Checks access on logfile and writes hyperparameters of the current run as a new line."""

  # Generate run id
  run_id = str(uuid.uuid4())[:8]
  log_path = '/content/drive/MyDrive/task4-iml/data/logfile_2.csv'
  user_path = '/content/drive/MyDrive/task4-iml/data/current_user.txt'
  user = kwargs["user"]

  while True:
    with open(user_path, 'r+') as text:
      current_user = text.read()
      if current_user == '':

        text.write(user)
        text.close()

        print('Writing logfile..\n')
        time.sleep(10)

        with open(user_path, 'r') as text:
          current_user = text.read()
          if current_user == user:

            try:
              logfile = pd.read_csv(log_path, sep=';')

              kwargs["datetime"] = datetime.now()
              kwargs["run_id"] = run_id
              new_data = pd.DataFrame.from_dict({k: [w] for k, w in kwargs.items()})

              new_log = pd.concat([logfile, new_data])

              new_log.to_csv(log_path, sep=';',index=False)
              print('Logs saved..\n')
            except BaseException as e:
              print(f"Something went wrong: {e}")

          with open(user_path, 'w') as text:
            text.write('')
            text.close()

          return run_id

      else:
        wait_time = 20 # [seconds]
        print(f'Waiting {int(wait_time)} seconds for access to logfile.. Current user: {current_user}\n')
        time.sleep(wait_time)


def sort_log(by="validation_error", version = "2", save=False, ascending=True):
  """
  Read logfile and get all the data.
  Sort it in ascending order wrt "validation_error" or "prevalidation_error".
  """
  logfile = pd.read_csv(f'/content/drive/MyDrive/task4-iml/data/logfile_{version}.csv', sep=";")
  logfile_asc = logfile.sort_values(by=by, ascending=ascending)
  if save:
    logfile_asc.to_csv('/content/drive/MyDrive/task4-iml/data/logfile_1_ascending.csv', sep=';',index=False)

  return logfile_asc

# MAIN

def remove_smiles(datas: List[pd.DataFrame]) -> List[pd.DataFrame]:
  return [data.drop(columns="smiles") for data in datas]


def scale(datas: List[pd.DataFrame], scaler=None) -> Tuple[List[np.ndarray], object]:
  if not scaler:
    scaler = StandardScaler()
    data = datas[0]

    return [scaler.fit_transform(data)], scaler

  return [scaler.transform(data) for data in datas], None


def save(all_data: List[torch.Tensor], paths) -> None:
  """Saves all data to the paths in paths"""
  for data, path in zip(all_data, paths):
    torch.save(data, path)


def to_tensor(datas: List[np.ndarray]) -> List[torch.Tensor]:
  """Convert data (without smiles column) to torch.tensor"""
  return [torch.tensor(x.to_numpy(), dtype=torch.float) if isinstance(x, pd.DataFrame) else torch.tensor(x, dtype=torch.float) for x in datas]


def load_data(feature_path, label_path = None, keep_id = False) -> Tuple[pd.DataFrame]:
  """Loads data without the id column"""
  X = pd.read_csv(feature_path)
  if not keep_id:
    X = X.drop(columns="Id")

  if not label_path:
    return X

  y = pd.read_csv(label_path)
  y = y.drop(columns="Id")

  return X, y


def setup_gpu():
  torch.cuda.empty_cache()

  use_cuda = torch.cuda.is_available()
  cuda_available = "" if use_cuda else " not"
  print(f"CUDA is{cuda_available} available.")

  return use_cuda


smiles = ""  # "", "_total",  "_onlysmiles"
preprocess ="" # "_preprocess"  # ""


# Paths
base_path = "/content/drive/MyDrive/task4-iml/data"
PREDICTION_PATH = "/content/drive/MyDrive/task4-iml/predictions/"
PRETRAIN_FEATURES_PATH = f"{base_path}/csv/pretrain_features{smiles}{preprocess}.csv"
PRETRAIN_LABELS_PATH = f"{base_path}/csv/pretrain_labels.csv"
TRAIN_FEATURES_PATH = f"{base_path}/csv/train_features{smiles}{preprocess}.csv"
TRAIN_LABELS_PATH = f"{base_path}/csv/train_labels.csv"
TEST_FEATURES_PATH = f"{base_path}/csv/test_features{smiles}{preprocess}.csv"

if preprocess == "_preprocess":
  num_feature_preprocessed = preprocess_feature(PRETRAIN_FEATURES_PATH, TRAIN_FEATURES_PATH, TEST_FEATURES_PATH, min=0.001, max=1)

# Saved data as tensors
TENSOR_FILE_NAMES = ["train_features", "train_labels", "val_features", "val_labels", "test_features"]
TENSOR_PATHS_TRAIN = [f"{base_path}/tensor/{name}_tensor.pt" for name in TENSOR_FILE_NAMES]

TENSOR_FILE_NAMES_NOT_ASSESS_PRETRAIN = ["pretrain_features", "pretrain_labels"]
TENSOR_PATHS_NOT_ASSESS_PRETRAIN = [f"{base_path}/tensor/{name}_tensor.pt" for name in TENSOR_FILE_NAMES_NOT_ASSESS_PRETRAIN]

TENSOR_FILE_NAMES_ASSESS_PRETRAIN = ["pretrain_features_train", "pretrain_labels_train", "pretrain_features_val", "pretrain_labels_val"]
TENSOR_PATHS_ASSESS_PRETRAIN = TENSOR_PATHS = [f"{base_path}/tensor/{name}_tensor.pt" for name in TENSOR_FILE_NAMES_ASSESS_PRETRAIN]

def main():
  # Settings
  reload_data = True              # If true recompute and retransform data, else load from file

  assess_pretraining = True       # If true pretrained model performance on validation set is tracked.
                                  # When optimal parameters are found we should pretrain on the full dataset,
                                  # by setting this param to False

  user = ''                 # your name please

  comment = 'with preprocessing(delete features that are 99.9% zero in pretrain). Same as best 0711bfdb, but with 80% training'                    # if you want to write a specific string to be saved in the logfile for this run

  # Set use_smile = True above to use engineered features

  # General hyperparameters
  random_state = 1
  batch_size_train = 128
  batch_size_val = 128
  test_size_train = 0.2

  num_epochs_unsup = 30
  num_epochs_pretrain = 60
  num_epochs = 3000

  learning_rate_unsup = 1e-3
  learning_rate_pretrain = 1e-4
  learning_rate = 1e-4

  weight_decay_unsup = 1e-3
  weight_decay_pretrain = 1e-1
  weight_decay = 1e-1

  use_n_first_layers = 1      # ZERO AS DEFAULT -> USE ALL LAYERS


  prebuild_regressor = False                             # False, if you want to build our custom NN as last model stage
  regressor_model = XGBRegressor(colsample_bytree = 0.3,
                            learning_rate = 0.05,
                            max_depth = 3,
                            n_estimators = 1000)


  # Model hyperparams
  activation = torch.nn.ReLU()
  use_batch_norm = True

  if preprocess == "_preprocess":
    dims_encoder = [num_feature_preprocessed, 512, 256, 128]
  elif smiles == "_total":
    dims_encoder = [3048, 512, 256, 128]
  elif smiles == "_onlysmiles":
    dims_encoder = [2048, 512, 256, 128]
  else:
    dims_encoder = [1000, 512, 256, 128]

  # dims_encoder = [1000, 512, 256, 128]
  dims_decoder = [_ for _ in reversed(dims_encoder)]

  if use_n_first_layers == 0:
    dims_regressor_pretrain = [dims_encoder[-1], 1]
    dims_regressor = [dims_encoder[-1], 1]
  else:
    dims_regressor_pretrain = [dims_encoder[use_n_first_layers], 1]
    dims_regressor = [dims_encoder[use_n_first_layers], 1]


  # Setup GPU
  use_cuda = setup_gpu()

  # Load data
  print(f"Loading data...")
  pretrain_paths = TENSOR_PATHS_ASSESS_PRETRAIN if assess_pretraining else TENSOR_FILE_NAMES_NOT_ASSESS_PRETRAIN
  tensor_paths = [*TENSOR_PATHS_TRAIN, *pretrain_paths]


  if reload_data:
    X_pretrain, y_pretrain = load_data(PRETRAIN_FEATURES_PATH, PRETRAIN_LABELS_PATH)
    X_train, y_train = load_data(TRAIN_FEATURES_PATH, TRAIN_LABELS_PATH)
    X_test = load_data(TEST_FEATURES_PATH, keep_id=True)

    # Remove smiles column
    X_train, X_test = remove_smiles([X_train, X_test])
    X_pretrain, = remove_smiles([X_pretrain])

    # Split into training and validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size_train, shuffle=True, random_state=random_state)

    all_data = [X_train, y_train, X_val, y_val, X_test]

    if assess_pretraining:
      # Split pretraining set into training and validation set,
      X_train_pretrain, X_val_pretrain, y_train_pretrain, y_val_pretrain = train_test_split(X_pretrain, y_pretrain, test_size=test_size_train, shuffle=True, random_state=random_state)

      all_data += [X_train_pretrain, y_train_pretrain, X_val_pretrain, y_val_pretrain]

    else:
      # Full pretraining data
      all_data += [X_pretrain, y_pretrain]

    # Transform to tensor
    all_data = to_tensor(all_data)

    # Save to disk
    save(all_data, tensor_paths)

  else:
    # Load data from file
    all_data = [torch.load(path) for path in tensor_paths]

  # Unpack and wrap data in dataloader
  if assess_pretraining:
    X_train, y_train, X_val, y_val, X_test, X_train_pretrain, y_train_pretrain, X_val_pretrain, y_val_pretrain = all_data

    data_train_pretrain, data_val_pretrain = MoleculeDataset(X_train_pretrain, y_train_pretrain), MoleculeDataset(X_val_pretrain, y_val_pretrain)

    train_loader_pretrain = DataLoader(data_train_pretrain, batch_size=batch_size_train, shuffle=False)
    val_loader_pretrain = DataLoader(data_val_pretrain, batch_size=batch_size_train, shuffle=False)

    print("Pretraining statistics:")
    print(f"# training samples: {len(data_train_pretrain)}")
    print(f"# validation samples: {len(data_val_pretrain)}\n")
  else:
    X_train, y_train, X_val, y_val, X_test, X_pretrain, y_pretrain = all_data
    data_train_pretrain = MoleculeDataset(X_pretrain, y_pretrain)

    train_loader_pretrain = DataLoader(data_train_pretrain, batch_size=batch_size_train, shuffle=False)
    val_loader_pretrain = None

    print("Pretraining statistics:")
    print(f"# training samples: {len(data_train_pretrain)}")
    print(f"# validation samples: 0\n")

  dims_encoder[0] = X_train.size()[1]


  # Wrap as training datasets
  data_train, data_val, data_test = MoleculeDataset(X_train, y_train), MoleculeDataset(X_val, y_val), MoleculeDataset(X_test)

  # Feed training data into Dataloaders
  train_loader = DataLoader(data_train, batch_size=batch_size_train, shuffle=False)
  val_loader = DataLoader(data_val, batch_size=batch_size_val, shuffle=False)
  test_loader = DataLoader(data_test, batch_size=batch_size_val, shuffle=False)

  # Print data statistics
  print("Training statistics:")
  print(f"# training samples: {len(data_train)}")
  print(f"# validation samples: {len(data_val)}")
  print(f"# test samples: {len(data_test)}\n")

  # Unsupervised pretraining
  model_unsup = FullNetwork(dims_encoder, dims_decoder, activation, use_batch_norm)
  if use_cuda:
        print("Enabling GPU parallelism on pretrained model...\n")
        model_unsup = torch.nn.DataParallel(model_unsup).cuda()

  # Loss and optimizer
  loss = RMSELoss()
  optimizer = torch.optim.Adam(model_unsup.parameters(),
                               learning_rate_unsup,
                               weight_decay=weight_decay_unsup)
  scheduler_unsup = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


  # Unsupervised pretraining
  print("Pretraining the following encoder...\n")
  if use_cuda:
    print(model_unsup.module, "\n")
  model_unsup, unsup_error, unsup_val_error = train(model_unsup,
                                                              loss,
                                                              optimizer,
                                                              train_loader_pretrain,
                                                              val_loader_pretrain,
                                                              num_epochs_unsup,
                                                              use_cuda,
                                                              assess_pretraining,
                                                              unsupervised=True,
                                                              scheduler=scheduler_unsup)

  encoder = model_unsup.module.get_encoder(use_n_first_layers) if use_cuda else model_unsup.get_encoder(use_n_first_layers)
  del model_unsup

  # Prepare pretrained model
  model_pretrain = FullNetwork(encoder, dims_regressor_pretrain, activation, use_batch_norm)

  if use_cuda:
    print("Enabling GPU parallelism on pretrained model...\n")
    model_pretrain = torch.nn.DataParallel(model_pretrain).cuda()


  # Loss and optimizer
  optimizer = torch.optim.Adam(model_pretrain.parameters(),
                               learning_rate_pretrain,
                               weight_decay=weight_decay_pretrain)

  # Pretrain model
  print("Pretraining the following encoder...\n")
  if use_cuda:
    print(model_pretrain.module, "\n")
  model_pretrain, pretrain_error, prevalidation_error = train(model_pretrain,
                                                              loss,
                                                              optimizer,
                                                              train_loader_pretrain,
                                                              val_loader_pretrain,
                                                              num_epochs_pretrain,
                                                              use_cuda,
                                                              assess_pretraining)

  # Get encoder from pretrained network, and feed into new model
  encoder = model_pretrain.module.get_encoder() if use_cuda else model_pretrain.get_encoder()

  # Split Code in Tailor-made or Prebuild Regressor

  if prebuild_regressor:

    if use_n_first_layers == 0:
      encoder_out_dim = dims_encoder[-1]
    else:
      encoder_out_dim = dims_encoder[use_n_first_layers]

    # Training
    encoder_output, encoder_output_y = data_through_encoder(encoder, train_loader, encoder_out_dim, use_cuda, test=False)
    regressor_model.fit(encoder_output, encoder_output_y)
    #train_error = regressor_model.score(encoder_output, encoder_output_y)
    train_predict = regressor_model.predict(encoder_output)
    train_error = mean_squared_error(encoder_output_y, train_predict)**(1./2.)


    # Validation
    encoder_output_val, encoder_output_y_val = data_through_encoder(encoder, val_loader, encoder_out_dim, use_cuda, test=False)
    #validation_error = regressor_model.score(encoder_output_val, encoder_output_y_val)
    val_predict = regressor_model.predict(encoder_output_val)
    validation_error = mean_squared_error(encoder_output_y_val, val_predict)**(1./2.)



    # Prediction
    encoder_output_prediction, ids = data_through_encoder(encoder, test_loader, encoder_out_dim, use_cuda, test=True)
    predictions = regressor_model.predict(encoder_output_prediction)
    predictions = np.expand_dims(predictions, 1)
    predictions = np.append(ids, predictions, axis=1)
    predictions = pd.DataFrame(predictions)

  else:

    model = FullNetwork(encoder, dims_regressor, activation, use_batch_norm)

    if use_cuda:
          print("Enabling GPU parallelism on model...\n")
          model = torch.nn.DataParallel(model).cuda()

    # Train model on main task
    print("Training following network on main task...\n")
    print(model.module, "\n")
    optimizer = torch.optim.Adam(model.parameters(),
                                learning_rate,
                                weight_decay=weight_decay)
    scheduler_train = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    model, train_error, validation_error = train(model,
                                                loss,
                                                optimizer,
                                                train_loader,
                                                val_loader,
                                                num_epochs,
                                                use_cuda,
                                                scheduler=scheduler_train)

    predictions = predict(model, test_loader, use_cuda)
    predictions = pd.DataFrame(predictions.cpu().numpy())
    predictions = pd.DataFrame(predictions)



  # Log
  log_kwargs = {"user": user,
              "use_n_first_layers": use_n_first_layers,
              "batch_size_train": batch_size_train,
              "batch_size_val": batch_size_val,
              "num_epochs_unsup": num_epochs_unsup,
              "num_epochs_pretrain": num_epochs_pretrain,
              "num_epochs": num_epochs,
              "learning_rate_unsup": learning_rate_unsup,
              "learning_rate_pretrain": learning_rate_pretrain,
              "learning_rate": learning_rate,
              "weight_decay_unsup": weight_decay_unsup,
              "weight_decay_pretrain": weight_decay_pretrain,
              "weight_decay": weight_decay,
              "activation": activation,
              "use_batch_norm": use_batch_norm,
              "dims_encoder": dims_encoder,
              "dims_regressor_pretrain": dims_regressor_pretrain,
              "dims_regressor": dims_regressor,
              "assess_pretraining": assess_pretraining,
              "unsup_error": unsup_error,
              "unsup_val_error": unsup_val_error,
              "pretrain_error": pretrain_error,
              "prevalidation_error": prevalidation_error,
              "train_error": train_error,
              "validation_error": validation_error,
              "comment": comment}
  run_id = logger(log_kwargs)

  print(f"Saving predictions to {PREDICTION_PATH}...")
  predictions.to_csv(PREDICTION_PATH+str(run_id)+'.csv', index=False, header=["Id", "y"])

  print(f"\nRun id: {run_id}")

if __name__ == "__main__":
  main()