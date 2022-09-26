import torch 
import numpy as np

from train import get_label

def predict(model, loss, test_loader, use_cuda) -> torch.Tensor:
  """Predict labels 0/1 based on test set triplets"""

  num_batches = len(test_loader)
  predictions = np.array([])

  for index, (a, p, n) in enumerate(test_loader):
    if use_cuda:
         a, p, n = [_.cuda() for _ in [a, p, n]]

    a_pred, p_pred, n_pred = model(a, p, n)

    predictions.append(get_label(a_pred, p_pred, n_pred))

    if index % 10 == 0:
      print(f"{index / num_batches * 100:.1f}% of batches processed")

  return torch.from_numpy(predictions)