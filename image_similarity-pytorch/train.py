import torch

def get_label(a, p, n):
  """Computes the predicted label based on the relative distance between a, p and a, n, respectively"""
  distance_p = torch.norm(a - p, 2)
  distance_n = torch.norm(a - n, 2)
  
  label = distance_p < distance_n 

  return label


def accuracy(a, p, n) -> torch.Tensor:
  """Computes the accuracy of a batch (fraction of correct classifications)"""
  correct_label = get_label(a, p, n)
  acc = torch.mean(correct_label.float())

  return acc


def train(model, loss, optimizer, train_loader, val_loader, num_epochs, use_cuda: bool) -> torch.nn.Module:
  """Train a model"""

  total_examples_seen = 0

  for epoch in range(num_epochs):
    # Put model in train mode
    model.train()

    # Initialize statistics
    train_loss_cum = 0

    for index, (a, p, n) in enumerate(train_loader):
      if use_cuda:
         a, p, n = [_.cuda() for _ in [a, p, n]]
      
      a_pred, p_pred, n_pred = model(a, p, n)

      optimizer.zero_grad()
      train_loss = loss(a_pred, p_pred, n_pred)

      train_loss.backward()
      optimizer.step()

      # Compute statistics
      batch_size = a.shape[0] # not necessarily constant 
      total_examples_seen += batch_size
      train_loss_cum += train_loss * batch_size
      accuracy_cum += accuracy(a_pred, p_pred, n_pred) * batch_size 

      if index % 25:
        print(f"Epoch: {epoch} | Training iteration: {index} | Training loss: " 
              f"{train_loss_cum / total_examples_seen:.4f} | Training accuracy: {accuracy_cum / total_examples_seen}")


    # Put model in evaluation mode
    model.eval()

    # Compute validation loss once (from the beginning) every epoch 
    with model.no_grad():
      val_loss_cum = 0
      total_examples_seen_val = 0
      accuracy_cum_val = 0

      for index, (a, p, n) in enumerate(val_loader):
        if use_cuda:
         a, p, n = [_.cuda() for _ in [a, p, n]]
        
        a_pred, p_pred, n_pred = model(a, p, n)

        val_loss = loss(a_pred, p_pred, n_pred)

        # Compute statistics
        batch_size = a.shape[0]
        total_examples_seen_val += batch_size
        val_loss_cum += val_loss * batch_size
        accuracy_cum_val += accuracy(a_pred, p_pred, n_pred) * batch_size 

    print(f"Epoch: {epoch} | Validation loss: {val_loss_cum / total_examples_seen_val} |" 
          f" Validation accuracy: {accuracy_cum_val / total_examples_seen_val}") 
  
  return model 