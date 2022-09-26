import torch
from torch.utils.data import Dataset

class TripletDataset(Dataset):

    def __init__(self, triplet_list, images):
        self.images = images
        self.triplet_list = triplet_list

    def __len__(self):
        return len(self.triplet_list)

    def __getitem__(self, idx):
        a, p, n = self.triplet_list[idx]

        a_image = self.images[int(a)]
        p_image = self.images[int(p)]
        n_image = self.images[int(n)]

        return a_image, p_image, n_image


  