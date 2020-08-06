import torch
class Alex_Dataset(torch.utils.data.Dataset):
  def __init__(self, data):
    self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    Dictionary = {}
    Image_ids = self.data[index][0]
    Image_label = torch.Tensor([self.data[index][1]]).long()
    Dictionary['Image_ids'] = Image_ids
    Dictionary['Image_label'] = Image_label

    return Dictionary

from torch.utils.data import DataLoader
BATCH_SIZE = 8
Train_dataloader = DataLoader(Alex_Dataset(Train_data),
                              batch_size = BATCH_SIZE,
                              shuffle = True,
                              drop_last = True)
Valid_dataloader = DataLoader(Alex_Dataset(Valid_data),
                              batch_size = BATCH_SIZE,
                              shuffle = True,
                              drop_last = True)
