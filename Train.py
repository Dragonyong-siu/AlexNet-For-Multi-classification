from tqdm import tqdm
import torch.nn.functional as F
def Train_one_time(dataloader, model, optimizer, device):
  model.train()
  Book = tqdm(dataloader, total = len(dataloader))
  total_loss = 0.0
  for bi, Dictionary in enumerate(Book):
    Image_ids = Dictionary['Image_ids'].to(device)
    Image_label = Dictionary['Image_label'].squeeze(1).to(device)

    model.zero_grad()
    Logits_Image = model(Image_ids)
    
    Loss = Alex_Loss(Logits_Image, Image_label)
    Loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    total_loss += Loss.item()
  Average_Train_Loss = total_loss / len(dataloader)
  print(" Average Train Loss: {0:.2f}".format(Average_Train_Loss))

def Train_Epochs(dataloader, model, Epochs, Learning_rate):
  optimizer = torch.optim.AdamW(ALEX_Model.parameters(), lr = Learning_rate)

  for i in range(Epochs):
    print(f"EPOCHS:{i+1}")
    print('TRAIN')
    Train_one_time(dataloader, model, optimizer, device)
  torch.save(ALEX_Model, '/content/gdrive/My Drive/' + f'ALEX_Model')

Train_Epochs(Train_dataloader, ALEX_Model, Epochs = 100, Learning_rate = 0.0002)
