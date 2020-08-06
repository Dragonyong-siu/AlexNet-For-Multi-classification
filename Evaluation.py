import numpy as np
def Evaluation_index(dataloader, model, device):
    model.eval()
    Labels = []
    Outputs = []
    with torch.no_grad():
      for bi, Dictionary in enumerate(dataloader):
        Image_ids = Dictionary['Image_ids'].to(device)
        Image_label = Dictionary['Image_label'].to(device).squeeze(1)
        Logits_Prob = model(Image_ids).cpu()
        output = np.argmax(Logits_Prob, axis = 1)
        Labels.extend(Image_label.cpu().detach().numpy().tolist())
        Outputs.extend(output.cpu().detach().numpy().tolist())
        
    return Labels, Outputs

Labels, Outputs = Evaluation_index(Valid_dataloader, ALEX_Model, device)

from sklearn.metrics import accuracy_score
accuracy_score(Labels, Outputs)
