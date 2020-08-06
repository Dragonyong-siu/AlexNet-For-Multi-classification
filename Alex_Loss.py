def Alex_Loss(Logits_Image, Image_label):
  Loss_function = nn.CrossEntropyLoss()
  Loss = Loss_function(Logits_Image, Image_label)
  return Loss
