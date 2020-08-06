import torchvision
from torchvision import transforms
transform_train = transforms.Compose([transforms.Resize(256), 
                                      transforms.RandomCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                            (0.2675, 0.2565, 0.2761))])

Train_data = torchvision.datasets.CIFAR100(
    root = './data', train = True, download = True, transform = transform_train)

Valid_data = torchvision.datasets.CIFAR100(
    root = './data', train = False, download = True, transform = transform_train)
