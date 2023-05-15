import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader

def get_data(batch_size =8):
    train_dir = 'data/train/'
    test_dir = 'data/val/'

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform = transform)
    test_dataset = datasets.ImageFolder(test_dir, transform = transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

if __name__ == '__main__':
    train, test = get_data()
    for images, labels in train:
        print(labels)


