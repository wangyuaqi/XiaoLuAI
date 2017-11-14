import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from facescore.config import *
from facescore.face_beauty_pytorch import FaceBeautyDataset

transform = transforms.Compose(
    [transforms.RandomCrop(config['image_size']),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transformed_dataset = FaceBeautyDataset(transform=None)
dataloader = DataLoader(transformed_dataset, batch_size=config['batch_size'],
                        shuffle=True, num_workers=4)

for i in range(len(transformed_dataset)):
    sample = transformed_dataset.__getitem__(i)

    print(i, sample['image'].shape, sample['score'])
