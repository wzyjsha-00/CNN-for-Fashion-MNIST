import torchvision.transforms as transforms
from PIL import Image

from config.hyper_param import *
from config.path import *
from config.torch_package import *


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.406], [0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.406], [0.225])
    ])
}


class FashionMNIST(Dataset):

    def __init__(self, is_train, transform):
        """
        Args:
            transform: (torchvision.transforms) transformation to apply on image
        """

        img_path_list, img_label_list, img_status_list = [], [], []

        with open(train_label_file) as f:
            for line in f:
                img_name = line.strip().split()[0]
                img_label = int(line.strip().split()[-1])

                img_path_list.append(os.path.join(train_data_folder, img_name))
                img_label_list.append(img_label)
                img_status_list.append(int(1))
        
        with open(test_label_file) as f:
            for line in f:
                img_name = line.strip().split()[0]
                img_label = int(line.strip().split()[-1])

                img_path_list.append(os.path.join(test_data_folder, img_name))
                img_label_list.append(img_label)
                img_status_list.append(int(0))

        if is_train:
            self.samples = [[p, l] for p, l, s in zip(img_path_list, img_label_list, img_status_list) if s == 1]
        else:
            self.samples = [[p, l] for p, l, s in zip(img_path_list, img_label_list, img_status_list) if s == 0]

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image_path, image_label = self.samples[idx]
        # image = Image.open(image_path).convert('RGB')
        image = Image.open(image_path)
        image = self.transform(image)

        return image, image_label


def fetch_dataloader(status):
    dataloaders = {}

    for state in status:
        if state == 'train':
            dataset = FashionMNIST(is_train=True, transform=data_transforms['train'])
            dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        else:
            dataset = FashionMNIST(is_train=False, transform=data_transforms['test'])
            dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

        dataloaders[state], dataloaders[state+'_size'] = dl, len(dataset)

    print("DataLoaders: ", dataloaders)

    return dataloaders
