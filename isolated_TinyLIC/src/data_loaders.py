import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from torch.utils.data import Dataset
from src.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import os
from src.patch_utils import patches_from_tensor

# def toSVD(x, args):
#     if args.im_size > args.embedding_dim:
#         y = patches_from_tensor(x, args.embedding_dim)
#     y = rank_vecs(y, args.rank)
#     y = y.reshape(y.shape[0], -1, y.shape[-1])
#     return y

def get_loaders(args, worker_init_fn, dev_count, device):
    # batch_size = args.batch_size
    # workers = args.num_workers
    root_dir=args.dataset_root
    dataset=args.dataset
    im_size = 32
    
    if dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root=root_dir+'CIFAR10_data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
            ]), download=True)

        val_dataset = datasets.CIFAR10(root=root_dir+'CIFAR10_data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
            ]))

        test_dataset = None
        
    elif dataset == 'MNIST':
        train_dataset = datasets.MNIST(root=root_dir+'MNIST_data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomCrop(32, 2),
                               transforms.ToTensor()
                           ]))
    
        val_dataset = datasets.MNIST(root=root_dir+'MNIST_data', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.RandomCrop(32, 2),
                               transforms.ToTensor()
                           ]))

        test_dataset = None
        
    elif dataset == 'CELEBA':
        train_dir = root_dir+"CELEBA_data/training"
        val_dir = root_dir+"CELEBA_data/validation"
        test_dir = root_dir+"CELEBA_data/testing"
        
        paths_exist = [os.path.exists(train_dir),
                       os.path.exists(test_dir),
                       os.path.exists(val_dir)]
        if False in paths_exist:
            print("Did you download the 64x64 version from Kaggle or another repo?\n\
                   Download and unzip the dataset into the following structure:\n\
                   ./datasets/CELEBA_data/\n\
                       -> testing/\n\
                            -> *.jpg\n\
                       -> training/\n\
                            -> *.jpg\n\
                       -> validation/\n\
                            -> *.jpg\n\
                    You can use the command below to get the zip file, then use $ unzip celeba-small-images-dataset.zip: \
                    $ curl -L -o ./celeba-small-images-dataset.zip https://www.kaggle.com/api/v1/datasets/download/arnrob/celeba-small-images-dataset")
        class CelebADataset(Dataset):
            """Face Landmarks dataset."""
            def __init__(self, root_dir, transform=None):
                """
                Args:
                    root_dir (string): Directory with all the images.
                    transform (callable, optional): Optional transform to be applied
                """
                self.root_dir = root_dir
                self.images = next(os.walk(root_dir))[2]
                self.transform = transform

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                if torch.is_tensor(idx):
                    idx = idx.tolist()

                image = Image.open(os.path.join(self.root_dir, self.images[idx]))
                if self.transform:
                    image = self.transform(image)
                if args.model != 'lr-svdinit': 
                    return image
                else:
                    raise NotImplementedError("No SVD init for CelebA")
                    # return (toSVD(image, args), image)  # target, label
        
        train_dataset = CelebADataset(root_dir=train_dir, transform=transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(64, 4),
                transforms.ToTensor(),
            ]))
        val_dataset = CelebADataset(root_dir=val_dir, transform=transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]))
        test_dataset = CelebADataset(root_dir=test_dir, transform=transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]))
        
        im_size = 64
        
    elif dataset=="ImageNet":
        train_path = root_dir+"ImageNet/training"
        val_path = root_dir+"ImageNet/validation"
        test_path = root_dir+"ImageNet/test"
        paths_exist = [os.path.exists(train_path),
                       os.path.exists(val_path),
                       os.path.exists(test_path)]
        if False in paths_exist:
            print("Download and format the dataset into the following structure:\n\
                   ImageNet/\n\
                       -> testing/\n\
                            -> *.JPEG\n\
                       -> training/\n\
                            -> *.JPEG\n\
                       -> validation/\n\
                            -> *.JPEG\n\
                                ")
            raise FileNotFoundError(f"Found? train: [{paths_exist[0]}] val: [{paths_exist[1]}] test: [{paths_exist[2]}]")

        class ImageNetDataset(Dataset):
            """Face Landmarks dataset."""
            def __init__(self, root_dir, transform=None):
                """
                Args:
                    root_dir (string): Directory with all the images.
                    transform (callable, optional): Optional transform to be applied
                """
                self.root_dir = root_dir
                self.images = next(os.walk(root_dir))[2]
                self.transform = transform

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                if torch.is_tensor(idx):
                    idx = idx.tolist()

                image = Image.open(os.path.join(self.root_dir, self.images[idx]))
                # print(image.size)
                if self.transform:
                    image = self.transform(image)
                sample = image  # target, label
                if (sample.shape[0] != 3): return None # print(self.root_dir, self.images[idx])
                # print(sample.shape)
                # sample = image
                return sample # (sample, sample)
            
        train_dataset = ImageNetDataset(root_dir=train_path, transform=transforms.Compose([
                transforms.RandomResizedCrop((256, 256)),
                # transforms.Resize((256, 256)),
                # transforms.RandomCrop(256, 4),
                transforms.ToTensor(),
            ]))
        val_dataset = ImageNetDataset(root_dir=val_path, transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]))
        test_dataset = ImageNetDataset(root_dir=test_path, transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]))
        
        im_size = 256
    
    elif dataset=="AID":
        path = "/home/kneehaw/.cache/kagglehub/datasets/jiayuanchengala/aid-scene-classification-datasets/versions/1"
        paths_exist = [os.path.exists(path)]
        if False in paths_exist:
            print("Download and format the dataset into the following structure:\n\
                   /home/kneehaw/.cache/kagglehub/datasets/jiayuanchengala/aid-scene-classification-datasets/versions/1\n\
                       -> testing/\n\
                            -> *.JPEG\n\
                       -> training/\n\
                            -> *.JPEG\n\
                       -> validation/\n\
                            -> *.JPEG\n\
                                ")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        dataset_ = ImageFolder(root=path, transform=transform)

        # Split dataset into training and validation sets
        train_idx, val_idx = train_test_split(list(range(len(dataset_))), test_size=0.2, random_state=42)
        train_dataset = torch.utils.data.Subset(dataset_, train_idx)
        val_dataset = torch.utils.data.Subset(dataset_, val_idx)
        test_dataset = None
        im_size = 256
        
    elif dataset=='flicker':
        train_transforms = transforms.Compose([
            transforms.RandomCrop(args.patch_size), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
            ]
        )

        test_transforms = transforms.Compose(
            [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
        )

        train_dataset = ImageFolder(root=root_dir+'flicker', split="train", transform=train_transforms)
        val_dataset=None
        test_dataset = ImageFolder(root=root_dir+'flicker', split="test", transform=test_transforms)
        im_size = 256
        
    else:
        raise ValueError("Incorrect dataset selection, check paramaeters")
    

    def collate_fn_IMG(batch):
            batch = list(filter(lambda x: x is not None, batch))
            return (torch.stack(batch, dim=0), torch.stack(batch, dim=0))#torch.cat([torch.zeros(1) for x in batch]))
    def collate_fn_AID(batch):
        images, labels = zip(*batch)
        # Stack the images into a batch tensor
        images = torch.stack(images)
        
        return (images, images)
    # print("\n\nTEST: ", dataset, "\n\n")
    collate_fn = collate_fn_IMG if dataset=="ImageNet" else None
    collate_fn = collate_fn_AID if dataset=="AID" else collate_fn
    # train_sampler = DistributedSampler(train_dataset)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn,
            batch_size=args.batch_size * dev_count, shuffle=False, num_workers=args.num_workers, 
            pin_memory=(device=="cuda"), drop_last=True, worker_init_fn=worker_init_fn)
    if val_dataset is None:
        val_loader = None
    else:
        val_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=collate_fn,
            batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, 
            pin_memory=(device=="cuda"), drop_last=True, worker_init_fn=worker_init_fn)
    if test_dataset is None:
        test_loader = None
    else:
        test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn,
            batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, 
            pin_memory=(device=="cuda"), drop_last=True, worker_init_fn=worker_init_fn)
        


    return train_loader, val_loader, test_loader, im_size
