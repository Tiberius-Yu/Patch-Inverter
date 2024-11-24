import os
import glob
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from io import BytesIO

from PIL import Image
from torchvision import transforms, utils
import lmdb


class MyDataSet(data.Dataset):

    def __init__(self, path=None, resolution=(256, 256), train=True, test_num=300, split=0.9):
        self.path = path
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.resize = transforms.Compose([transforms.Resize(resolution), transforms.ToTensor()])
        self.random_rotation = transforms.Compose([
            transforms.Resize(resolution),
            transforms.RandomPerspective(distortion_scale=0.05, p=1.0),
            transforms.ToTensor()
        ])

        # load image file
        train_len = None
        self.length = 0
        self.path = path
        if path is not None:
            img_list = glob.glob(self.path + "**/*.jpg", recursive=True)
            img_list.extend(glob.glob(self.path + "**/*.png", recursive=True))
            img_list.extend(glob.glob(self.path + "**/*.JPEG", recursive=True))
            # breakpoint()
            # image_list = [item for sublist in img_list for item in sublist]
            image_list = img_list
            image_list.sort()
            train_len = int(split * len(image_list))
            if train:
                self.image_list = image_list[:train_len]
            else:
                self.image_list = image_list[train_len:]
            print(f"[INFO]: Images loaded, using {len(self.image_list)} of {len(img_list)} images")
            self.length = len(self.image_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = None
        if self.path is not None:
            img_name = os.path.join(self.path, self.image_list[idx])
            image = Image.open(img_name).convert('RGB')
            img = self.resize(image)
            if img.size(0) == 1:
                img = torch.cat((img, img, img), dim=0)
            img = self.normalize(img)

        # generate image
        return img


class TestDataSet(data.Dataset):

    def __init__(self, path=None, resolution=(256, 256), train=False, test_num=None, split=0.9):
        self.path = path
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.resize = transforms.Compose([transforms.Resize(resolution), transforms.ToTensor()])
        self.random_rotation = transforms.Compose([
            transforms.Resize(resolution),
            transforms.RandomPerspective(distortion_scale=0.05, p=1.0),
            transforms.ToTensor()
        ])

        # load image file
        train_len = None
        self.length = 0
        self.path = path
        if path is not None:
            img_list = glob.glob(self.path + "**/*.jpg", recursive=True)
            img_list.extend(glob.glob(self.path + "**/*.png", recursive=True))
            img_list.extend(glob.glob(self.path + "**/*.JPEG", recursive=True))
            # breakpoint()
            # image_list = [item for sublist in img_list for item in sublist]
            image_list = img_list
            image_list.sort()
            n = len(image_list)
            used_len = n if test_num is None else test_num
            self.image_list = image_list[::n // used_len]
            print(f"[INFO]: Images loaded, using {len(self.image_list)} of {len(img_list)} images")
            self.length = len(self.image_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = None
        if self.path is not None:
            img_name = os.path.join(self.path, self.image_list[idx])
            image = Image.open(img_name).convert('RGB')
            img = self.resize(image)
            if img.size(0) == 1:
                img = torch.cat((img, img, img), dim=0)
            img = self.normalize(img)

        # generate image
        return img


class MultiResolutionDataset(Dataset):

    def __init__(self, path=None, split=None, train=True, test_num=None, resolution=None):
        resolution = (1024, 1024) if resolution is None else resolution
        self.path = os.path.join(path)
        self.env = lmdb.open(
            self.path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.resolution = resolution[0] if isinstance(resolution, tuple) else resolution
        with self.env.begin(write=False) as txn:
            length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
            print(" [*] Loaded data with length {}".format(length))

        split_num = int(split * length)
        if train:
            self.length = split_num
            self.st_pt = 0
            print(" [*] Using length {} to train".format(self.length))
        else:
            self.length = test_num if test_num is not None else length - split_num
            self.st_pt = split_num
            print(" [*] Using length {} to test".format(self.length))

        if train:
            self.transform = transforms.Compose([  # Center crop for fare comparison
                transforms.Resize(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(resolution),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])
        self.n_zfill = 8

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        try:
            with self.env.begin(write=False) as txn:
                key = f'{self.resolution}-{str(self.st_pt + index).zfill(self.n_zfill)}'.encode(
                    'utf-8')
                img_bytes = txn.get(key)
            buffer = BytesIO(img_bytes)
            if buffer is None:
                raise ValueError(" [!] Meet empty image while loading with key {}".format(key))
            full_img = Image.open(buffer)
        except Exception as e:
            print(" [!] Error at idx {}".format(index))
            raise e
        full_img = self.transform(full_img)

        return full_img
