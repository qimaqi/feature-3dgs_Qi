import copy
import glob
import os
import itertools
import functools
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as torch_transforms
import encoding.datasets as enc_ds
from PIL import Image

encoding_datasets = {
    x: functools.partial(enc_ds.get_dataset, x)
    for x in ["coco", "ade20k", "pascal_voc", "pascal_aug", "pcontext", "citys"]
}

class FolderLoaderpp(enc_ds.ADE20KSegmentation):#(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, selected_frames=None):
        self.root = root
        self.transform = transform
        self.images = get_folder_image_pp(root, selected_frames)
        print("self.images", self.images)
        # raise ValueError("selected_frames is not None, please check the code")
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n"))
        # self.num_class = 150  # ADE20k

    def __getitem__(self, index):
        # check rgba image
        img = Image.open(self.images[index])
        img_np = np.array(img) / 255.
        if img_np.shape[-1] == 4:
            img_np = img_np[...,:3]*img_np[...,-1:] + (1.-img_np[...,-1:])
        # go back to PIL image
        img = Image.fromarray((img_np*255).astype(np.uint8))
        img = img.convert('RGB')
        # img = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)
    

class FolderLoader(enc_ds.ADE20KSegmentation):#(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = get_folder_images(root)
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n"))
        # self.num_class = 150  # ADE20k

    def __getitem__(self, index):
        # check rgba image
        img = Image.open(self.images[index])
        img_np = np.array(img) / 255.
        if img_np.shape[-1] == 4:
            img_np = img_np[...,:3]*img_np[...,-1:] + (1.-img_np[...,-1:])
        # go back to PIL image
        img = Image.fromarray((img_np*255).astype(np.uint8))
        img = img.convert('RGB')
        # img = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)


def get_folder_image_pp(img_folder, selected_frames=None):
    # img_paths = []
    glist = []
    # glist = list(glob.glob(img_folder.rstrip("/") + '/*.png')) + list(glob.glob(img_folder.rstrip("/") + '/*.jpg')) \
    #       + list(glob.glob(img_folder.rstrip("/") + '/*.PNG')) + list(glob.glob(img_folder.rstrip("/") + '/*.JPG'))###
    for i in range(len(selected_frames)):
        glist.append(img_folder.rstrip("/") + '/' + selected_frames[i])
        

    return list(sorted(glist))


def get_folder_images(img_folder):
    img_paths = []
    glist = list(glob.glob(img_folder.rstrip("/") + '/*.png')) + list(glob.glob(img_folder.rstrip("/") + '/*.jpg')) \
          + list(glob.glob(img_folder.rstrip("/") + '/*.PNG')) + list(glob.glob(img_folder.rstrip("/") + '/*.JPG'))###

    return list(sorted(glist))


def get_dataset(name, **kwargs):
    if name in encoding_datasets:
        return encoding_datasets[name.lower()](**kwargs)
    if os.path.isdir(name):
        print("load", name, "as image directroy for FolderLoader")
        return FolderLoader(name, transform=kwargs["transform"])
    assert False, f"dataset {name} not found"


def get_original_dataset(name, **kwargs):
    if os.path.isdir(name):
        print("load", name, "as image directroy for FolderLoader")
        return FolderLoader(name, transform=kwargs["transform"])
    assert False, f"dataset {name} not found"


def get_original_dataset_pp(name, **kwargs):
    if os.path.isdir(name):
        print("load", name, "as image directroy for FolderLoader")
        return FolderLoaderpp(name, transform=kwargs["transform"], selected_frames=kwargs["selected_frames"])
    assert False, f"dataset {name} not found"



def get_available_datasets():
    return list(encoding_datasets.keys())
