import torch
import os
import pandas as pd
from PIL import Image
from vocab import Vocab
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from utils import collate_fn

class Flickr8kDataset(Dataset):
    def __init__(self, txt_file, image_folder, max_length):
        self.data = pd.read_csv(txt_file)
        self.image_folder = image_folder
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
        ])
        self.vocab = Vocab(self.data["caption"], max_length= max_length)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_folder, self.data["image"][index])
        image = Image.open(img_path)
        image = self.transform(image)

        caption = self.data["caption"][index]
        caption_ids = self.vocab.convert_tokens_to_ids(caption)

        return {
            "image_path": img_path,
            "image": image,
            "caption": caption,
            "caption_ids": caption_ids
        }
    
    def __len__(self):
        return len(self.data)
    
class Load_Data:
    def __init__(self, config):
        self.txt_file = config["data"]["txt_file"]
        self.image_folder = config["data"]["image_folder"]
        self.max_length = config["text_embedding"]["max_length"]

        self.dataset = Flickr8kDataset(self.txt_file, self.image_folder, self.max_length)

        self.train, self.dev, self.test = random_split(self.dataset, [0.8, 0.1, 0.1])

    def load_train_dev(self):
        train_loader = DataLoader(
            self.train,
            batch_size= 128,
            shuffle= True,
            collate_fn= collate_fn
        )

        dev_loader = DataLoader(
            self.dev,
            batch_size= 128,
            shuffle= False,
            collate_fn= collate_fn
        )
        return train_loader, dev_loader
    
    def load_test(self):
        test_loader = DataLoader(
            self.test,
            batch_size= 128,
            shuffle= False,
            collate_fn= collate_fn
        )
        return test_loader