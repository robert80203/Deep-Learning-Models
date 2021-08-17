from torch.utils.data.dataset import Dataset
import json
from PIL import Image
from torchvision import datasets, transforms
import torch

class iclevr_dataset(Dataset):
    def __init__(self, datapath, mode='gan'):

        self.datadir = datapath #'../../progressive_gan/Text_to_image_generation/progressive_gan/iclevr/GeNeVA-v1/'
        with open('data/dicts/iclevr_object_map.json','r') as fp:
            self.name2objects = json.load(fp)
            #print(self.name2objects)
            self.name = list(self.name2objects)
            if mode == 'gan':
                newname = list()
                for n in self.name:
                    if "_0.png" in n or "_1.png" in n or "_2.png" in n:
                        newname.append(n)
                self.name = newname
                self.maxlen = 3
            elif mode == 'classifier':
                self.maxlen = 5
        with open('data/dicts/iclevr_objects.json','r') as fp:
            self.labels2idx = json.load(fp)
        self.image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        print(self.labels2idx)
    def __getitem__(self, index):
        onehot_labels = torch.zeros((len(self.labels2idx)))
        k = len(self.name2objects[self.name[index]])
        labels = torch.zeros(self.maxlen)+24
        
        image = Image.open(self.datadir+self.name[index]).convert('RGB')
        image = self.image_transform(image)
        
        for i in self.name2objects[self.name[index]]:
            onehot_labels[self.labels2idx[i]] = 1
        for i in range(k):
            obj = self.name2objects[self.name[index]][i]
            labels[i] = self.labels2idx[obj]
        return image, onehot_labels, labels
    def __len__(self):
        return len(self.name)

class test_dataset(Dataset):
    def __init__(self, name):
        with open('data/dicts/'+name+'.json','r') as fp:
            self.data = json.load(fp)
        self.maxlen = 3
        with open('data/dicts/iclevr_objects.json','r') as fp:
            self.labels2idx = json.load(fp)
    def __getitem__(self, index):
        onehot_labels = torch.zeros(24)
        labels = torch.zeros(self.maxlen)+24
        idx = 0
        for i in self.data[index]:
            obj = self.labels2idx[i]
            onehot_labels[obj] = 1
            labels[idx] = obj
            idx += 1
        return onehot_labels, labels
    def __len__(self):
        return len(self.data)