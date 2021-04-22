from output_model.converted_pytorch import KitModel
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch.utils import data
import pandas as pd
import os
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x
import imp


transforms = T.Compose([T.ToTensor()])
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MainModel = imp.load_source('MainModel', "./output_model/converted_pytorch.py")

img_dir = 'test_img/val_img'
csv_path = 'test_img/val_img.csv'
# inc_v3 = KitModel('./output_model/converted_pytorch.npy')
inc_v3 = torch.load('./output_model/converted_pytorch.pth')
# print(inc_v3)


class ImageNet(data.Dataset):
    def __init__(self, dir, csv_path, transforms = None):
        self.dir = dir 
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms
    
    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['filename']
        Truelabel = img_obj['label']  
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            data = pil_img
        return data, ImageID, Truelabel

    def __len__(self):
        return len(self.csv)

inputs = ImageNet(img_dir, csv_path, transforms)
data_loader = DataLoader(inputs, batch_size=30, shuffle=False, pin_memory=True, num_workers=8)
sum_num = 0
inc_v3 = inc_v3.cuda().eval()

for images, name, labels in tqdm(data_loader):
    gt = labels.cuda()
    images = images.cuda()
    images = images*2.0-1.0  # convert [0,1] to [-1,1]
    with torch.no_grad():
        logits = inc_v3(images)
        sum_num += (torch.argmax(logits, axis=1) == gt).detach().sum().cpu()

print('inc_v3:',sum_num/10)

