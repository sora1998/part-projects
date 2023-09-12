from numpy import genfromtxt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch

batch_size=64
########## DO NOT change this function ##########
# If you change it to achieve better results, we will deduct points. 
def train_val_split(train_dataset):
    train_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], 
                                            generator=torch.Generator().manual_seed(42))
    return train_subset, val_subset
#################################################

########## DO NOT change this variable ##########
# If you change it to achieve better results, we will deduct points. 
transform_test = transforms.Compose([
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
transform_train = transforms.Compose([
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  #mean and std I found
  #transforms.Normalize(mean=[0.5439, 0.4397, 0.3346], std=[0.2752, 0.2751, 0.2765])]
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  #,transforms.RandomHorizontalFlip()
  ]
)
#################################################

class FoodDataset(Dataset):
    def __init__(self, data_csv, transforms=None):
        self.data = genfromtxt(data_csv, delimiter=',', dtype=str)
        self.transforms = transforms
        
    def __getitem__(self, index):
        fp, _, idx = self.data[index]
        idx = int(idx)
        img = Image.open(fp)
        if self.transforms is not None:
            img = self.transforms(img)
        return (img, idx)

    def __len__(self):
        return len(self.data)

def get_dataset(csv_path, transform):
    return FoodDataset(csv_path, transform)

def data_mean_var(dataloader):
    e =0
    e_s=0
    k=0
    for batch_idx, (data,target) in enumerate(dataloader):
        #print(batch_idx)
        k+=data.shape[0]*data.shape[2]*data.shape[3]
        e+=data.sum(axis=[0,2,3])
        e_s+=(data**2).sum(axis=[0,2,3])
    mean=e/k
    var=(e_s/k)-(mean**2)
    print('pixel mean:', mean, 'pixel std:', (var**0.5))

def create_dataloaders(train_set, val_set, test_set, args=None):
    train_loader=DataLoader(train_set,batch_size=args["bz"],shuffle=True,num_workers=4)
    #data_mean_var(train_loader)
    val_loader=DataLoader(val_set,batch_size=args["bz"],shuffle=True,num_workers=4)
    test_loader=DataLoader(test_set,batch_size=args["bz"],shuffle=False,num_workers=4)
    return train_loader,val_loader,test_loader
def get_dataloaders(train_csv, test_csv, args=None):
    train_dataset=get_dataset(train_csv,transform_train)
########## DO NOT change the following two lines ##########
# If you change it to achieve better results, we will deduct points. 
    test_dataset = get_dataset(test_csv, transform_test)
    train_set, val_set = train_val_split(train_dataset)
###########################################################
    return create_dataloaders(train_set,val_set,test_dataset,args)