from curses import keyname
import torch
import torch.nn as nn
import torchvision.models as models
activation={}
class baseline(nn.Module):
    def __init__(self):
        super(baseline,self).__init__()
        self.relu=nn.ReLU()
        self.conv1=nn.Conv2d(3,64,kernel_size=3)
        self.bn1=nn.BatchNorm2d(64)
        self.conv2=nn.Conv2d(64,128,kernel_size=3)
        self.bn2=nn.BatchNorm2d(128)
        self.conv3=nn.Conv2d(128,128,kernel_size=3)
        self.bn3=nn.BatchNorm2d(128)
        self.maxpool1=nn.MaxPool2d(kernel_size=3)
        self.conv4=nn.Conv2d(128,128,kernel_size=3,stride=2)
        self.bn4=nn.BatchNorm2d(128)
        self.adp_avg=nn.AdaptiveAvgPool2d((1,1))
        self.fc1=nn.Linear(128,128)
        self.drop=nn.Dropout()
        self.fc2=nn.Linear(128,20)
    def forward(self,x):
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.relu(self.bn2(self.conv2(x)))
        x=self.maxpool1(self.relu(self.bn3(self.conv3(x))))
        x=self.relu(self.bn4(self.conv4(x)))
        x=self.adp_avg(x)
        x=x.view(-1, 128) 
        x=self.fc1(x)
        x=self.drop(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x

class custom(nn.Module):
    def __init__(self):
    #experiment for edit layers
        super(custom,self).__init__()
        self.relu=nn.ReLU()
        self.conv1=nn.Conv2d(3,64,kernel_size=3)
        self.bn1=nn.BatchNorm2d(64)
        self.conv2=nn.Conv2d(64,128,kernel_size=3)
        self.bn2=nn.BatchNorm2d(128)
        self.conv2_5=nn.Conv2d(128,168,kernel_size=3)
        self.bn2_5=nn.BatchNorm2d(168)
        self.conv3=nn.Conv2d(168,168,kernel_size=3)
        self.bn3=nn.BatchNorm2d(168)
        self.conv3_5=nn.Conv2d(168,168,kernel_size=3)
        self.bn3_5=nn.BatchNorm2d(168)
        self.maxpool1=nn.MaxPool2d(kernel_size=3)
        self.maxpool2=nn.MaxPool2d(kernel_size=3)
        self.conv4=nn.Conv2d(168,128,kernel_size=3,stride=2)
        self.bn4=nn.BatchNorm2d(128)
        self.adp_avg=nn.AdaptiveAvgPool2d((1,1))
        self.drop=nn.Dropout()
        self.fc1=nn.Linear(128,20)
    #experiment try different augmentation
        # super(custom,self).__init__()
        # self.relu=nn.ReLU()
        # self.conv1=nn.Conv2d(3,64,kernel_size=3)
        # self.bn1=nn.BatchNorm2d(64)
        # self.conv2=nn.Conv2d(64,128,kernel_size=3)
        # self.bn2=nn.BatchNorm2d(128)
        # self.conv3=nn.Conv2d(128,128,kernel_size=3)
        # self.bn3=nn.BatchNorm2d(128)
        # self.maxpool1=nn.MaxPool2d(kernel_size=3)
        # self.conv4=nn.Conv2d(128,128,kernel_size=3,stride=2)
        # self.bn4=nn.BatchNorm2d(128)
        # self.adp_avg=nn.AdaptiveAvgPool2d((1,1))
        # self.fc1=nn.Linear(128,128)
        # self.drop=nn.Dropout()
        # self.fc2=nn.Linear(128,20)
    def forward(self,x):
    #experiment for edit layers
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.maxpool1(self.relu(self.bn2(self.conv2(x))))
        x=self.relu(self.bn2_5(self.conv2_5(x)))
        x=self.maxpool2(self.relu(self.bn3(self.conv3(x))))
        x=self.relu(self.bn3_5(self.conv3_5(x)))
        x=self.drop(x)
        x=self.relu(self.bn4(self.conv4(x)))
        x=self.adp_avg(x)
        x=x.view(-1, 128) 
        x=self.fc1(x)
        return x
     #experiment try different augmentation
        # x=self.relu(self.bn1(self.conv1(x)))
        # x=self.relu(self.bn2(self.conv2(x)))
        # x=self.maxpool1(self.relu(self.bn3(self.conv3(x))))
        # x=self.relu(self.bn4(self.conv4(x)))
        # x=self.adp_avg(x)
        # x=x.view(-1, 128) 
        # x=self.fc1(x)
        # x=self.drop(x)
        # x=self.relu(x)
        # x=self.fc2(x)
        # return x
class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        i=0
        resnet_temp=models.resnet18(pretrained=True)
        self.resnet_modify=torch.nn.Sequential(*(list(resnet_temp.children())[:-1]))
        for param in self.resnet_modify.parameters():
            if(i!=2):
                param.requires_grad = False
            i+=1
        self.fc=nn.Linear(512,20)
        
    def forward(self,x):
        x=self.resnet_modify(x)
        x=x.view(-1, 512) 
        x=self.fc(x)
        return x

def get_activation(name):
    def hook(model,input,output):
        activation[name]=output.detach()
    return hook
    
class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.model=models.vgg16(pretrained=True)
        rm=list(self.model.classifier[:-1])
        self.model.classifier=torch.nn.Sequential(*rm)
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc=nn.Linear(4096,20)    
    def forward(self,x):
        x=self.model(x)
        x=x.view(-1, 4096) 
        x=self.fc(x)
        return x



def get_model(args):
    model = None
    return model
