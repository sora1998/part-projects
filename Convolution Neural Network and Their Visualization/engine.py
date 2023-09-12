# Modified by Colin Wang, Weitang Liu

from cgitb import reset
import torch
import gc
from model import *
import numpy as np
def prepare_model(device, args=None):
    #load model, criterion, optimizer, and learning rate scheduler
    if(args["model"]=="baseline"):
        model=baseline()
    elif(args["model"]=="custom"):
        model=custom()
    elif(args["model"]=="resnet"):
        model=resnet()
    elif(args["model"]=="vgg"):
        model=vgg()
    lr=args["lr"]
    m=args["momentum"]
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    criterion=torch.nn.CrossEntropyLoss()
    return model, criterion, optimizer
    raise NotImplementedError()

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data, mean=0.0, std=1.0)
def train_model(model,criterion,optimizer, device,dataloaders, args=None):
    train,val,test=dataloaders
    count=0
    avg_train_loss=[]
    avg_val_loss=[]
    test_accuracy_list=[]
    val_acc=[]
    train_acc=[]
    if( (args["model"]=="baseline") | (args["model"]=="custom")):
        model.apply(weights_init)
    model.to(device)
    model.train()
    for e in range(args["epoch"]):
        train_loss=[]
        best_val=float('inf')
        val_loss=[]
        correct_train=0
        correct_val=0
        for batch_idx,(data,target) in enumerate(train):
            #target=torch.nn.functional.one_hot(target,20)
            #print("target",target.size())
            data,target=data.to(device),target.to(device)
            optimizer.zero_grad()
            output=model(data)
            #print("output",output.size())
            loss=criterion(output,target)
            loss.backward()
            optimizer.step()
            pred =torch.argmax(output,dim=1, keepdim=True)
            #target=torch.argmax(target,dim=1, keepdim=True)
            correct_train+=pred.eq(target.view_as(pred)).sum().item()
            train_loss.append(loss.item())
        with torch.no_grad():
            for batch_idx,(data,target) in enumerate(val):
                #target=torch.nn.functional.one_hot(target,20)
                data,target=data.to(device),target.to(device)
                output=model(data)
                loss=criterion(output,target)
                pred = torch.argmax(output,dim=1, keepdim=True) # get the index of maximum fc output. Q4. Why?
                #target=torch.argmax(target,dim=1, keepdim=True)
                correct_val += pred.eq(target.view_as(pred)).sum().item()
                val_loss.append(loss.item())
        train_acc.append(correct_train/len(train.dataset))
        val_acc.append(correct_val/len(val.dataset))
        avg_train_loss.append(np.average(train_loss))
        avg_val_loss.append(np.average(val_loss))
        #"best_int" is the file we use to store the weight for each model but we can't submit because the size is too large
        if(avg_val_loss[-1]<best_val):
            best_val=avg_val_loss[-1]
            torch.save(model.state_dict(),"best_4")
        # if(avg_val_loss[-1]>best_val):
        #     count+=1
        # if(count==3):
        #     break
        torch.cuda.empty_cache()
    model.load_state_dict(torch.load("best_4"))
    return model,avg_train_loss,train_acc,avg_val_loss,val_acc
def test_model(model,device, test_loader, args=None):
    model.to(device)
    model.eval()
    test_loss=[]
    correct=0
    criterion=torch.nn.CrossEntropyLoss()
    with torch.no_grad(): # stop storing gradients for the variables
        for data,target in test_loader:
            #target=torch.nn.functional.one_hot(target,20)
            data,target = data.to(device),target.to(device)
            output = model(data)
            loss=criterion(output,target)
            #target=torch.argmax(target,dim=1, keepdim=True)
            pred = torch.argmax(output,dim=1, keepdim=True) # get the index of maximum fc output. Q4. Why?
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss.append(loss.item())
    test_loss=np.average(test_loss)
    return test_loss,correct/len(test_loader.dataset)
        


    return model # return the model with weight selected by best performance 

# add your own functions if necessary
