from urllib.request import DataHandler
import torch
import argparse
import os, sys, json
from datetime import datetime
from data import *
from engine import *
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()

parser.add_argument('--log', default=1, type=int,
                    help='Determine if we log the outputs and experiment configurations to local disk')
parser.add_argument('--path', default=datetime.now().strftime('%Y-%m-%d-%H%M%S'), type=str,
                    help='Default log output path if not specified')
parser.add_argument('--device_id', default=0, type=int,
                    help='the id of the gpu to use')    
# Model Related
parser.add_argument('--model', default='vgg', type=str,
                    help='Model being used')
parser.add_argument('--pt_ft', default=1, type=int,
                    help='Determine if the model is for partial fine-tune mode')
parser.add_argument('--model_dir', default=None, type=str,
                    help='Load some saved parameters for the current model')
parser.add_argument('--num_classes', default=20, type=int,
                    help='Number of classes for classification')

# Data Related
parser.add_argument('--bz', default=32 , type=int,
                    help='batch size')
parser.add_argument('--shuffle_data', default=True, type=bool,
                    help='Shuffle the data')
parser.add_argument('--normalization_mean', default=(0.485, 0.456, 0.406), type=tuple,
                    help='Mean value of z-scoring normalization for each channel in image')
parser.add_argument('--normalization_std', default=(0.229, 0.224, 0.225), type=tuple,
                    help='Mean value of z-scoring standard deviation for each channel in image')
parser.add_argument('--augmentation', default=0, type=int)

# feel free to add more augmentation/regularization related arguments

# Other Choices & hyperparameters
parser.add_argument('--epoch', default=25, type=int,
                    help='number of epochs')
    # for loss
parser.add_argument('--criterion', default='cross_entropy', type=str,
                    help='which loss function to use')
    # for optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    help='which optimizer to use')
parser.add_argument('--lr', default=0.95e-3, type=float,
                    help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--dampening', default=0, type=float,
                    help='dampening for momentum')
parser.add_argument('--nesterov', default=False, type=bool,
                    help='enables Nesterov momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay')
# for scheduler
parser.add_argument('--lr_scheduler', default='steplr', type=str,
                    help='learning rate scheduler')
parser.add_argument('--step_size', default=7, type=int,
                    help='Period of learning rate decay')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Multiplicative factor of learning rate decay.')

# feel free to add more arguments if necessary

args = vars(parser.parse_args())

def main(args):
    device = torch.device("cuda:{}".format(args['device_id']) if torch.cuda.is_available() else "cpu")
    dataloaders = get_dataloaders('food-101/train.csv', 'food-101/test.csv', args)
    model,criterion,optimizer = prepare_model(device, args)
    model,train_loss,train_acc,val_loss,val_acc = train_model(model, criterion, optimizer, device, dataloaders, args)
    plt.plot(train_loss,label="train")
    plt.plot(val_loss,label="val")
    plt.title("Loss")
    plt.legend()
    plt.show()
    plt.plot(train_acc,label="train")
    plt.plot(val_acc,label="val")
    plt.title("Trian/Val Accuracy")
    plt.legend()
    plt.show()
    test_loss,test_acc=test_model(model,device,dataloaders[2],args)
    print("Best Model Test Loss:",test_loss)
    print("Best Model Test Accuracy:",test_acc*100)
def print_weight():
    device = torch.device("cuda:{}".format(args['device_id']) if torch.cuda.is_available() else "cpu")
    dataloaders = get_dataloaders('food-101/train.csv', 'food-101/test.csv', args)
    model,criterion,optimizer = prepare_model(device, args)

    if args["model"]=="custom":
        model.load_state_dict(torch.load("best_7"))
        filters=model.conv1.weight.data.clone()
        print(filters.size())
        # plot first few filters
        columns=8
        rows=8
        n_filters, ix = 64,1
        for i in range(1,n_filters+1):
            # get the filter
            f = filters[i-1, :, :, :]
            f = f.swapaxes(0, 1)
            f = f.swapaxes(1, 2)
            image=plt.subplot(rows,columns,i)
            image.set_xticks([])
            image.set_yticks([])
            plt.imshow(f)
        # show the figure
        plt.show()
    elif args["model"]=="resnet":
        model.load_state_dict(torch.load("best_5"))
        filters=model.resnet_modify[0].weight
        print(filters.size())
        # plot first few filters
        columns=8
        rows=8
        for param in model.resnet_modify.parameters():
             param.requires_grad = False
        n_filters, ix = 64,1
        for i in range(1,n_filters+1):
            # get the filter
            f = filters[i-1, :, :, :]
            f = f.swapaxes(0, 1)
            f = f.swapaxes(1, 2)
            image=plt.subplot(rows,columns,i)
            image.set_xticks([])
            image.set_yticks([])
            plt.imshow(f)
        # show the figure
        plt.show()
    elif args["model"]=="vgg":
        model.load_state_dict(torch.load("best_6"))
        filters= model.model.features[0].weight
        print(filters.size())
        # plot first few filters
        columns=8
        rows=8
        for param in model.model.parameters():
             param.requires_grad = False
        n_filters, ix = 64,1
        for i in range(1,n_filters+1):
            # get the filter
            f = filters[i-1, :, :, :]
            f = f.swapaxes(0, 1)
            f = f.swapaxes(1, 2)
            image=plt.subplot(rows,columns,i)
            image.set_xticks([])
            image.set_yticks([])
            plt.imshow(f)
        # show the figure
        plt.show()
def print_map():
    device = torch.device("cuda:{}".format(args['device_id']) if torch.cuda.is_available() else "cpu")
    dataloaders = get_dataloaders('food-101/train.csv', 'food-101/test.csv', args)
    model,criterion,optimizer = prepare_model(device, args)
    train,val,test=dataloaders
    for data,target in test:
            data=data[0]
            break
    data=data.view(1,3,224,224)
    if args["model"]=="custom":
        model.load_state_dict(torch.load("best_7"))
        model.conv1.register_forward_hook(get_activation('conv1'))
        model.conv3.register_forward_hook(get_activation('conv3'))
        model.conv4.register_forward_hook(get_activation('conv4'))
        output=model(data)
        act=[]
        act.append(activation['conv1'])
        act.append(activation['conv3'])
        act.append(activation['conv4'])
    if args["model"]=="resnet":
        model.load_state_dict(torch.load("best_5"))
        model.resnet_modify[0].register_forward_hook(get_activation('conv1'))
        model.resnet_modify[-3].register_forward_hook(get_activation('conv2'))
        model.resnet_modify[-2].register_forward_hook(get_activation('conv3'))
        output=model(data)
        act=[]
        act.append(activation['conv1'])
        act.append(activation['conv2'])
        act.append(activation['conv3'])
    elif args["model"]=="vgg":
        model.load_state_dict(torch.load("best_6"))
        model.model.features[0].register_forward_hook(get_activation('conv1'))
        model.model.features[-15].register_forward_hook(get_activation('conv2'))
        model.model.features[-2].register_forward_hook(get_activation('conv3'))
        output=model(data)
        act=[]
        act.append(activation['conv1'])
        act.append(activation['conv2'])
        act.append(activation['conv3'])
    # plot all 64 maps in an 8x8 squares
    square = 8
    for index in range(len(act)):
        act[index] = act[index].swapaxes(1,2)
        act[index] = act[index].swapaxes(2,3)
    for j in range(3):
        ix = 1
        for i in range(square):
            for k in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(act[j][0,:, :, ix-1], cmap='gray')
                ix += 1
        # show the figure
        plt.show()
if __name__ == '__main__':
    #print_map()
    main(args)
