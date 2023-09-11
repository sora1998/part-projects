################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torch.nn as nn
from datetime import datetime
from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
import nltk
from model_factory import get_model
from torch.nn.utils.rnn import pack_padded_sequence
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.

        # Init Model
        self.__model = get_model(config_data, self.__vocab)

        # TODO: Set these Criterion and Optimizers Correctly
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(),lr=config_data['experiment']['learning_rate'])
        self.__init_model()

        # Load Experiment Data if available
#         self.__load_experiment

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)
            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.to(device)
            self.__criterion = self.__criterion.to(device)

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        count=3
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            
            if(len(self.__val_losses)!=0):
                if(val_loss>self.__val_losses[-1]):
                    count=count-1
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            torch.cuda.empty_cache()
            
            if count==0:
                break
        self.__save_model()

    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__model.train()
        training_loss = []
        
        # Iterate over the data, implement the training function
        for i, (images, captions, _) in enumerate(self.__train_loader):
#             print(images.shape)
#             print(captions.shape)
            images = images.to(device)
            captions = captions.to(device)
            outputs = self.__model(images, captions)
            loss = self.__criterion(outputs.reshape(-1,outputs.shape[2]), captions.reshape(-1))
            training_loss.append(loss.item())
            self.__optimizer.zero_grad()
            loss.backward()     
            self.__optimizer.step()
        return np.average(training_loss)

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.eval()
        val_loss = []
        with torch.no_grad():
            for i, (images, captions, _) in enumerate(self.__val_loader):
                images = images.to(device)
                captions = captions.to(device)
                outputs = self.__model(images, captions)
                loss = self.__criterion(outputs.reshape(-1,outputs.shape[2]), captions.reshape(-1))
                val_loss.append(loss.item())
        avg=np.average(val_loss)
        return avg

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        self.__model.eval()
        test_loss = []
        bleu1_s = []
        bleu4_s = []
        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                images = images.to(device)
                captions = captions.to(device)
#                 print(captions.shape)
                for k in range(images.shape[0]):
                    image=images[k]
                    image=image.reshape((1,3,256,256))
                    caption=captions[k]
                    caption=caption.reshape(1,-1)
                    outputs = self.__model(image, caption)
                    loss = self.__criterion(outputs.reshape(-1,outputs.shape[2]), caption.reshape(-1))
                    ann=self.__coco_test.imgToAnns[img_ids[k]]
                    cap=[a['caption'] for a in ann]
                    token_cap=[nltk.tokenize.word_tokenize(c.lower()) for c in cap]
                    sentence = self.__model.sample(image)
#                     print("caption: ", token_cap)
#                     print("sentence: ",sentence)
#                     print("bleu1: ",bleu1(token_cap,sentence))
#                     print("bleu4 ",bleu4(token_cap,sentence))
                    bleu1_s.append(bleu1(token_cap,sentence))
                    bleu4_s.append(bleu4(token_cap,sentence))
                    test_loss.append(loss.item())
        bleu1_s=np.average(bleu1_s)
        bleu4_s=np.average(bleu4_s)
        test_loss=np.average(test_loss)
        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss,
                                                                                               bleu1_s,
                                                                                               bleu4_s)
        self.__log(result_str)
        return test_loss, bleu1, bleu4

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__best_model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)
#         print(len(self.__val_losses)==1)
        if(len(self.__val_losses)==1):
            self.__best_model=self.__model
            self.__save_model()
        if(val_loss<np.min(self.__val_losses)):
            self.__best_model=self.__model
            self.__save_model()
#         print("train: ", self.__training_losses)
#         print("val: ", self.__val_losses)
        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
        
    def plot_good_bad_example(self):
        self.__model.eval()
        bleu1_s=0
        good=0
        bad=0
        bleu4_s=0
        list2=[]
        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                images = images.to(device)
                captions = captions.to(device)
                for k in range(images.shape[0]):
                    image=images[k]
                    image=image.reshape((1,3,256,256))
                    ann=self.__coco_test.imgToAnns[img_ids[k]]
                    cap=[a['caption'] for a in ann]
                    token_cap=[nltk.tokenize.word_tokenize(c.lower()) for c in cap]
                    sentence = self.__model.sample(image)
#                     print("caption: ", token_cap)
#                     print("sentence: ",sentence)
#                     print("bleu1: ",bleu1(token_cap,sentence))
#                     print("bleu4 ",bleu4(token_cap,sentence))
                    bleu1_s=bleu1(token_cap,sentence)
                    bleu4_s=bleu4(token_cap,sentence)
                    plt.figure()
                    show=image.cpu()
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
                    show = show * std + mean
                    if (bleu1_s>80) and (good<3):
                        show=show.permute(0,2,3,1)
                        good+=1
                        plt.imshow(show.squeeze(0))
                        list2.append(img_ids[k])
                        print('good ',good)
                        print('actual captions:', cap)
                        print('predict captions:', sentence)
                        print("Test Performance: Bleu1: {}, Bleu4: {}".format(
                                                                                               bleu1_s,
                                                                                               bleu4_s))
                        print('\n')
                    if (bleu1_s<45) and (bad<3):
                        show=show.permute(0,2,3,1)
                        bad+=1
                        plt.imshow(show.squeeze(0))
                        list2.append(img_ids[k])
                        print('bad ',bad)
                        print('actual captions:', cap)
                        print('predict captions:',sentence)
                        print("Test Performance: Bleu1: {}, Bleu4: {}".format(
                                                                                               bleu1_s,
                                                                                               bleu4_s))
                        print('\n')
                    if (bad==3) and (good==3):
                        plt.show()
                        break
                if (bad==3) and (good==3):
                    break
        print("id list: ", list2)
    def caption_id(self,id_list):
        self.__model.eval()
        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                images = images.to(device)
                captions = captions.to(device)
                for k in range(images.shape[0]):
                    if(img_ids[k] in id_list):
                        image=images[k]
                        image=image.reshape((1,3,256,256))
                        sentence = self.__model.sample(image)
                        print(img_ids[k]," : ",sentence)
                        id_list.remove(img_ids[k])
                        if(len(id_list)==0):
                            break
                if(len(id_list)==0):
                    break