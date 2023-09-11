
The experiment data file is the file we use to save the experiment data we got from training the model. There are different experiment data stored in different files. 
dataset_factory is the file we use to get and separate the data. default is the config set file we use for the model. 

The experiment.py is the file we create an experiment file with different functions like train, test, plot and etc. simply please just use run(train the model), test(test the model),plot_good_bad_example(plot 1 good and 1 bad examples), and load_experiment(load previous model).

The model_factory.py is where we define three different models for this project one is the baseline, one is RNN and one is A2(which is architecture 2). The only thing unique is the sample function which will produce sentences by one image.

How to use this project is pretty much to edit the main.py and default.json. 

If you want to use the baseline model you can just simply set the argument in json. Experiment name to baseline and model to baseline. Or you can just pass them as argument variables through command. Then run main.py with exp.load_experiment() exp.run() and exp.test(), then it trying to load the last baseline experiment, if you do have previous experiment please remove exp.run() or it will report error. It will either train the model or load the model depend on depend on exp.run() , it will print the loss graph and test performance. The method to change the model is pretty simple which is just the change Experiment name   and model type. The experiment can be any name but please avoid the same experiment name with a different model setting. We have a total of 3 models that can be set as the model name described above. If you want to print 1 good and 1 bad image captions, please just call the function exp.plot_good_bad_example it will choose 1 good examples with a bleu1 score>80 and 1 bad examples which bleu1 score<45. If you want to create a new model with same experiment name please delete the original file for that experiment or it will print an error message. We also provide a function that can print the caption by image_id.

