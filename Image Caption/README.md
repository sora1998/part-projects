## Introduction
This repository contains the code and models for a project focused on exploring different recurrent neural network (RNN) architectures for image captioning. In this project, we developed three distinct RNN models, each with the same image encoding backbone based on the ResNet50 model, followed by custom modifications. The goal was to investigate the impact of different RNN architectures on image captioning performance.

### Project Overview
In this project, we built three RNN models with the following key components:
Image Encoding: All three models employ the ResNet50 model to encode input images. We removed the last fully connected (fc) layer of ResNet50 and replaced it with a custom 2048-dimensional trainable fc layer.
### Model Architectures

#### Baseline Model:
Image Encoder: ResNet50 + Custom fc layer
Decoder:
Embedding Layer: Projects words to vectors
LSTM Layer: Captures image-caption relationships
Fully Connected Layer: Transforms LSTM output to vocabulary size

#### Vanilla RNN Model:
Image Encoder: ResNet50 + Custom fc layer
Decoder:
Embedding Layer: Projects words to vectors
Vanilla RNN Layer: Replaces LSTM for sequence modeling
Fully Connected Layer: Transforms RNN output to vocabulary size

#### Architecture 2 (A2) Model:
Image Encoder: ResNet50 + Custom fc layer
Decoder:
Image and Hidden State Fusion: Combines the hidden state and image features as input to the LSTM layer
LSTM Layer: Utilizes the combined input for sequence modeling
Fully Connected Layer: Transforms LSTM output to vocabulary size

### Model Performance
The project evaluated the performance of these three RNN architectures using BLEU1 and BLEU4 metrics. Here are the results:

#### Baseline Model:
BLEU1 Score: 66
BLEU4 Score: 8.2

#### Vanilla RNN Model:
BLEU1 Score: 65.7
BLEU4 Score: 7.85

#### Architecture 2 (A2) Model:
BLEU1 Score: 67.5
BLEU4 Score: 7.8

### Conclusion
Based on the evaluation results, it can be concluded that our models were successful in generating image captions. The Architecture 2 (A2) model outperformed the others slightly in terms of BLEU1 score, suggesting that combining image features and hidden states can lead to improved caption quality. However, all three models demonstrated promising performance, showcasing the potential of different RNN architectures in image captioning tasks.Feel free to explore the code and trained models in this repository to gain a deeper understanding of each architecture and further experiment with image captioning tasks
