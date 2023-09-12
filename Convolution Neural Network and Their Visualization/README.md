## Convolutional Neural Networks (CNNs) for Food-101 Classification
### Introduction
Welcome to our project repository, where we delve into the world of convolutional neural networks (CNNs) to tackle the Food-101 classification task. In this project, we've implemented four distinct CNN models, each with its own unique characteristics and performance. Our objective is to understand the impact of various architectural changes and strategies on the models' accuracy.

### Model Overview
#### 1. Baseline Model
Initial Accuracy: Approximately 46.2%
This basic model serves as our starting point for the Food-101 classification task.
#### 2. Custom Model
Improved Accuracy: Approximately 55.5%
Significant improvement (over 20%) compared to the baseline model.
We've made architectural enhancements, including increased depth and the addition of a 3x3 max-pooling layer between conv4 and conv5.
#### 3. Transfer Learning with VGG16
Accuracy: Approximately 68%
We employed VGG16 as a pre-trained model and fine-tuned it by replacing the last fully connected layer.
Significantly higher accuracy compared to the baseline and custom models.
#### 4. Transfer Learning with ResNet18
Accuracy: Approximately 68%
Similar to VGG16, we utilized ResNet18 as a pre-trained model and fine-tuned it.
Achieved accuracy comparable to the VGG16 model.
Key Observations
Model Depth and Performance
We noticed a consistent trend that suggests deeper models tend to perform better. While this may not hold true for all models, our experiments indicate that, in many cases, increasing the model's depth can lead to improved performance.
### Weight and Feature Map Visualization
We conducted visualizations of the models' weight and feature maps for the first convolutional layer and three different position convolutional layers (excluding the baseline model). We found that models with better performance exhibited simpler weight and feature maps. Simplicity, in this context, refers to more colorless maps with fewer gray areas. This suggests that high-performing models have more specific and less ambiguous representations of important image features for each class.

### Performance Enhancements Method
In the custom model, we implemented several changes that contributed significantly to its improved performance:
#### Architecture Modifications: 
We increased the model's depth, added a 3x3 max-pooling layer, and repositioned the dropout layer. These changes allowed us to extract more features from the training images, even with a limited dataset.
#### Learning Rate Adjustment: 
We decreased the learning rate to 0.95e-3, reducing the model's sensitivity to the training set and mitigating the risk of overfitting.
#### Data Augmentation: 
By adding horizontal randomize flips during the data preprocessing stage, we introduced noise and increased dataset variety, further reducing the risk of overfitting.
### Conclusion
Our project demonstrates the power of CNNs in tackling the Food-101 classification task. We've observed that model depth plays a crucial role in performance, with deeper models generally outperforming their shallower counterparts. Additionally, visualizations of weight and feature maps have provided insights into the models' interpretability and feature representation.We've highlighted key enhancements that significantly improved model performance, including architecture modifications, learning rate adjustments, and data augmentation. These findings contribute to our understanding of CNNs for image classification tasks and offer valuable insights for future research and model development.Feel free to explore our code, datasets, and results to gain a deeper understanding of our experiments and findings. If you have any questions or feedback, please don't hesitate to reach out. Thank you for visiting our repository!