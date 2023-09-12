### Introduction
This repository documents our research project, which explores the use of emotion-assisted prompts and back translation with emotions to enhance the quality of synthetic data and improve the performance of the BERT classification model. We conducted experiments with various prompts provided to GPT-3.5-turbo and observed significant improvements in test accuracy, particularly when using prompt 4 or prompt 5 compared to the baseline prompt.

### Experimental Insights
In summary, our project involved five distinct prompts, each shedding light on the behavior and performance of our model:

#### Prompt 4 and Prompt 5:
These prompts involved incorporating the text's emotion label and employing a 'translate-back' mechanism.
Results showed a substantial improvement in test accuracy.
These strategies effectively guided ChatGPT in correctly interpreting and rephrasing the text.

#### Prompt 3:
This prompt aimed to enrich context by extending sentence length.
Results were mixed, with some cases showing improved accuracy, while others did not.
Indicating that additional context does not universally lead to increased accuracy.

#### Prompt 2:
Designed to mimic the original speaker's tone.
Highlighted the importance of maintaining the unique style and voice of the text.
Effectiveness depended on the explicitness of the author's tone in the original text.
These findings underline the complexity of emotion detection in text and suggest that a combination of strategies, including explicit emotional context, language translation mechanics, and tonality mimicry, may be necessary to enhance model accuracy. Despite certain limitations and variable results, these refined prompts led to an overall improvement in performance compared to our baseline model. This study contributes valuable insights for the further development and refinement of AI models in the field of emotion detection.

### Project Challenges
We encountered several challenges in this project:

#### Time and Usage Limitations:
Generating augmented text for each original text using GPT-3.5-turbo is time-consuming and subject to usage limitations.
GPT-3.5-turbo's usage is token-based, which can be limiting for large datasets, potentially requiring the use of partial datasets for training.

#### Limited Training Data:
GPT-3.5-turbo's training data primarily consists of internet text, which may not cover all domains or specific topics.
This limitation can impact the model's ability to generate diverse and accurate augmentations for specialized or domain-specific contexts, such as the Poem Sentiment dataset.

### Future Work
For future work, explore the following avenues:

#### Transition to GPT-4:
Leveraging the API of GPT-4 when it becomes available.
Expecting improved accuracy due to GPT-4's superior understanding and interpretation of emotional tone in text inputs.
Word-Level Analysis:

#### Investigating the impact of individual word choices on rephrasing.
Analyzing how slight modifications in input words influence GPT's rephrasing style.
Our findings provide valuable insights into emotion-aware text augmentation and have the potential to guide the development of more effective models for emotion detection and text rephrasing.
We invite you to explore the code, data, and detailed experiment results in this repository for a deeper understanding of our project's methodologies and outcomes.
