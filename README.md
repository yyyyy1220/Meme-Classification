# Meme-Classification
# Meme-Classification
Model
Our model uses the state-of-the-art visual language model LLaVA and fine-tunes it for our specific task of meme classification. Initially, we used LLaVA's pre-training weights and trained on the MemeCap dataset using both image and text inputs to better interpret the metaphor in the memes. This yielded a model capable of recognizing and understanding metaphors in memes.

We then used the finetuned model on the Hateful Memes dataset and categorized the memes into positive and negative using the pre-trained language model RoBERTa. We also explored methods of training from scratch without using pre-trained weights to assess the impact of this initialization on classification accuracy.


Evaluation
We assessed the performance of our meme classification models using several metrics: Validation Accuracy, Validation Precision, Validation Recall, Validation F1 Score, and Validation AUROC. The models were trained with different configurations: image-only and image + text inputs, and tested with the same configurations to evaluate their effectiveness.


Results
Following are examples for users
![Deployment](https://github.com/yyyyy1220/Meme-Classification/blob/main/ezgif-1-3525ac64c4.gif)
