# itr introduction 
This repository contains pytorch lightning implementation of Bert model for Translation of text from hindi to english. One of the main benefit of using pytorch lightning over pytorch is that we can very easily automate training and validation loops without explicitly needing to write code for them.<br/>

The datset used is hin-eng dataset and it can be found  [here](https://www.manythings.org/anki/hin-eng.zip) <br/>

## Steps to run

It is advisable to run the code on google colab as your local GPU might run out of the memory <br/>

1. Upload the whole code on your google drive <br/>
2. Open the train.ipynb file in colab and run all the cells <br/>

## Requirements

1. torch <br/> 
2. pytorch lightning <br/>
3. tranformers <br/>

All these can be installed using pip <br/>

## Extra information

To learn more about pytorch lightning , click  [here](https://pytorch-lightning.readthedocs.io/en/latest/index.html) <br/>

To learn more about Bert , check out hugging face documentation [here](https://huggingface.co/transformers/_modules/transformers/modeling_bert.html)



