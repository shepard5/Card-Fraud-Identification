# Card-Fraud-Identification
Using credit card data from 2023, this Pytorch neural network is meant to identify instances of fraudulent activity when they occur. I started this to play around with PyTorch and familiarize myself with neural networks, but I think this neural network will serve as a template for future projects. 

This neural network was designed and trained on the data from the spreadsheet 'Credit Card Fraud.xlsx' obtained from Kaggle.com. The data gathered is 2023 transaction information from Europe. Variables have been replaced with V1, V2, ... , V31 to protect sensitive user info.

The network works surprisingly well with accuracy 95%, most inconsistences I think are due to error in changing data types in Python from float to integer values. True accuracy is higher than this, I think >99%.

Next steps will be to visualize the model. 

The goal is to use data to build models that will drive decision making. Given that I don't have the big decisions to make, I'll have to communicate models that use data to someone who will make a decision. Given the black box nature of a NN, it's difficult to explain how the model iteratively got to where it is, but it can be visualized with renderings.

