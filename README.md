# Card-Fraud-Identification
Using credit card data from 2023, this Pytorch neural network is meant to identify instances of fraudulent activity when they occur. 

This neural network was designed and trained on the data from the spreadsheet 'Credit Card Fraud.xlsx' obtained from Kaggle.com. The data gathered is 2023 transaction information from Europe. Variables have been replaced with V1, V2, ... , V31 to protext sensitive user info.

The network works surprisingly well with accuracy 95%, most inconsistences I think are due to error in changing data types in Python from float to integer values. True accuracy is higher than this, I think >99%.
