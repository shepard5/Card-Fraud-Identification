import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os
#Import Transaction Data
#card_data = pd.read_excel("C:\\Users\\Sam\\Desktop\\Credit Card Fraud\\Credit Card Fraud.xlsx") #Windows directory
print(os.getcwd())
card_data = pd.read_excel("./Credit_Card_Fraud.xlsx") #Mac directory

##Pre-processing data
df = pd.DataFrame(card_data)
df = df.dropna(axis=0, how='all')
features = df.drop('Class', axis=1) #Isolate predictor variables; Axis = 1: Columns, Axis = 0: Rows
target = df['Class'] #Isolate dependent variables

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2) #Assign datasets for training & testing, test_size=0.2 indicating 20% of data saved for testing

#Data transformation remedies issues caused by high feature variance in gradient descent models
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

##Setting up Neural Network
class FraudDetection(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetection, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128) #Setting up layers & nodes of the NN
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):#Forward step process for training
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x)) #Sigmoid activation function for outputs 0<x<1
        return x
    
    
##Initialize model training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Using GPU for faster training

input_dim = X_train.shape[1] #.shape[1] attribute spits out # of columns, in this case dimensionality of input vectors
model = FraudDetection(input_dim).to(device) #Feeds the model to GPU memory
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device) #Feeds the model to GPU memory - slow due to CPU to GPU memory bottleneck
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device) #Feeds the model to GPU memory

#Model training
num_epochs = 1000 #No. times training data is passed through nodes collectively/weight & bias reassignment/gradient adjusted 
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad() #Set gradients back to zero
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor) #(outputs - y_train_tensor)
    loss.backward() #Calculate new gradients
    optimizer.step()#Adjust weights based on new gradient informaiton

    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}") #Returns error to four decimal places every 100 iterations
        

def Model_Evaluation():
    #fraud_test_file = "C:\\Users\\Sam\\Desktop\\Credit Card Fraud\\Credit Card Fraud.xlsx" #Windows directory 
    fraud_test_file = "\\Users\\samscott\\Desktop\\Credit Card Fraud.xlsx" #Mac directory
    model.eval()
   
    df_fraud_test = pd.read_excel(fraud_test_file)

    test_features = df_fraud_test.drop('Class', axis=1)
    test_labels = df_fraud_test['Class']

    test_labels_tensor = torch.tensor(test_labels.values, dtype = torch.float32).to(device) #Creating tensor - Check why .values attribute is needed
    test_features_tensor = torch.tensor(scaler.transform(test_features), dtype = torch.float32).to(device) #Transforming data, creating tensor, converting to float32, and sending to GPU

    #test_labels_tensor = test_labels_tensor.to(device) 
    #test_features_tensor = test_features_tensor.to(device)

    dataset = TensorDataset(test_features_tensor,test_labels_tensor)

    batch_size = 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) #Initialize dataloader, divides large datasets into batches to prevent bottlenecking

    all_predictions = [] #Will hold all model predictions
    all_actual_labels = [] #Will hold correponding true values from raw datA

    correct = 0
    total = 0
    count = 0

    #Batch iteration from DataLoader
    with torch.no_grad():#Tells the model to set gradients to zero ie no changes to the model
        for batch_features, batch_labels in data_loader: #iterating through each data point
        
            batch_features = batch_features.to(device) #to gpu
            batch_predictions = model(batch_features).to(device) #to gpu
            binary_predictions = (batch_predictions >= 0.5).to(torch.int) #Sigmoid output not always equal to one or zero, this line forces output to one of them
        
            incorrect = (binary_predictions != batch_labels).sum().item() #For evaluation metric
            correct += batch_labels.size(0) - incorrect  
            total += batch_labels.size(0)

            all_predictions.append(binary_predictions.cpu().numpy()) #Sequenitially add predictions and true values to lists
            all_actual_labels.append(batch_labels.cpu().numpy())

            count = count+1
        
    all_predictions = np.concatenate(all_predictions)
    all_predictions = all_predictions.squeeze()
    all_actual_labels = np.concatenate(all_actual_labels)

    print("Predictions:", all_predictions)
    print(F"Total Incorrect Predictions: {total - correct}")
    print(F"Accuracy:{(correct/total)*100} %")
    data = {
        'ID': df_fraud_test['id'],
        'Actual Labels': all_actual_labels,
        'Binary Predictions': all_predictions
    }

    df = pd.DataFrame(data)
    output_file_path = "Neural Net Model Predictions.xlsx"
    df.to_excel(output_file_path, index = False)
