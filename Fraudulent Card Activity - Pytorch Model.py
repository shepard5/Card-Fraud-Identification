import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# List of your Excel files for each season
card_data = pd.read_excel("C:\\Users\\Sam\\Desktop\\Credit Card Fraud\\Credit Card Fraud.xlsx")

# Load and concatenate data from all sheets into one DataFrame
#col_to_keep = ['ESPN Quarterback Rating', 'Net Total Yards','Sacks','3rd down %','Total Penalty Yards','Turnover Ratio','Total Sacks','Total Points','Stuffs','Defensive Touchdown','Passes Defended','Win Totals']


#Columns not consistent in all spreadsheets
#.drop(columns=["Red Zone Touchdown Percentage","Red Zone Scoring Percentage","Red Zone Field Goal Percentage","Red Zone Efficiency Percentage","Two Point Pass Attempts","Two Point Pass","Two Point Pass Conversions", "Average Gain","Unnamed: 0"])for file in file_names]

df = pd.DataFrame(card_data)
df = df.dropna(axis=0, how='all')

print(df)

#Data Preprocessing
features = df.drop('Class', axis=1)
target = df['Class']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Setting up Neural Network

class FraudDetection(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetection, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

#Training the network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = X_train.shape[1]
model = FraudDetection(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        







##Model Evaluation

## Assuming you have loaded the model, criterion (for MSE), and the scaler from previous training


## Load the new data for prediction
fraud_test_file = "C:\\Users\\Sam\\Desktop\\Credit Card Fraud\\Credit Card Fraud.xlsx"
df_fraud_test = pd.read_excel(fraud_test_file)

## Extract the features and target values
test_features = df_fraud_test.drop('Class', axis=1)
test_labels = df_fraud_test['Class']

#Scale data the same as it was during training
test_features_scaled = scaler.transform(test_features)

#Convert dataframes to numpy array
test_features_scaled_numpy = test_features_scaled
test_labels_numpy = test_labels.values

#test_features = test_features.astype('float32')
#test_labels = test_labels.astype('int64')


test_labels_tensor = torch.tensor(test_labels_numpy, dtype = torch.float32).to(device)
test_features_tensor = torch.tensor(test_features_scaled_numpy, dtype = torch.float32).to(device)

test_labels_tensor = test_labels_tensor.to(device)
test_features_tensor = test_features_tensor.to(device)

#print(test_labels)
#print(test_features)

#test_labels = pd.read_excel(fraud_test_file, header=None, nrows=1)

## Create TensorDataset
dataset = TensorDataset(test_features_tensor,test_labels_tensor)

## Create a DataLoader
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

## Initialize a list to store predictions and actual labels

all_predictions = []
all_actual_labels = []

## Loop through batches
model.eval()
correct = 0
total = 0
count = 0
with torch.no_grad():
    for batch_features, batch_labels in data_loader:
        
        ## Make predictions using the model
        batch_features = batch_features.to(device)
        batch_predictions = model(batch_features).to(device)

        binary_predictions = (batch_predictions >= 0.5).to(torch.int)
        
        ## Append batch predictions and actual labels to the lists
        #all_predictions.append(batch_predictions.cpu().numpy())
        
        incorrect = (binary_predictions != batch_labels).sum().item()
        correct += batch_labels.size(0) - incorrect  
        total += batch_labels.size(0)

        all_predictions.append(binary_predictions.cpu().numpy())
        all_actual_labels.append(batch_labels.cpu().numpy())
        if count%1000 == 0:
            print("Batch Features:", batch_features.shape)
            print("Batch Labels:", batch_labels.shape)
            print("Batch Predictions:", batch_predictions)
            
        #print("Batch Features:", batch_features.shape)
        #print("Batch Labels:", batch_labels.shape)
        count = count+1
        
all_predictions = np.concatenate(all_predictions)
all_predictions = all_predictions.squeeze()
all_actual_labels = np.concatenate(all_actual_labels)

print("Predictions:", all_predictions)
print(F"Total Incorrect Predictions: {total - correct}")
data = {
    'ID': df_fraud_test['id'],
    'Actual Labels': all_actual_labels,
    'Binary Predictions': all_predictions
}

df = pd.DataFrame(data)

output_file_path = "Neural Net Model Predictions.xlsx"

df.to_excel(output_file_path, index = False)
### Convert the data to a PyTorch tensor
##input_tensor_new = torch.tensor(input_data_scaled_new, dtype=torch.float32).to(device)
##actual_tensor_new = torch.tensor(actual_win_totals.values, dtype=torch.float32).unsqueeze(1).to(device) 
##
### Predict with the model
##model.eval()  # Set model to evaluation mode
##predictions_new = model(input_tensor_new)
##
### Calculate MSE
##loss = criterion(predictions_new, actual_tensor_new)
##print(f"MSE Loss on New Data: {loss.item():.4f}")
##
### Convert predictions to numpy array for other uses if needed
##predictions_np_new = predictions_new.cpu().detach().numpy()
##print(predictions_np_new)
