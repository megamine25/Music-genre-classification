import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from torch.optim.lr_scheduler import (
    StepLR, ReduceLROnPlateau, MultiStepLR, ExponentialLR, CosineAnnealingLR
)
import random
from sklearn.model_selection import train_test_split
import audmetric
import wandb
wandb.login()
run=wandb.init(
    # set the wandb project where this run will be logged
    project="WavLM_Large_1stLayer", # ===================================== Change this ==========================

    # track hyperparameters and run metadata
    config={
    "architecture": "simpleMLP",
    "dataset": "wavLM_Large_1stLayer",
    },reinit=True
)

# Path to feature files
path_features = 'WavLM/Wavlm_Large/wavlmLarge_InterFeat/1st_Layer/' #====================================== Change this ========================== 
extension     = 'csv'
all_files = [file for file in os.listdir(path_features) if file.endswith(extension)]
sorted_all_files = sorted(all_files)
#print(sorted_all_files[:5])
# Create a stratified 3-fold holdout with shuffled 334, 333, and 333 samples
filenames = sorted_all_files
random.Random(10).shuffle(filenames)
genres = [filename.split('.')[0] for filename in filenames]

fold1_gtzan, remaining_filenames, fold1_genres, remaining_genres = train_test_split(
    filenames, genres, train_size=334, stratify=genres, random_state=42
)

fold2_gtzan, fold3_gtzan, fold2_genres, fold3_genres = train_test_split(
    remaining_filenames, remaining_genres, train_size=333, stratify=remaining_genres, random_state=42
)
# Example usage:
#print("Fold 1:", len(fold1_gtzan), fold1_gtzan[:5], fold1_genres[:5])
#print("Fold 2:", len(fold2_gtzan),fold2_gtzan[:5], fold2_genres[:5])
#print("Fold 3:", len(fold3_gtzan),fold3_gtzan[:5], fold3_genres[:5])
# All feature vectors into a single dataframe
# GTzan fold1
dfs = []
for file in fold1_gtzan:
    df = pd.read_csv(os.path.join(path_features, file))
    dfs.append(df)
    #print(df.shape)
    #print(df.head)
    

df_fold1_gtzan = pd.concat(dfs, ignore_index=True)
df_fold1_gtzan.drop(df_fold1_gtzan.columns[[0]], axis=1, inplace=True)
print(df_fold1_gtzan.shape)
#print(df_fold1_gtzan.head)
# GTzan fold2 
dfs = []
for file in fold2_gtzan:
    df = pd.read_csv(os.path.join(path_features, file))
    dfs.append(df)

df_fold2_gtzan = pd.concat(dfs, ignore_index=True)
df_fold2_gtzan.drop(df_fold2_gtzan.columns[[0]], axis=1, inplace=True)
print(df_fold2_gtzan.shape)
# GTzan fold3
dfs = []
for file in fold3_gtzan:
    df = pd.read_csv(os.path.join(path_features, file))
    dfs.append(df)

df_fold3_gtzan = pd.concat(dfs, ignore_index=True)
df_fold3_gtzan.drop(df_fold3_gtzan.columns[[0]], axis=1, inplace=True)
print(df_fold3_gtzan.shape)

# Process label files
gtzan_path_features = "WavLM/Wavlm_Large/wavlmLarge_InterFeat/1st_Layer/" #======================================= Change this ==========================
genre_label_map = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9
}
# GTzan Fold1
data_list = []
for filename in fold1_gtzan:
    genre_path = os.path.join(gtzan_path_features, filename)
    genre_track, extension = os.path.splitext(filename)
    genre, track_number = genre_track.split('.')
    feature_filename = f"{genre}.{track_number}.csv"
    feature_filepath = os.path.join(gtzan_path_features, feature_filename)
    if os.path.exists(feature_filepath):
        with open(feature_filepath, 'r') as f:
            num_lines = sum(1 for line in f)
            num_lines = num_lines - 1
        # Append a dictionary to the list replicated by the number of lines
        for _ in range(num_lines):
            data_list.append({
                "genre": genre,
                "track_number": int(track_number),
                "label": genre_label_map[genre]
            })
df_fold1_lab = pd.DataFrame(data_list)
#print(df_fold1_lab.head())
# GTzan Fold2
data_list = []
for filename in fold2_gtzan:
    genre_path = os.path.join(gtzan_path_features, filename)
    genre_track, extension = os.path.splitext(filename)
    genre, track_number = genre_track.split('.')
    feature_filename = f"{genre}.{track_number}.csv"
    feature_filepath = os.path.join(gtzan_path_features, feature_filename)
    if os.path.exists(feature_filepath):
        with open(feature_filepath, 'r') as f:
            num_lines = sum(1 for line in f)
            num_lines = num_lines - 1
        # Append a dictionary to the list replicated by the number of lines
        for _ in range(num_lines):
            data_list.append({
                "genre": genre,
                "track_number": int(track_number),
                "label": genre_label_map[genre]
            })
df_fold2_lab = pd.DataFrame(data_list)
#print(df_fold2_lab.head())
# GTzan Fold3
data_list = []
for filename in fold3_gtzan:
    genre_path = os.path.join(gtzan_path_features, filename)
    genre_track, extension = os.path.splitext(filename)
    genre, track_number = genre_track.split('.')
    feature_filename = f"{genre}.{track_number}.csv"
    feature_filepath = os.path.join(gtzan_path_features, feature_filename)
    if os.path.exists(feature_filepath):
        with open(feature_filepath, 'r') as f:
            num_lines = sum(1 for line in f)
            num_lines = num_lines - 1
        # Append a dictionary to the list replicated by the number of lines
        for _ in range(num_lines):
            data_list.append({
                "genre": genre,
                "track_number": int(track_number),
                "label": genre_label_map[genre]
            })
df_fold3_lab = pd.DataFrame(data_list)
#print(df_fold3_lab.head())
#------ check gpu -----
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.get_device_name(0))
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# =======================
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, dropout_prob, l2_reg):
        super(SimpleMLP, self).__init__()
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_size1)
        # Batch normalization layer for input
        self.input_batchnorm = nn.BatchNorm1d(hidden_size1)
        # First hidden layer
        self.hidden_layer1 = nn.Linear(hidden_size1, hidden_size2)
        # Batch normalization layer for first hidden layer
        self.hidden_batchnorm1 = nn.BatchNorm1d(hidden_size2)
        # Second hidden layer
        self.hidden_layer2 = nn.Linear(hidden_size2, hidden_size2)
        # Batch normalization layer for second hidden layer
        self.hidden_batchnorm2 = nn.BatchNorm1d(hidden_size2)
        # Output layer
        self.output_layer = nn.Linear(hidden_size2, num_classes)
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)
        # L2 regularization parameter
        self.l2_reg = l2_reg

    def forward(self, x):
        # Remove the extra dimension
        x = x.squeeze(1)
        # Forward pass through input layer
        x = self.input_layer(x)
        # Apply batch normalization to the output of the input layer
        x = self.input_batchnorm(x)
        x = F.relu(x)
        x = self.dropout(x)
        # Forward pass through first hidden layer
        x = self.hidden_layer1(x)
        # Apply batch normalization to the output of the first hidden layer
        x = self.hidden_batchnorm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        # Forward pass through second hidden layer
        x = self.hidden_layer2(x)
        # Apply batch normalization to the output of the second hidden layer
        x = self.hidden_batchnorm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        # Forward pass through output layer
        x = self.output_layer(x)
        # Apply softmax activation
        output = F.softmax(x, dim=1)

        return output

# Train the model
num_epochs     = 200
batch_size     = 1500
validate_every = 1  # Validate every 2 epochs
patience       = 30  # Stop training if validation loss doesn't improve for 5 consecutive validations
# **Which fold is which?** ================================================ 123 FOLD ==========================
df_train_feat = df_fold1_gtzan
df_train_lab  = df_fold1_lab

df_valid_feat = df_fold2_gtzan
df_valid_lab  = df_fold2_lab

df_test_feat = df_fold3_gtzan
df_test_lab  = df_fold3_lab
#=============================================================
# Training fold
features = df_train_feat.values.astype(np.float32)
labels   = df_train_lab['label'].values.astype(np.float32)
features_tensor = torch.from_numpy(features)
labels_tensor   = torch.from_numpy(labels)
sequence_length = 1
num_features    = features.shape[1]
num_samples     = features.shape[0]
num_sequences = num_samples // sequence_length
features_tensor = features_tensor[:num_sequences * sequence_length, :]
# get tensors for Training
labels_tensor = labels_tensor[:num_sequences * sequence_length]
features_tensor = features_tensor.view(num_sequences, sequence_length, num_features)
# Validation fold
features2 = df_valid_feat.values.astype(np.float32)
labels2   = df_valid_lab['label'].values.astype(np.float32)
features_tensor2 = torch.from_numpy(features2)
labels_tensor2   = torch.from_numpy(labels2)
sequence_length2 = 1
num_features2    = features2.shape[1]
num_samples2     = features2.shape[0]
num_sequences2 = num_samples2 // sequence_length2
features_tensor2 = features_tensor2[:num_sequences2 * sequence_length2, :]
# get tensors for validation 
labels_tensor2 = labels_tensor2[:num_sequences2 * sequence_length2]
features_tensor2 = features_tensor2.view(num_sequences2, sequence_length2, num_features2)
# Test fold
features3 = df_test_feat.values.astype(np.float32)
labels3   = df_test_lab['label'].values.astype(np.float32)
features_tensor3 = torch.from_numpy(features3)
labels_tensor3   = torch.from_numpy(labels3)
sequence_length3 = 1
num_features3    = features3.shape[1]
num_samples3     = features3.shape[0]
num_sequences3 = num_samples3 // sequence_length3
features_tensor3 = features_tensor3[:num_sequences3 * sequence_length3, :]
# get tensors for test
labels_tensor3   = labels_tensor3[:num_sequences3 * sequence_length3]
features_tensor3 = features_tensor3.view(num_sequences3, sequence_length3, num_features3)
######################################

# Split the data into training and testing sets
X_train = features_tensor
X_test  = features_tensor2
y_train = labels_tensor
y_test  = labels_tensor2

# Initialize the model, loss function, and optimizer
input_size   = num_features
hidden_size  = 64 #128, 64, 32, 16
hidden_size1  = 128 #128
hidden_size2  = 64 #64
num_layers   = 2 # 4
num_classes  = 10 # GTzan
dropout_prob = 0.40
l2_reg       = 0.001


model = SimpleMLP(input_size, hidden_size1, hidden_size2, num_classes, dropout_prob, l2_reg)
#==============================
# Move the model to the GPU
model = model.to("cuda")
# Define your loss function (criterion)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
#optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=ls_reg)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
#scheduler = StepLR(optimizer, step_size=10, gamma=0.01)  # Reduce lr by 10% every 10 epochs



##################
train_dataset = TensorDataset(X_train, y_train.to(torch.int64))
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

validation_dataset = TensorDataset( X_test, y_test.to(torch.int64))
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)
##################

# Initialize a list to store the training loss values
train_loss_values = []
validation_loss_values = []

best_validation_loss = float('inf')
early_stop_counter = 0
best_model_path = 'WavLM/Wavlm_Large/train_1st_layer/best123_model_1stLayerLarge.pth'  # ============================= Change this ===============
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total

print("Start 123 fold Training \n")
# Train the model
"""
for epoch in range(num_epochs):

    epoch_loss = 0.0
    correct = 0
    total = 0

    model.train()  # Set the model to training mode

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to("cuda"), batch_y.to("cuda", dtype=torch.int64)
        optimizer.zero_grad()

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        total += batch_y.size(0)
        correct += (outputs.argmax(1) == batch_y).sum().item()

    train_accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.2f}%')
    wandb.log({"train/loss": loss.item(), "train/accuracy": train_accuracy})

    average_epoch_loss = epoch_loss / len(train_loader)
    train_loss_values.append(average_epoch_loss)

    # Validation
    if epoch % validate_every == 0:
        model.eval()  # Set the model to evaluation mode
        validation_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in validation_loader:
                batch_X, batch_y = batch_X.to("cuda"), batch_y.to("cuda", dtype=torch.int64)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                validation_loss += loss.item()
                total += batch_y.size(0)
                correct += (outputs.argmax(1) == batch_y).sum().item()

        val_accuracy = 100 * correct / total
        validation_loss_values.append(loss.item())
        print(f'Validation Loss: {validation_loss / len(validation_loader):.4f}, Accuracy: {val_accuracy:.2f}%')
        wandb.log({"valid/loss": loss.item(), "valid/accuracy": val_accuracy})

        # Save the model with the best validation loss
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            early_stop_counter = 0

            # Save the model with the best validation loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved model with best validation loss to {best_model_path}')
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch+1} as validation loss has not improved for {patience} consecutive validations.')
            break

        model.train()  # Set the model back to training mode
"""
# Test the best model on train, validation and test sets
def predict_label(predictions):
  _, predicted_labels = torch.max(predictions, dim=1)
  return predicted_labels

print("END OF TRAINING")

best_model = SimpleMLP(input_size, hidden_size1, hidden_size2, num_classes, dropout_prob, l2_reg)
best_model.load_state_dict(torch.load(best_model_path))
best_model.to("cuda")
#print(best_model.eval())

# Training set
with torch.no_grad():
    test_outputs = best_model(X_train.to("cuda"))
    test_loss    = criterion(test_outputs, y_train.to(torch.int64).to("cuda"))
print(f'123 Train Loss: {test_loss.item():.4f}')
print("123 Training Accuracy : " + str(audmetric.accuracy(df_train_lab['label'], predict_label(test_outputs.cpu()))))
Train_123=f'123 Train Loss: {test_loss.item():.4f}'+"/"+"123 Training Accuracy : " + str(audmetric.accuracy(df_train_lab['label'], predict_label(test_outputs.cpu())))
#print("\n")
# Validation set
with torch.no_grad():
    test_outputs = best_model(X_test.to("cuda"))
    test_loss    = criterion(test_outputs, y_test.to(torch.int64).to("cuda"))
print(f'123 Validation Loss: {test_loss.item():.4f}')
print("123 Validation Accuracy : " + str(audmetric.accuracy(df_valid_lab['label'], predict_label(test_outputs.cpu()))))
Validation_123=f'123 Validation Loss: {test_loss.item():.4f}'+" / "+"123 Validation Accuracy : " + str(audmetric.accuracy(df_valid_lab['label'], predict_label(test_outputs.cpu())))
#print("\n")
# Test set
with torch.no_grad():
    test_outputs = best_model(features_tensor3.to("cuda"))
    test_loss    = criterion(test_outputs, labels_tensor3.to(torch.int64).to("cuda"))
print(f'123 Test Loss: {test_loss.item():.4f}')
print("123 Test Accuracy: " + str(audmetric.accuracy(df_test_lab['label'], predict_label(test_outputs.cpu()))))
Test_123=f'123 Test Loss: {test_loss.item():.4f}'+" / "+ "123 Test Accuracy: " + str(audmetric.accuracy(df_test_lab['label'], predict_label(test_outputs.cpu())))
# Aggregation of Predictions for each track (wav)
df_test_lab['pred'] = predict_label(test_outputs.cpu())
df_test_probs = pd.DataFrame(test_outputs.cpu())
cols = df_test_lab.columns.to_list() + df_test_probs.columns.to_list()
dft = [df_test_lab, df_test_probs]
df_test_pred = np.concatenate(dft, axis=1)
df_test_pred = pd.DataFrame(df_test_pred, columns=cols)
average_values = df_test_pred.groupby(['genre', 'track_number', 'label'])[list(range(10))].mean().reset_index()
average_values['max_value_column'] = average_values.iloc[:, 3:].astype(float).idxmax(axis=1)
print("123 Average Test Accuracy : " + str(100*audmetric.accuracy(average_values['label'], average_values['max_value_column'])) + " %")
#Train_123=f'123 Train Loss: {test_loss.item():.4f}'+"/"+"123 Training Accuracy : " + str(audmetric.accuracy(df_train_lab['label'], predict_label(test_outputs.cpu())))
#Validation_123=f'123 Validation Loss: {test_loss.item():.4f}'+" / "+"123 Validation Accuracy : " + str(audmetric.accuracy(df_valid_lab['label'], predict_label(test_outputs.cpu())))
#Test_123=f'123 Test Loss: {test_loss.item():.4f}'+" / "+ "123 Test Accuracy: " + str(audmetric.accuracy(df_test_lab['label'], predict_label(test_outputs.cpu())))
AVG_Test_123= "123 Average Test Accuracy : " + str(100*audmetric.accuracy(average_values['label'], average_values['max_value_column'])) + " %"

####################################################################################################################
run.finish()
run=wandb.init(
    # set the wandb project where this run will be logged
    project="WavLM_Large_1stLayer", # ===================================== Change this ==========================

    # track hyperparameters and run metadata
    config={
    "architecture": "simpleMLP",
    "dataset": "wavLM_Large_1stLayer",
    },reinit=True
)
####################################################################################################################
# **Which fold is which?** ================================================ 231 FOLD ==========================

df_train_feat = df_fold2_gtzan
df_train_lab  = df_fold2_lab

df_valid_feat = df_fold3_gtzan
df_valid_lab  = df_fold3_lab

df_test_feat = df_fold1_gtzan
df_test_lab  = df_fold1_lab
#=============================================================
# Training fold
features = df_train_feat.values.astype(np.float32)
labels   = df_train_lab['label'].values.astype(np.float32)
features_tensor = torch.from_numpy(features)
labels_tensor   = torch.from_numpy(labels)
sequence_length = 1
num_features    = features.shape[1]
num_samples     = features.shape[0]
num_sequences = num_samples // sequence_length
features_tensor = features_tensor[:num_sequences * sequence_length, :]
# get tensors for Training
labels_tensor = labels_tensor[:num_sequences * sequence_length]
features_tensor = features_tensor.view(num_sequences, sequence_length, num_features)
# Validation fold
features2 = df_valid_feat.values.astype(np.float32)
labels2   = df_valid_lab['label'].values.astype(np.float32)
features_tensor2 = torch.from_numpy(features2)
labels_tensor2   = torch.from_numpy(labels2)
sequence_length2 = 1
num_features2    = features2.shape[1]
num_samples2     = features2.shape[0]
num_sequences2 = num_samples2 // sequence_length2
features_tensor2 = features_tensor2[:num_sequences2 * sequence_length2, :]
# get tensors for validation 
labels_tensor2 = labels_tensor2[:num_sequences2 * sequence_length2]
features_tensor2 = features_tensor2.view(num_sequences2, sequence_length2, num_features2)
# Test fold
features3 = df_test_feat.values.astype(np.float32)
labels3   = df_test_lab['label'].values.astype(np.float32)
features_tensor3 = torch.from_numpy(features3)
labels_tensor3   = torch.from_numpy(labels3)
sequence_length3 = 1
num_features3    = features3.shape[1]
num_samples3     = features3.shape[0]
num_sequences3 = num_samples3 // sequence_length3
features_tensor3 = features_tensor3[:num_sequences3 * sequence_length3, :]
# get tensors for test
labels_tensor3   = labels_tensor3[:num_sequences3 * sequence_length3]
features_tensor3 = features_tensor3.view(num_sequences3, sequence_length3, num_features3)
######################################

# Split the data into training and testing sets
X_train = features_tensor
X_test  = features_tensor2
y_train = labels_tensor
y_test  = labels_tensor2

input_size   = num_features
hidden_size  = 64 #128, 64, 32, 16
hidden_size1  = 128 #128
hidden_size2  = 64 #64
num_layers   = 2 # 4
num_classes  = 10 # GTzan
dropout_prob = 0.40
l2_reg       = 0.001


model = SimpleMLP(input_size, hidden_size1, hidden_size2, num_classes, dropout_prob, l2_reg)
#==============================

model = model.to("cuda")
# Define your loss function (criterion)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
"""
"""
##################
train_dataset = TensorDataset(X_train, y_train.to(torch.int64))
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

validation_dataset = TensorDataset( X_test, y_test.to(torch.int64))
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)
##################

# Initialize a list to store the training loss values
train_loss_values = []
validation_loss_values = []

best_validation_loss = float('inf')
early_stop_counter = 0
best_model_path = 'WavLM/Wavlm_Large/train_1st_layer/best231_model_1stLayerLarge.pth'  # ============================= Change this ===============
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total

print("Start 231 fold Training \n")
"""
# Train the model
for epoch in range(num_epochs):

    epoch_loss = 0.0
    correct = 0
    total = 0

    model.train()  # Set the model to training mode

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to("cuda"), batch_y.to("cuda", dtype=torch.int64)
        optimizer.zero_grad()

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        total += batch_y.size(0)
        correct += (outputs.argmax(1) == batch_y).sum().item()

    train_accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.2f}%')
    wandb.log({"train/loss": loss.item(), "train/accuracy": train_accuracy})

    average_epoch_loss = epoch_loss / len(train_loader)
    train_loss_values.append(average_epoch_loss)

    # Validation
    if epoch % validate_every == 0:
        model.eval()  # Set the model to evaluation mode
        validation_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in validation_loader:
                batch_X, batch_y = batch_X.to("cuda"), batch_y.to("cuda", dtype=torch.int64)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                validation_loss += loss.item()
                total += batch_y.size(0)
                correct += (outputs.argmax(1) == batch_y).sum().item()

        val_accuracy = 100 * correct / total
        validation_loss_values.append(loss.item())
        print(f'Validation Loss: {validation_loss / len(validation_loader):.4f}, Accuracy: {val_accuracy:.2f}%')
        wandb.log({"valid/loss": loss.item(), "valid/accuracy": val_accuracy})

        # Save the model with the best validation loss
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            early_stop_counter = 0

            # Save the model with the best validation loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved model with best validation loss to {best_model_path}')
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch+1} as validation loss has not improved for {patience} consecutive validations.')
            break

        model.train()  # Set the model back to training mode
"""
# Test the best model on train, validation and test sets
def predict_label(predictions):
  _, predicted_labels = torch.max(predictions, dim=1)
  return predicted_labels

print("END OF TRAINING")

best_model = SimpleMLP(input_size, hidden_size1, hidden_size2, num_classes, dropout_prob, l2_reg)
best_model.load_state_dict(torch.load(best_model_path))
best_model.to("cuda")
#print(best_model.eval())

# Training set
with torch.no_grad():
    test_outputs = best_model(X_train.to("cuda"))
    test_loss    = criterion(test_outputs, y_train.to(torch.int64).to("cuda"))
print(f'231 Train Loss: {test_loss.item():.4f}')
print("231 Training Accuracy : " + str(audmetric.accuracy(df_train_lab['label'], predict_label(test_outputs.cpu()))))
Train_231=f'231 Train Loss: {test_loss.item():.4f}'+"/"+"231 Training Accuracy : " + str(audmetric.accuracy(df_train_lab['label'], predict_label(test_outputs.cpu())))
#print("\n")
# Validation set
with torch.no_grad():
    test_outputs = best_model(X_test.to("cuda"))
    test_loss    = criterion(test_outputs, y_test.to(torch.int64).to("cuda"))
print(f'231 Validation Loss: {test_loss.item():.4f}')
print("231 Validation Accuracy : " + str(audmetric.accuracy(df_valid_lab['label'], predict_label(test_outputs.cpu()))))
Validation_231=f'231 Validation Loss: {test_loss.item():.4f}'+" / "+"231 Validation Accuracy : " + str(audmetric.accuracy(df_valid_lab['label'], predict_label(test_outputs.cpu())))
#print("\n")
# Test set
with torch.no_grad():
    test_outputs = best_model(features_tensor3.to("cuda"))
    test_loss    = criterion(test_outputs, labels_tensor3.to(torch.int64).to("cuda"))
print(f'231 Test Loss: {test_loss.item():.4f}')
print("231 Test Accuracy: " + str(audmetric.accuracy(df_test_lab['label'], predict_label(test_outputs.cpu()))))
Test_231=f'231 Test Loss: {test_loss.item():.4f}'+" / "+ "231 Test Accuracy: " + str(audmetric.accuracy(df_test_lab['label'], predict_label(test_outputs.cpu())))
# Aggregation of Predictions for each track (wav)
df_test_lab['pred'] = predict_label(test_outputs.cpu())
df_test_probs = pd.DataFrame(test_outputs.cpu())
cols = df_test_lab.columns.to_list() + df_test_probs.columns.to_list()
dft = [df_test_lab, df_test_probs]
df_test_pred = np.concatenate(dft, axis=1)
df_test_pred = pd.DataFrame(df_test_pred, columns=cols)
average_values = df_test_pred.groupby(['genre', 'track_number', 'label'])[list(range(10))].mean().reset_index()
average_values['max_value_column'] = average_values.iloc[:, 3:].astype(float).idxmax(axis=1)
print("231 Average Test Accuracy : " + str(100*audmetric.accuracy(average_values['label'], average_values['max_value_column'])) + " %")

AVG_Test_231= "231 Average Test Accuracy : " + str(100*audmetric.accuracy(average_values['label'], average_values['max_value_column'])) + " %"

####################################################################################################################
run.finish()
run=wandb.init(
    # set the wandb project where this run will be logged
    project="WavLM_Large_1stLayer", # ===================================== Change this ==========================

    # track hyperparameters and run metadata
    config={
    "architecture": "simpleMLP",
    "dataset": "wavLM_Large_1stLayer",
    },reinit=True
)
####################################################################################################################
# **Which fold is which?** ================================================ 312 FOLD ==========================
df_train_feat = df_fold3_gtzan
df_train_lab  = df_fold3_lab

df_valid_feat = df_fold1_gtzan
df_valid_lab  = df_fold1_lab

df_test_feat = df_fold2_gtzan
df_test_lab  = df_fold2_lab
#=============================================================
# Training fold
features = df_train_feat.values.astype(np.float32)
labels   = df_train_lab['label'].values.astype(np.float32)
features_tensor = torch.from_numpy(features)
labels_tensor   = torch.from_numpy(labels)
sequence_length = 1
num_features    = features.shape[1]
num_samples     = features.shape[0]
num_sequences = num_samples // sequence_length
features_tensor = features_tensor[:num_sequences * sequence_length, :]
# get tensors for Training
labels_tensor = labels_tensor[:num_sequences * sequence_length]
features_tensor = features_tensor.view(num_sequences, sequence_length, num_features)
# Validation fold
features2 = df_valid_feat.values.astype(np.float32)
labels2   = df_valid_lab['label'].values.astype(np.float32)
features_tensor2 = torch.from_numpy(features2)
labels_tensor2   = torch.from_numpy(labels2)
sequence_length2 = 1
num_features2    = features2.shape[1]
num_samples2     = features2.shape[0]
num_sequences2 = num_samples2 // sequence_length2
features_tensor2 = features_tensor2[:num_sequences2 * sequence_length2, :]
# get tensors for validation 
labels_tensor2 = labels_tensor2[:num_sequences2 * sequence_length2]
features_tensor2 = features_tensor2.view(num_sequences2, sequence_length2, num_features2)
# Test fold
features3 = df_test_feat.values.astype(np.float32)
labels3   = df_test_lab['label'].values.astype(np.float32)
features_tensor3 = torch.from_numpy(features3)
labels_tensor3   = torch.from_numpy(labels3)
sequence_length3 = 1
num_features3    = features3.shape[1]
num_samples3     = features3.shape[0]
num_sequences3 = num_samples3 // sequence_length3
features_tensor3 = features_tensor3[:num_sequences3 * sequence_length3, :]
# get tensors for test
labels_tensor3   = labels_tensor3[:num_sequences3 * sequence_length3]
features_tensor3 = features_tensor3.view(num_sequences3, sequence_length3, num_features3)
######################################

# Split the data into training and testing sets
X_train = features_tensor
X_test  = features_tensor2
y_train = labels_tensor
y_test  = labels_tensor2

# Initialize the model, loss function, and optimizer
input_size   = num_features
hidden_size  = 64 #128, 64, 32, 16
hidden_size1  = 128 #128
hidden_size2  = 64 #64
num_layers   = 2 # 4
num_classes  = 10 # GTzan
dropout_prob = 0.40
l2_reg       = 0.001


model = SimpleMLP(input_size, hidden_size1, hidden_size2, num_classes, dropout_prob, l2_reg)
#==============================
# Move the model to the GPU
model = model.to("cuda")
# Define your loss function (criterion)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
#optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=ls_reg)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
#scheduler = StepLR(optimizer, step_size=10, gamma=0.01)  # Reduce lr by 10% every 10 epochs
"""

"""
##################
train_dataset = TensorDataset(X_train, y_train.to(torch.int64))
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

validation_dataset = TensorDataset( X_test, y_test.to(torch.int64))
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)
##################

# Initialize a list to store the training loss values
train_loss_values = []
validation_loss_values = []

best_validation_loss = float('inf')
early_stop_counter = 0
best_model_path = 'WavLM/Wavlm_Large/train_1st_layer/best312_model_1stLayerLarge.pth'  # ============================= Change this ===============
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total

print("Start 312 fold Training \n")
# Train the model
"""
for epoch in range(num_epochs):

    epoch_loss = 0.0
    correct = 0
    total = 0

    model.train()  # Set the model to training mode

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to("cuda"), batch_y.to("cuda", dtype=torch.int64)
        optimizer.zero_grad()

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        total += batch_y.size(0)
        correct += (outputs.argmax(1) == batch_y).sum().item()

    train_accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.2f}%')
    wandb.log({"train/loss": loss.item(), "train/accuracy": train_accuracy})

    average_epoch_loss = epoch_loss / len(train_loader)
    train_loss_values.append(average_epoch_loss)

    # Validation
    if epoch % validate_every == 0:
        model.eval()  # Set the model to evaluation mode
        validation_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in validation_loader:
                batch_X, batch_y = batch_X.to("cuda"), batch_y.to("cuda", dtype=torch.int64)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                validation_loss += loss.item()
                total += batch_y.size(0)
                correct += (outputs.argmax(1) == batch_y).sum().item()

        val_accuracy = 100 * correct / total
        validation_loss_values.append(loss.item())
        print(f'Validation Loss: {validation_loss / len(validation_loader):.4f}, Accuracy: {val_accuracy:.2f}%')
        wandb.log({"valid/loss": loss.item(), "valid/accuracy": val_accuracy})

        # Save the model with the best validation loss
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            early_stop_counter = 0

            # Save the model with the best validation loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved model with best validation loss to {best_model_path}')
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch+1} as validation loss has not improved for {patience} consecutive validations.')
            break

        model.train()  # Set the model back to training mode
"""
# Test the best model on train, validation and test sets
def predict_label(predictions):
  _, predicted_labels = torch.max(predictions, dim=1)
  return predicted_labels

print("END OF TRAINING")

best_model = SimpleMLP(input_size, hidden_size1, hidden_size2, num_classes, dropout_prob, l2_reg)
best_model.load_state_dict(torch.load(best_model_path))
best_model.to("cuda")
#print(best_model.eval())

# Training set
with torch.no_grad():
    test_outputs = best_model(X_train.to("cuda"))
    test_loss    = criterion(test_outputs, y_train.to(torch.int64).to("cuda"))
print(f'312 Train Loss: {test_loss.item():.4f}')
print("312 Training Accuracy : " + str(audmetric.accuracy(df_train_lab['label'], predict_label(test_outputs.cpu()))))
Train_312=f'312 Train Loss: {test_loss.item():.4f}'+"/"+"312 Training Accuracy : " + str(audmetric.accuracy(df_train_lab['label'], predict_label(test_outputs.cpu())))
#print("\n")
# Validation set
with torch.no_grad():
    test_outputs = best_model(X_test.to("cuda"))
    test_loss    = criterion(test_outputs, y_test.to(torch.int64).to("cuda"))
print(f'312 Validation Loss: {test_loss.item():.4f}')
print("312 Validation Accuracy : " + str(audmetric.accuracy(df_valid_lab['label'], predict_label(test_outputs.cpu()))))
Validation_312=f'312 Validation Loss: {test_loss.item():.4f}'+" / "+"312 Validation Accuracy : " + str(audmetric.accuracy(df_valid_lab['label'], predict_label(test_outputs.cpu())))
#print("\n")
# Test set
with torch.no_grad():
    test_outputs = best_model(features_tensor3.to("cuda"))
    test_loss    = criterion(test_outputs, labels_tensor3.to(torch.int64).to("cuda"))
print(f'312 Test Loss: {test_loss.item():.4f}')
print("312 Test Accuracy: " + str(audmetric.accuracy(df_test_lab['label'], predict_label(test_outputs.cpu()))))
Test_312=f'312 Test Loss: {test_loss.item():.4f}'+" / "+ "312 Test Accuracy: " + str(audmetric.accuracy(df_test_lab['label'], predict_label(test_outputs.cpu())))

# Aggregation of Predictions for each track (wav)
df_test_lab['pred'] = predict_label(test_outputs.cpu())
df_test_probs = pd.DataFrame(test_outputs.cpu())
cols = df_test_lab.columns.to_list() + df_test_probs.columns.to_list()
dft = [df_test_lab, df_test_probs]
df_test_pred = np.concatenate(dft, axis=1)
df_test_pred = pd.DataFrame(df_test_pred, columns=cols)
average_values = df_test_pred.groupby(['genre', 'track_number', 'label'])[list(range(10))].mean().reset_index()
average_values['max_value_column'] = average_values.iloc[:, 3:].astype(float).idxmax(axis=1)
print("312 Average Test Accuracy : " + str(100*audmetric.accuracy(average_values['label'], average_values['max_value_column'])) + " %")
AVG_Test_312= "312 Average Test Accuracy : " + str(100*audmetric.accuracy(average_values['label'], average_values['max_value_column'])) + " %"
print("Recap")
print(Test_123)
print(AVG_Test_123)
print(Test_231)
print(AVG_Test_231)
print(Test_312)
print(AVG_Test_312)
run.finish()
