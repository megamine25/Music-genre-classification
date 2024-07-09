import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.get_device_name(0))

model_dir = "Wav2vec2/model/Wav2vec_large_self"
processor = Wav2Vec2Processor.from_pretrained(model_dir)
model = Wav2Vec2Model.from_pretrained(model_dir)
model = model.to("cuda")
"""
class CAM(nn.Module):

    def __init__(self, feature_dim1, feature_dim2, num_heads):
        super(CAM, self).__init__()
        self.feature_dim1 = feature_dim1
        self.feature_dim2 = feature_dim2
        self.num_heads = num_heads
        self.joint_dim = feature_dim1 + feature_dim2

        # Cross-attention layers
        self.attn1 = nn.MultiheadAttention(embed_dim=self.joint_dim, num_heads=num_heads)
        self.attn2 = nn.MultiheadAttention(embed_dim=self.joint_dim, num_heads=num_heads)

        # Fully connected layer to combine attended features
        self.fc = nn.Linear(self.joint_dim, self.joint_dim)

    def forward(self, features1, features2):
        # Ensure the feature dimensions are the same
        if features1.size(-1) != features2.size(-1):
            raise ValueError("Feature dimensions must be the same for cross-attention")

        # Concatenate the features along the last dimension
        joint_features = torch.cat((features1, features2), dim=-1)  # Shape: [batch_size, sequence_length, joint_dim]

        # Transpose to match the input shape expected by nn.MultiheadAttention
        joint_features = joint_features.transpose(0, 1)  # Shape: [sequence_length, batch_size, joint_dim]

        # Apply cross-attention
        attn_output1, _ = self.attn1(joint_features, joint_features, joint_features)
        attn_output2, _ = self.attn2(joint_features, joint_features, joint_features)

        # Combine the attention outputs
        combined_output = attn_output1 + attn_output2

        # Transpose back the attended features to [batch_size, sequence_length, joint_dim]
        combined_output = combined_output.transpose(0, 1)  # Shape: [batch_size, sequence_length, joint_dim]

        # Apply fully connected layer
        output = F.relu(self.fc(combined_output))  # Shape: [batch_size, sequence_length, joint_dim]

        return output
"""
class JointCrossAttention(nn.Module):
    def __init__(self, feature_dim, encoded_dim, num_heads):
        super(JointCrossAttention, self).__init__()
        self.feature_dim = feature_dim
        self.encoded_dim = encoded_dim
        self.num_heads = num_heads
        self.joint_dim = 2 * encoded_dim

        # Encoding layers
        self.encoder1 = nn.Linear(feature_dim, encoded_dim)
        self.encoder2 = nn.Linear(feature_dim, encoded_dim)

        # Affine transformation layers
        self.affine_a = nn.Linear(encoded_dim, encoded_dim, bias=False)
        self.affine_v = nn.Linear(encoded_dim, encoded_dim, bias=False)

        # Attention transformation layers
        self.W_a = nn.Linear(encoded_dim, encoded_dim, bias=False)
        self.W_v = nn.Linear(encoded_dim, encoded_dim, bias=False)
        self.W_ca = nn.Linear(encoded_dim, encoded_dim, bias=False)
        self.W_cv = nn.Linear(encoded_dim, encoded_dim, bias=False)

        # Output transformation layers
        self.W_ha = nn.Linear(encoded_dim, encoded_dim, bias=False)
        self.W_hv = nn.Linear(encoded_dim, encoded_dim, bias=False)

        # Activation functions
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        # Final fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.joint_dim, self.encoded_dim),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(self.encoded_dim, 1)
        )

    def forward(self, features1, features2):
        # Encode features
        enc1 = self.encoder1(features1)  # [batch_size, encoded_dim]
        enc2 = self.encoder2(features2)  # [batch_size, encoded_dim]

        # Check dimensions
        #print("enc1 shape:", enc1.shape)  # Should be [batch_size, encoded_dim]
        #print("enc2 shape:", enc2.shape)  # Should be [batch_size, encoded_dim]

        # Affine transformations
        aff_a = self.affine_a(enc1)
        aff_v = self.affine_v(enc2)

        # Attention mechanisms
        attn_a = torch.bmm(enc1.unsqueeze(2), aff_a.unsqueeze(1)).squeeze(2)
        attn_v = torch.bmm(enc2.unsqueeze(2), aff_v.unsqueeze(1)).squeeze(2)

        attn_a = self.tanh(attn_a / math.sqrt(self.encoded_dim))
        attn_v = self.tanh(attn_v / math.sqrt(self.encoded_dim))

        # Attention transformations
        H_a = self.relu(self.W_ca(attn_a) + self.W_a(enc1))
        H_v = self.relu(self.W_cv(attn_v) + self.W_v(enc2))

        # Combine attended features
        attn_enc1 = self.W_ha(H_a) + enc1
        attn_enc2 = self.W_hv(H_v) + enc2

        # Concatenate attended features
        joint_features = torch.cat((attn_enc1, attn_enc2), dim=-1)

        # Final fully connected layer
        output = self.fc(joint_features)

        return output

# Instantiate the JointCrossAttention model
modelca = JointCrossAttention(feature_dim=1024, encoded_dim=512, num_heads=8).to("cuda")
#------------------------------------------------------------------------
path2 = 'Wav2vec2/GTzan_16k_Wav'

# Read a directory and put all files in a list
file_list = []
i = 0 
for path, subdirs, files in os.walk( path2 ):
    for name in files:
        file_list.append( os.path.join( path, name) )
        i += 1
print("Files processed: "+str(i) )


#-------------------------------------------------------------------------
# sample rate = 16,000
#  1s = 16,000 x 1 =  16,000
# 60s = 16,000 x 60 = 960,500

length = list()
i      = 0
avg    = 0

for file in file_list:
    data, samplerate = sf.read( file )
    
    #if len(data) <= 960500:                         
           # print("Audio length: "+str(len(data))+" with less than 30s: "+str(file) )
    #computer average lenght of files
    avg = avg + len(data)
    length.append(len(data))
    i += 1

print( "Files processed: "+str(i) )
print( "Average file length: "+str(avg/i) + " samples   "+str(avg/i/samplerate)+" s   "+str(avg/i/samplerate/60)+" min" )
print( "Max length: "+str(max(length))+ " samples   "+str(max(length)/samplerate)+" s   "+str(max(length)/samplerate/60)+" min" )
print( "Min length: "+str(min(length))+ " samples   "+str(min(length)/samplerate)+" s   "+str(min(length)/samplerate/60)+" min" )

#---------------------------------------------------------------------------
sampling_rate = 16000
track_count = 0
print('\n extraction started')
#file_list=file_list[:1]
for file in file_list:
    data , samplerate = sf.read( file )
    print ("--------------")
    print ("Sample Rate: " + str(samplerate) + " Length: " + str(data.shape) + " " + str( file ) )

    input_values = processor(data, return_tensors="pt", sampling_rate=samplerate).input_values  
    input_values = input_values.to("cuda") 
    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states  
    
    rep6 = hidden_states[5]  # 6th layer
    rep6 = rep6.squeeze().cpu().numpy()

    rep12 = hidden_states[11]  # 12th layer
    rep12 = rep12.squeeze().cpu().numpy()
    #print(len(rep6))
    result_array = np.empty((len(rep6), 512))
    for i in range(len(rep6)):
        features1 = torch.tensor(rep6[i]).unsqueeze(0).to("cuda") 
        features2 = torch.tensor(rep12[i]).unsqueeze(0).to("cuda") 
    
        outputs = modelca(features1, features2)
        #print(outputs)
        outputs = outputs.squeeze().detach().cpu().numpy()
        #print(type(outputs))
        result_array[i] = outputs
    
    #print(result_array)
    rep_df = pd.DataFrame(result_array)
    #print(rep_df)
    file_id = 'Wav2vec2/Wave2vec_Large_Self/FeatExt/tst/'+file.split('/')[2].split('.')[0]+'.'+file.split('/')[2].split('.')[1]+'.csv'
    rep_df.to_csv(file_id)

    print( file_id, " ", str(track_count) )
    track_count += 1
