import pandas as pd
import numpy as np
import torch
import os
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.get_device_name(0))

model_dir = "Wav2vec2/model/Wav2vec_large"
processor = Wav2Vec2Processor.from_pretrained(model_dir)
model = Wav2Vec2Model.from_pretrained(model_dir)
model = model.to("cuda")
#print(model.eval())

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
for file in file_list:
    data , samplerate = sf.read( file )
    print ("--------------")
    print ("Sample Rate: " + str(samplerate) + " Length: " + str(data.shape) + " " + str( file ) )

    input_values = processor(data, return_tensors="pt", sampling_rate=samplerate).input_values  
    input_values = input_values.to("cuda") 
    with torch.no_grad():
        outputs = model(input_values)
        features = outputs.last_hidden_state
        features = features.squeeze().cpu().numpy()  
    rep_df = pd.DataFrame(features)

    file_id = 'Wav2vec2/Wav2vec_Large/wav2vec_large_feat/'+file.split('/')[2].split('.')[0]+'.'+file.split('/')[2].split('.')[1]+'.csv'
    rep_df.to_csv(file_id)
    print( file_id, " ", str(track_count) )
    track_count += 1
