#%run WavLM
import runpy
runpy.run_path('WavLM/WavLM.py')

import pandas as pd
import numpy as np
import torch
from WavLM import WavLM, WavLMConfig
import os

import soundfile as sf

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.get_device_name(0))

# load the pre-trained checkpoints
#checkpoint = torch.load('model/WavLM-Base+.pt')
checkpoint = torch.load('WavLM/model/WavLM-Large.pt')
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model = model.to("cuda")
#model = model.to(device) #, dtype=torch.float32)
model.load_state_dict(checkpoint['model'])
model.eval()
#------------------------------------------------------------------------
path2 = 'WavLM/GTzan_16k_Wav'

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
    print('WavLM/wavlmlargefeat/'+file.split('/')[2].split('.')[0]+'.'+file.split('/')[2].split('.')[1]+'.csv')
    
    # extract the representation of last layer
    wav_input = torch.from_numpy(data).float()
    wav_input_16khz = torch.unsqueeze(wav_input,0)
    # wav_input_16khz = torch.randn(1,48000)
    # wav_input_16khz = torch.randn(1,2786987)
    if cfg.normalize:
        wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)

    # last layer
    #rep = model.extract_features(wav_input_16khz.to(device, dtype=torch.float32))[0]
    rep = model.extract_features(wav_input_16khz.to("cuda", dtype=torch.float32))[0]
    #rep = model.extract_features(wav_input_16khz)[0]

    # intermediate layers
    #rep, layer_results = model.extract_features(wav_input_16khz, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
    # Layers 0 to 12 layer_reps[0] .... layer_reps[12] 
    #layer_reps = [x.transpose(0, 1) for x, _ in layer_results]

    # rep = layer_reps[11] # 12th layer
    # rep = layer_reps[7] # 8th layer
    # rep = layer_reps[3] # 4th layer

    rep[0].shape
    rep_np = rep[0].detach().cpu().numpy()
    rep_df = pd.DataFrame(rep_np)

    #file_id = 'features/'+file.split('/')[2].split('.')[0]+'.wavlmbasefeat'  

    #file_id = 'features/'+file.split('/')[2].split('.')[0]+'.'+file.split('/')[2].split('.')[1]+'.wavlmlargefeat'

    file_id = 'WavLM/wavlmlargefeat/'+file.split('/')[2].split('.')[0]+'.'+file.split('/')[2].split('.')[1]+'.csv'
    rep_df.to_csv(file_id)
    
    # Pooling 
    #df_temp = pd.DataFrame(rep[0].detach().numpy())
    #df_pool = df_temp.rolling(2, step=2).mean().drop(index=0) 
    
    #file_id = 'Recola2018_16k/features/'+file.split('/')[3].split('.')[0]+'.wavlmbasefeatpool'  
    #df_pool.to_csv(file_id)
    
    print( file, " ", str(track_count) )
    
    track_count += 1

file_id