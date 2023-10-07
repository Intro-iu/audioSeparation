import os
import random
import numpy as np
from scipy.io.wavfile import read, write

audio_folders = ["samples/1", "samples/2", "samples/3", "samples/4", "samples/5"]

for t in range(100):
   num_voices = random.randint(1, 5)
   selected_folders = random.sample(audio_folders, num_voices)
   
   selected_audio_files = [random.choice([os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".wav")]) for folder in selected_folders]

   rates = []
   data = []
   for audio_file in selected_audio_files:
       rate, audio = read(audio_file)
       rates.append(rate)
       data.append(audio)

   merged_data = np.zeros_like(data[0])
   for audio in data:
       merged_data += audio

   output_file = f"samples/Merge/Merge-{t+1}.wav"
   write(output_file, rates[0], merged_data)
   source_array = np.zeros((1, 5))
   for folder in selected_folders:
      source_array[0, audio_folders.index(folder)] = 1
   print(f"{output_file}  Sources: {source_array}")
   
