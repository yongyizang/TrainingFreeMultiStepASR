import soundfile as sf
from espnet2.bin.enh_inference import SeparateSpeech
from tqdm import tqdm
import numpy as np
import torch
from torch_pesq import PesqLoss
import os
import librosa

model = SeparateSpeech.from_pretrained(
    model_tag="wyz/vctk_dns2020_whamr_bsrnn_large_double_noncausal", # change to large or xlarge
    normalize_output_wav=False,
    device="cuda",
)

print("Model loaded successfully.")

def pesq_score(ref, est):
    ref = torch.tensor(ref, dtype=torch.float32)
    est = torch.tensor(est, dtype=torch.float32)
    mos = pesq.mos(ref, est)
    return mos.item()

def save_audio(file_path, audio, fs):
    # if clipping, then normalize.
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))
    sf.write(file_path, audio, fs)

clean_files, noisy_files = [], []
clean_dir, noisy_dir = "/root/autodl-fs/clean_testset_wav", "/root/autodl-fs/noisy_testset_wav"
# Get the list of files in the clean and noisy directories
for root, dirs, files in os.walk(clean_dir):
    for file in files:
        if file.endswith(".wav"):
            clean_files.append(os.path.join(root, file))
            
for root, dirs, files in os.walk(noisy_dir):
    for file in files:
        if file.endswith(".wav"):
            noisy_files.append(os.path.join(root, file))

# Sort the files to ensure they are in the same order
clean_files.sort()
noisy_files.sort()

print(f"Number of clean files: {len(clean_files)}")
print(f"Number of noisy files: {len(noisy_files)}")
# Check if the number of files is the same
if len(clean_files) != len(noisy_files):
    raise ValueError("The number of clean and noisy files must be the same.")
    
os.makedirs("/root/autodl-tmp/inferred_xlarge_demand", exist_ok=True)
    
for noisy_file in tqdm((noisy_files), total=len(noisy_files)):
    audio, fs = sf.read(noisy_file)
    inferred = model(audio[None, :], fs=fs)[0]
    filename = noisy_file.split("/")[-1]
    save_audio(f"/root/autodl-tmp/inferred_xlarge_demand/{filename}", inferred[0], fs)