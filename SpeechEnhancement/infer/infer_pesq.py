import soundfile as sf
from espnet2.bin.enh_inference import SeparateSpeech
from tqdm import tqdm
import numpy as np
import torch
from torch_pesq import PesqLoss
import os
import librosa

model = SeparateSpeech.from_pretrained(
    model_tag="wyz/vctk_dns2020_whamr_bsrnn_medium_noncausal",
    normalize_output_wav=False,
    device="cuda",
)

pesq = PesqLoss(0.5,
    sample_rate=48000, 
)

print("Model loaded successfully.")

def mix_inferred_with_noisy(inferred, noisy, ratio):
    if len(inferred) != len(noisy):
        # pad the shorter one with zeros
        if len(inferred) < len(noisy):
            inferred = np.pad(inferred, (0, len(noisy) - len(inferred)), mode='constant')
        else:
            noisy = np.pad(noisy, (0, len(inferred) - len(noisy)), mode='constant')
    if len(inferred) != len(noisy):
        raise ValueError("Inferred and noisy audio must have the same length after padding.")
    
    mixed = (1 - ratio) * noisy + ratio * inferred
    return mixed

# def sdr(clean, noisy):
#     noise = noisy - clean
#     sdr = 10 * np.log10(np.sum(clean**2) / np.sum(noise**2))
#     return sdr

def sdr(
    ref: np.ndarray, 
    est: np.ndarray, 
    eps: float = 1e-10
):
    r"""Calcualte SDR.
    """
    noise = est - ref
    numerator = np.clip(a=np.mean(ref ** 2), a_min=eps, a_max=None)
    denominator = np.clip(a=np.mean(noise ** 2), a_min=eps, a_max=None)
    sdr = 10. * np.log10(numerator / denominator)
    return sdr

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
    
os.makedirs("/root/autodl-tmp/inferred_output_vctkdemucs_pesq/step_0", exist_ok=True)
    
# for noisy_file in tqdm((noisy_files), total=len(noisy_files)):
#     audio, fs = sf.read(noisy_file)
#     inferred = model(audio[None, :], fs=fs)[0]
#     filename = noisy_file.split("/")[-1]
#     save_audio(f"/root/autodl-tmp/inferred_output_vctkdemucs_pesq/step_0/{filename}", inferred[0], fs)
    
inference_steps = 20
candidate_each_step = 10

for i in tqdm(range(inference_steps)):
    target_dir = f"/root/autodl-fs/clean_testset_wav"
    noisy_dir = f"/root/autodl-fs/noisy_testset_wav"
    prev_step_dir = f"/root/autodl-tmp/inferred_output_vctkdemucs_pesq/step_{i}"
    current_step_dir = f"/root/autodl-tmp/inferred_output_vctkdemucs_pesq/step_{i + 1}"
    os.makedirs(current_step_dir, exist_ok=True)
    target_files = []
    clean_files = []
    noisy_files = []
    
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".wav"):
                target_files.append(os.path.join(root, file))
                clean_files.append(os.path.join(prev_step_dir, file))
                noisy_files.append(os.path.join(noisy_dir, file))
    
    all_sdrs = []
    
    candidate_ratios = [0.1 * i for i in range(candidate_each_step)]
    
    for target_file, clean_file, noisy_file in tqdm(zip(target_files, clean_files, noisy_files), total=len(target_files)):
        fs = 48000
        target_audio, _ = librosa.load(target_file, sr=fs)
        candidate_inferred = []
        clean_audio, _ = librosa.load(clean_file, sr=fs)
        candidate_inferred.append(np.expand_dims(clean_audio, axis=0))
        noisy_audio, _ = librosa.load(noisy_file, sr=fs)
        for ratio in candidate_ratios:
            mixed = mix_inferred_with_noisy(clean_audio, noisy_audio, ratio)
            if np.max(np.abs(mixed)) > 1.0:
                mixed = mixed / np.max(np.abs(mixed))
            inferred = model(mixed[None, :], fs=fs)[0]
            candidate_inferred.append(inferred)
        sdrs = [pesq_score(target_audio, inferred) for inferred in candidate_inferred]
        best_index = np.argmax(sdrs)
        best_inferred = candidate_inferred[best_index]
        save_audio(f"{current_step_dir}/{clean_file.split('/')[-1]}", best_inferred[0], fs)
        all_sdrs.append(sdrs[best_index])