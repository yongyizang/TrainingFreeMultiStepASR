import soundfile as sf
from espnet2.bin.enh_inference import SeparateSpeech
from tqdm import tqdm
import numpy as np
import torch
from torch_pesq import PesqLoss
import os
import utmos
import librosa

model = SeparateSpeech.from_pretrained(
    model_tag="wyz/vctk_dns2020_whamr_bsrnn_medium_noncausal",
    normalize_output_wav=False,
    device="cuda",
)

utmos_model = utmos.Score()

print("Model loaded successfully.")

def mix_inferred_with_noisy(inferred, noisy, ratio):
    # if len(inferred) != len(noisy):
        # # pad the shorter one with zeros
        # if len(inferred) < len(noisy):
        #     inferred = np.pad(inferred, (0, len(noisy) - len(inferred)), mode='constant')
        # else:
        #     noisy = np.pad(noisy, (0, len(inferred) - len(noisy)), mode='constant')
        
    if len(inferred) != len(noisy):
        raise ValueError("Inferred and noisy audio must have the same length.")
    
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

def utmos_score(est):
    est = torch.tensor(est, dtype=torch.float32)
    mos = utmos_model.calculate_wav(est, 16000)
    return mos.item()

def save_audio(file_path, audio, fs):
    # if clipping, then normalize.
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))
    sf.write(file_path, audio, fs)

noisy_files = []
noisy_dir = "/root/autodl-tmp/V2_V3_DNSChallenge_Blindset/noisy_blind_testset_v3_challenge_withSNR_16k"
            
for root, dirs, files in os.walk(noisy_dir):
    for file in files:
        if file.endswith(".wav"):
            noisy_files.append(os.path.join(root, file))

noisy_files.sort()

print(f"Number of noisy files: {len(noisy_files)}")
    
os.makedirs("/root/autodl-tmp/inferred_output_dnsv3_utmos/step_0", exist_ok=True)
    
# for noisy_file in tqdm((noisy_files), total=len(noisy_files)):
#     audio, fs = sf.read(noisy_file)
#     inferred = model(audio[None, :], fs=fs)[0]
#     filename = noisy_file.split("/")[-1]
#     save_audio(f"/root/autodl-tmp/inferred_output_dnsv3_utmos/step_0/{filename}", inferred[0], fs)
    
inference_steps = 20
candidate_each_step = 10

for i in tqdm(range(inference_steps)):
    noisy_dir = f"/root/autodl-tmp/V2_V3_DNSChallenge_Blindset/noisy_blind_testset_v3_challenge_withSNR_16k"
    prev_step_dir = f"/root/autodl-tmp/inferred_output_dnsv3_utmos/step_{i}"
    current_step_dir = f"/root/autodl-tmp/inferred_output_dnsv3_utmos/step_{i + 1}"
    os.makedirs(current_step_dir, exist_ok=True)
    clean_files = []
    noisy_files = []
    
    for root, dirs, files in os.walk(prev_step_dir):
        for file in files:
            if file.endswith(".wav"):
                clean_files.append(os.path.join(prev_step_dir, file))
                noisy_files.append(os.path.join(noisy_dir, file))
    
    all_sdrs = []
    
    # start_ratio = i / inference_steps
    # ratio_step = (1 - start_ratio) / candidate_each_step
    candidate_ratios = [0.1 * i for i in range(candidate_each_step)]
    
    for clean_file, noisy_file in tqdm(zip(clean_files, noisy_files), total=len(clean_files)):
        fs = 16000
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
        sdrs = [utmos_score(inferred) for inferred in candidate_inferred]
        best_index = np.argmax(sdrs)
        best_inferred = candidate_inferred[best_index]
        save_audio(f"{current_step_dir}/{clean_file.split('/')[-1]}", best_inferred[0], fs)
        all_sdrs.append(sdrs[best_index])