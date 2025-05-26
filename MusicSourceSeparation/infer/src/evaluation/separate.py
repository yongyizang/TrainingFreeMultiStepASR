from os import listdir
from pathlib import Path

import torch
import numpy as np
import onnxruntime as ort
from einops import rearrange
import math
import os
import torch
from src.utils.utils import split_nparray_with_overlap, join_chunks

def mix_inferred_with_noisy(inferred, noisy, ratio):
    if len(inferred) != len(noisy):
        raise ValueError("Inferred and noisy audio must have the same length.")
    
    mixed = (1 - ratio) * noisy + ratio * inferred
    return mixed

def fast_sdr(
    ref: torch.Tensor, 
    est: torch.Tensor, 
    eps: float = 1e-10
):
    r"""Calculate SDR.
    """
    noise = est - ref
    numerator = torch.clamp(torch.mean(ref ** 2, dim=0), min=eps)
    denominator = torch.clamp(torch.mean(noise ** 2, dim=0), min=eps)
    sdr = 10. * torch.log10(numerator / denominator)
    return sdr

def sdr(clean, noisy):
    # use fast_bss_eval
    C = clean.shape[1]
    clean = rearrange(clean, 'b c t -> (b c) t')
    noisy = rearrange(noisy, 'b c t -> (b c) t')
    sdr = fast_sdr(clean.T, noisy.T)
    return rearrange(sdr, '(b c) -> b c', c=C)

def separate_with_onnx(batch_size, model, onnx_path: Path, mix):
    n_sample = mix.shape[1]

    trim = model.n_fft // 2
    gen_size = model.sampling_size - 2 * trim
    pad = gen_size - n_sample % gen_size
    mix_p = np.concatenate((np.zeros((2, trim)), mix, np.zeros((2, pad)), np.zeros((2, trim))), 1)

    mix_waves = []
    i = 0
    while i < n_sample + pad:
        waves = np.array(mix_p[:, i:i + model.sampling_size], dtype=np.float32)
        mix_waves.append(waves)
        i += gen_size
    mix_waves_batched = torch.tensor(mix_waves, dtype=torch.float32).split(batch_size)

    tar_signals = []

    with torch.no_grad():
        _ort = ort.InferenceSession(str(onnx_path))
        for mix_waves in mix_waves_batched:
            tar_waves = model.istft(torch.tensor(
                _ort.run(None, {'input': model.stft(mix_waves).numpy()})[0]
            ))
            tar_signals.append(tar_waves[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).numpy())
        tar_signal = np.concatenate(tar_signals, axis=-1)[:, :-pad]

    return tar_signal


def separate_with_ckpt(batch_size, model, ckpt_path: Path, mix, device, double_chunk):
    model = model.load_from_checkpoint(ckpt_path).to(device)
    if double_chunk:
        inf_ck = model.inference_chunk_size
    else:
        inf_ck = model.sampling_size
    true_samples = inf_ck - 2 * model.trim

    right_pad = true_samples + model.trim - ((mix.shape[-1]) % true_samples)
    mixture = np.concatenate((np.zeros((2, model.trim), dtype='float32'),
                              mix,
                              np.zeros((2, right_pad), dtype='float32')),
                             1)
    num_chunks = mixture.shape[-1] // true_samples
    mix_waves_batched = [mixture[:, i * true_samples: i * true_samples + inf_ck] for i in
                         range(num_chunks)]
    mix_waves_batched = torch.tensor(mix_waves_batched, dtype=torch.float32).split(batch_size)

    target_wav_hats = []

    with torch.no_grad():
        model.eval()
        for mixture_wav in mix_waves_batched:
            mix_spec = model.stft(mixture_wav.to(device))
            spec_hat = model(mix_spec)
            target_wav_hat = model.istft(spec_hat)
            target_wav_hat = target_wav_hat.cpu().detach().numpy()
            target_wav_hats.append(target_wav_hat)

        target_wav_hat = np.vstack(target_wav_hats)[:, :, model.trim:-model.trim]
        target_wav_hat = np.concatenate(target_wav_hat, axis=-1)[:, :mix.shape[-1]]
    return target_wav_hat




def separate_with_onnx_TDF(batch_size, model, onnx_path: Path, mix):
    n_sample = mix.shape[1]

    overlap = model.n_fft // 2
    gen_size = model.inference_chunk_size - 2 * overlap
    pad = gen_size - n_sample % gen_size
    mix_p = np.concatenate((np.zeros((2, overlap)), mix, np.zeros((2, pad)), np.zeros((2, overlap))), 1)

    mix_waves = []
    i = 0
    while i < n_sample + pad:
        waves = np.array(mix_p[:, i:i + model.inference_chunk_size], dtype=np.float32)
        mix_waves.append(waves)
        i += gen_size
    mix_waves_batched = torch.tensor(mix_waves, dtype=torch.float32).split(batch_size)

    tar_signals = []

    with torch.no_grad():
        _ort = ort.InferenceSession(str(onnx_path), providers=['CUDAExecutionProvider'])
        for mix_waves in mix_waves_batched:
            tar_waves = model.istft(torch.tensor(
                _ort.run(None, {'input': model.stft(mix_waves).numpy()})[0]
            ))
            tar_signals.append(tar_waves[:, :, overlap:-overlap].transpose(0, 1).reshape(2, -1).numpy())
        tar_signal = np.concatenate(tar_signals, axis=-1)[:, :-pad]

    return tar_signal



def separate_with_ckpt_TDF(batch_size, model, ckpt_path: Path, mix, prev_inferred, target, candidate_ratios, device, double_chunk, overlap_add):
    '''
    Args:
        batch_size: the inference batch size
        model: the model to be used
        ckpt_path: the path to the checkpoint
        mix: (c, t)
        device: the device to be used
        double_chunk: whether to use double chunk size
    Returns:
        target_wav_hat: (c, t)
    '''
    # checkpoint = torch.load(ckpt_path)
    # model.load_state_dict(checkpoint["state_dict"])
    # model = model.to(device)
    if double_chunk:
        inf_ck = model.inference_chunk_size
    else:
        inf_ck = model.chunk_size

    if overlap_add is None:
        target_wav_hat = no_overlap_inference(model, mix, device, batch_size, inf_ck)
    else:
        if not os.path.exists(overlap_add.tmp_root):
            os.makedirs(overlap_add.tmp_root)
        target_wav_hat = overlap_inference(model, mix, prev_inferred, target, candidate_ratios, device, batch_size, inf_ck, overlap_add.overlap_rate, overlap_add.tmp_root, overlap_add.samplerate)

    return target_wav_hat

def no_overlap_inference(model, mix, device, batch_size, inf_ck):
    true_samples = inf_ck - 2 * model.overlap

    right_pad = true_samples + model.overlap - ((mix.shape[-1]) % true_samples)
    mixture = np.concatenate((np.zeros((model.audio_ch, model.overlap), dtype='float32'),
                              mix,
                              np.zeros((model.audio_ch, right_pad), dtype='float32')),
                             1)
    num_chunks = mixture.shape[-1] // true_samples
    mix_waves_batched = [mixture[:, i * true_samples: i * true_samples + inf_ck] for i in
                         range(num_chunks)]
    mix_waves_batched = np.array(mix_waves_batched, dtype=np.float32)
    mix_waves_batched = torch.tensor(mix_waves_batched, dtype=torch.float32).split(batch_size)

    target_wav_hats = []

    with torch.no_grad():
        model.eval()
        for mixture_wav in mix_waves_batched:
            mix_spec = model.stft(mixture_wav.to(device))
            spec_hat = model(mix_spec)
            target_wav_hat = model.istft(spec_hat)
            target_wav_hat = target_wav_hat.cpu().detach().numpy()
            target_wav_hats.append(target_wav_hat) # (b, c, t)

        target_wav_hat = np.vstack(target_wav_hats)[:, :, model.overlap:-model.overlap] # (sum(b), c, t)
        target_wav_hat = np.concatenate(target_wav_hat, axis=-1)[:, :mix.shape[-1]]
    return target_wav_hat


def overlap_inference(model, mix, prev_inferred, target, candidate_ratios, device, batch_size, inf_ck, overlap_rate, tmp_root, samplerate):
    '''
    Args:
        mix: (c, t)
        prev_inferred: (c, t)
        target: (c, t)
    '''
    hop_length = math.ceil((1 - overlap_rate) * inf_ck)
    overlap_size = inf_ck - hop_length
    step_t = mix.shape[1]
    
    # Split inputs into overlapping chunks
    mix_waves_batched = split_nparray_with_overlap(mix.T, hop_length, overlap_size)
    mix_waves_batched = np.array(mix_waves_batched, dtype=np.float32)
    mix_waves_batched = torch.tensor(mix_waves_batched, dtype=torch.float32).split(batch_size) # [(b, c, t)]
    
    prev_inferred_waves_batched = split_nparray_with_overlap(prev_inferred.T, hop_length, overlap_size)
    prev_inferred_waves_batched = np.array(prev_inferred_waves_batched, dtype=np.float32)
    prev_inferred_waves_batched = torch.tensor(prev_inferred_waves_batched, dtype=torch.float32).split(batch_size) # [(b, c, t)]
    
    target_batched = split_nparray_with_overlap(target.T, hop_length, overlap_size)
    target_batched = np.array(target_batched, dtype=np.float32)
    target_batched = torch.tensor(target_batched, dtype=torch.float32).split(batch_size) # [(b, c, t)]

    target_wav_hats = []

    with torch.no_grad():
        model.eval()
        for mixture_wav, prev_inferred_wav, target_wav in zip(mix_waves_batched, prev_inferred_waves_batched, target_batched):
            # mixture_wav: (b, c, t)
            # prev_inferred_wav: (b, c, t)
            # target_wav: (b, c, t)
            batch_size_curr = mixture_wav.shape[0]
            num_channels = mixture_wav.shape[1]
            
            candidates_inferred = []
            # First candidate is the previous inferred signal
            # candidates_inferred.append(prev_inferred_wav.cpu().detach())
            
            # Generate candidates with different mixing ratios
            for ratio in candidate_ratios:
                mixed = mix_inferred_with_noisy(prev_inferred_wav.to(device), mixture_wav.to(device), ratio)
                # Assuming mixed is already a tensor on device
                mix_spec = model.stft(mixed)
                spec_hat = model(mix_spec)
                target_wav_hat = model.istft(spec_hat)
                candidates_inferred.append(target_wav_hat.cpu().detach())
            
            # Stack all candidates: (num_candidates, b, c, t)
            candidates_inferred = torch.stack(candidates_inferred, dim=0)
            num_candidates = candidates_inferred.shape[0]
            
            # Calculate SDR for each candidate
            sdrs = []
            for candidate in candidates_inferred:
                sdr_values = sdr(target_wav, candidate)  # Assuming this returns (b, c) or similar
                sdrs.append(sdr_values)
            sdrs = torch.stack(sdrs, dim=0)  # (num_candidates, b, c)
            
            # Find best candidate for each (batch, channel) position
            best_indices = torch.argmax(sdrs, dim=0)  # (b, c)
            
            # Select best candidate for each position
            # We need to use advanced indexing here
            batch_indices = torch.arange(batch_size_curr).unsqueeze(1).expand(-1, num_channels)  # (b, c)
            channel_indices = torch.arange(num_channels).unsqueeze(0).expand(batch_size_curr, -1)  # (b, c)
            
            # Index into candidates_inferred: (num_candidates, b, c, t)
            target_wav_hat = candidates_inferred[best_indices, batch_indices, channel_indices]  # (b, c, t)
            
            target_wav_hats.append(target_wav_hat.numpy())

        # Concatenate all batches
        target_wav_hat = np.vstack(target_wav_hats)  # (sum(b), c, t)
        target_wav_hat = np.transpose(target_wav_hat, (0, 2, 1))  # (sum(b), t, c)
        
        # Join overlapping chunks back together
        target_wav_hat = join_chunks(tmp_root, target_wav_hat, samplerate, overlap_size)  # (t, c)
    
    return target_wav_hat[:step_t].T  # (c, t)