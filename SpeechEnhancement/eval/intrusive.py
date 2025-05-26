import os
import numpy as np
import pandas as pd
import librosa
from pystoi import stoi
from pesq import pesq
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

def calculate_si_snr(reference, estimate):
    """Calculate SI-SNR between two signals"""
    ref = reference - np.mean(reference)
    est = estimate - np.mean(estimate)
    
    alpha = np.dot(ref, est) / np.dot(ref, ref)
    s_target = alpha * ref
    e_noise = est - s_target
    
    return 10 * np.log10(np.dot(s_target, s_target) / np.dot(e_noise, e_noise))

def evaluate_audio_pair(filename, reference_folder, estimate_folder):
    """Calculate STOI, PESQ, and SI-SNR for a pair of audio files"""
    try:
        ref_path = os.path.join(reference_folder, filename)
        est_path = os.path.join(estimate_folder, filename)
        
        if not os.path.exists(est_path):
            return None
            
        # Load audio
        ref_16k, _ = librosa.load(ref_path, sr=16000, mono=True)
        est_16k, _ = librosa.load(est_path, sr=16000, mono=True)
        
        # Match lengths
        min_len = min(len(ref_16k), len(est_16k))
        ref_16k = ref_16k[:min_len]
        est_16k = est_16k[:min_len]
        
        # Calculate metrics
        stoi_score = stoi(ref_16k, est_16k, fs_sig=16000)
        estoi_score = stoi(ref_16k, est_16k, fs_sig=16000, extended=True)
        pesq_score = pesq(16000, ref_16k, est_16k, 'wb')
        si_snr_score = calculate_si_snr(ref_16k, est_16k)
        
        return {
            'filename': filename,
            'stoi': stoi_score,
            'estoi': estoi_score,
            'pesq': pesq_score,
            'si_snr': si_snr_score
        }
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def process_batch(filenames, reference_folder, estimate_folder):
    """Process a batch of files"""
    results = []
    for filename in filenames:
        result = evaluate_audio_pair(filename, reference_folder, estimate_folder)
        if result is not None:
            results.append(result)
    return results

def main():
    reference_folder = "/root/autodl-fs/clean_testset_wav"  # UPDATE THIS
    for step_count in tqdm(range(21)):
        estimate_folder = "/root/autodl-tmp/inferred_output_vctkdemucs_pesq/step_" + str(step_count)   # UPDATE THIS
        
        # Get list of WAV files
        wav_files = [f for f in os.listdir(reference_folder) if f.endswith('.wav')]
        print(f"Found {len(wav_files)} audio files to process")
        
        # Set number of workers (leave one core free for system)
        num_workers = 4
        print(f"Using {num_workers} workers for parallel processing")
        
        # Create a partial function with fixed reference and estimate folders
        evaluate_func = partial(evaluate_audio_pair, 
                            reference_folder=reference_folder, 
                            estimate_folder=estimate_folder)
        
        # Process files in parallel with progress bar
        results = []
        with Pool(num_workers) as pool:
            with tqdm(total=len(wav_files), desc="Processing audio files") as pbar:
                for result in pool.imap_unordered(evaluate_func, wav_files):
                    if result is not None:
                        results.append(result)
                    pbar.update(1)
        
        if not results:
            print("No results found. Check file paths.")
            return
        
        # Calculate mean scores
        stoi_mean = np.mean([r['stoi'] for r in results])
        pesq_mean = np.mean([r['pesq'] for r in results])
        estoi_mean = np.mean([r['estoi'] for r in results])
        si_snr_mean = np.mean([r['si_snr'] for r in results])
        
        print(f"\nProcessed {len(results)} files successfully")
        print(f"Mean STOI: {stoi_mean:.4f}, Mean PESQ: {pesq_mean:.4f}, Mean SI-SNR: {si_snr_mean:.4f}")
        
        # # Save to CSV
        # df = pd.DataFrame(results)
        # df.to_csv('results.csv', index=False)
        # print("\nDone. Results saved to results.csv")
        
        # Save summary statistics
        summary = {
            'metric': ['STOI', 'PESQ', 'SI-SNR'],
            'mean': [stoi_mean, pesq_mean, si_snr_mean],
            'std': [
                np.std([r['stoi'] for r in results]),
                np.std([r['pesq'] for r in results]),
                np.std([r['si_snr'] for r in results])
            ],
            'min': [
                np.min([r['stoi'] for r in results]),
                np.min([r['pesq'] for r in results]),
                np.min([r['si_snr'] for r in results])
            ],
            'max': [
                np.max([r['stoi'] for r in results]),
                np.max([r['pesq'] for r in results]),
                np.max([r['si_snr'] for r in results])
            ]
        }
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv("summary_statistics_" + str(step_count) + ".csv", index=False)

if __name__ == "__main__":
    main()