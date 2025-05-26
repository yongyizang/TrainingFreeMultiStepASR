import os
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Tuple, List
import multiprocessing as mp
from functools import partial
import csv
import argparse
from tqdm import tqdm
from datetime import datetime

def sdr(
    ref: np.ndarray, 
    est: np.ndarray, 
    eps: float = 1e-10
):
    """Calculate SDR."""
    noise = est - ref
    numerator = np.clip(a=np.mean(ref ** 2), a_min=eps, a_max=None)
    denominator = np.clip(a=np.mean(noise ** 2), a_min=eps, a_max=None)
    sdr = 10. * np.log10(numerator / denominator)
    return sdr


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file and return (audio, sample_rate)."""
    data, sr = sf.read(file_path)
    return data, sr


def calculate_utterance_sdr(ref_audio: np.ndarray, est_audio: np.ndarray) -> float:
    """
    Calculate utterance-level SDR for stereo audio.
    Treats stereo as two mono files and averages the SDR.
    """
    # Ensure both arrays have the same shape
    min_len = min(len(ref_audio), len(est_audio))
    ref_audio = ref_audio[:min_len]
    est_audio = est_audio[:min_len]
    
    return sdr(ref_audio, est_audio)


def calculate_chunk_sdr(ref_audio: np.ndarray, est_audio: np.ndarray, 
                      chunk_duration: float = 1.0, sr: int = 44100) -> float:
    """
    Calculate chunk-level SDR (median of median SDR per channel).
    """
    # Ensure both arrays have the same shape
    min_len = min(len(ref_audio), len(est_audio))
    ref_audio = ref_audio[:min_len]
    est_audio = est_audio[:min_len]
    
    # Calculate chunk size in samples
    chunk_size = int(chunk_duration * sr)

    sdrs = []
    for start in range(0, len(ref_audio) - chunk_size + 1, chunk_size):
        end = start + chunk_size
        sdrs.append(sdr(ref_audio[start:end], est_audio[start:end]))
    
    return np.median(sdrs)


def process_single_song(args):
    """Process a single song and return SDR metrics."""
    song_dir, source_path, target_stem, source_stem, source_ext = args
    
    song_name = song_dir.name
    target_file = song_dir / target_stem
    source_file = source_path / f"{song_name}{source_ext}"
    
    if not target_file.exists() or not source_file.exists():
        print(f"Skipping {song_name}: Files not found")
        return None
    
    print(f"Processing: {song_name}")
    
    try:
        # Load audio files
        ref_audio, ref_sr = load_audio(str(target_file))
        est_audio, est_sr = load_audio(str(source_file))
        
        # Resample if necessary
        if ref_sr != est_sr:
            try:
                import librosa
                est_audio = librosa.resample(est_audio.T, orig_sr=est_sr, target_sr=ref_sr).T
                est_sr = ref_sr
            except ImportError:
                print(f"Error: librosa not installed. Cannot resample {song_name}")
                return None
        
        # Calculate SDR metrics
        utterance_sdr = calculate_utterance_sdr(ref_audio, est_audio)
        chunk_sdr = calculate_chunk_sdr(ref_audio, est_audio, sr=ref_sr)
        
        result = {
            'song_name': song_name,
            'utterance_sdr': utterance_sdr,
            'chunk_sdr': chunk_sdr
        }
        
        print(f"  {song_name} - Utterance SDR: {utterance_sdr:.2f} dB, Chunk SDR: {chunk_sdr:.2f} dB")
        
        return result
        
    except Exception as e:
        print(f"Error processing {song_name}: {e}")
        return None


def process_songs_parallel(target_dir: str, source_dir: str, target_stem: str = "bass.wav",
                         source_stem: str = "bass", source_ext: str = ".flac", 
                         step_num: int = 0, num_workers: int = None) -> List[Dict]:
    """
    Process all songs using multiprocessing and calculate SDR metrics.
    """
    target_path = Path(target_dir)
    source_path = Path(source_dir) / source_stem / f"step_{step_num}"
    
    # Prepare arguments for multiprocessing
    song_dirs = [d for d in target_path.iterdir() if d.is_dir()]
    args_list = [(song_dir, source_path, target_stem, source_stem, source_ext) 
                 for song_dir in song_dirs]
    
    # Set number of workers
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    print(f"Using {num_workers} workers for parallel processing")
    
    # Process songs in parallel
    with mp.Pool(num_workers) as pool:
        results = pool.map(process_single_song, args_list)
    
    # Filter out None results (failed processing)
    results = [r for r in results if r is not None]
    
    return results


def save_results_to_csv(results: List[Dict], output_file: str):
    """Save results to a CSV file."""
    if not results:
        print("No results to save")
        return
    
    # Create CSV file
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['song_name', 'utterance_sdr', 'chunk_sdr']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Calculate SDR metrics for audio files')
    parser.add_argument('--target-dir', type=str, default='/root/autodl-tmp/test',
                        help='Directory containing target audio files')
    parser.add_argument('--source-dir', type=str, default='/root/autodl-tmp/inferred_results',
                        help='Directory containing source audio files')
    # parser.add_argument('--target-stem', type=str, default='drums.wav',
    #                     help='Target stem filename')
    # parser.add_argument('--source-stem', type=str, default='drums',
    #                     help='Source stem name')
    parser.add_argument('--source-ext', type=str, default='.flac',
                        help='Source file extension')
    # parser.add_argument('--step', type=int, default=10,
    #                     help='Step number for source files')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of worker processes (default: CPU count)')
    
    args = parser.parse_args()
    
    stems = ["vocals", "drums", "bass", "other"]
    
    for stem in tqdm(stems):
        for step in tqdm(range(21)):
            # Generate output filename if not provided
            
            
            print(f"Processing files from step_{step}")
            
            # Process all songs with multiprocessing
            results = process_songs_parallel(
                target_dir=args.target_dir,
                source_dir=args.source_dir,
                target_stem=stem + ".wav",
                source_stem=stem,
                source_ext=args.source_ext,
                step_num=step,
                num_workers=args.workers
            )
            
            # Save results to CSV
            output_csv = stem + "_step_" + str(step) + ".csv"
            save_results_to_csv(results, output_csv)
            
            # Calculate overall metrics
            if results:
                utterance_sdrs = [r['utterance_sdr'] for r in results]
                chunk_sdrs = [r['chunk_sdr'] for r in results]
                
                mean_utterance_sdr = np.mean(utterance_sdrs)
                median_chunk_sdr = np.median(chunk_sdrs)
                
                print("\n=== Overall Results ===")
                print(f"Mean Utterance-level SDR: {mean_utterance_sdr:.4f} dB")
                print(f"Median Chunk-level SDR: {median_chunk_sdr:.4f} dB")
                print(f"Number of songs processed: {len(results)}")
                
                # Also save summary to CSV
                summary_file = output_csv.replace('.csv', '_summary.csv')
                with open(summary_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Metric', 'Value'])
                    writer.writerow(['uSDR (dB)', f'{mean_utterance_sdr:.4f}'])
                    writer.writerow(['cSDR (dB)', f'{median_chunk_sdr:.4f}'])
                
                print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()