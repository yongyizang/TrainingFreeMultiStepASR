import os
import numpy as np
import pandas as pd
import librosa
import torch
from tqdm import tqdm
import warnings
import utmos
warnings.filterwarnings('ignore')

from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

utmos_model = utmos.Score()

dnsmos_model = DeepNoiseSuppressionMeanOpinionScore(
    fs=16000,
    personalized=False,  # Set to True for personalized MOS
    device="cuda"
)

def evaluate_audio_file_speechmos(filepath):
    """Calculate UTMOS and DNSMOS-P808 metrics using speechmos package (easier)"""
    try:
        # Load audio at 16kHz
        audio, sr = librosa.load(filepath, sr=16000, mono=True)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
        dnsmos_scores = dnsmos_model(audio_tensor.cuda())
        utmos_score = utmos_model.calculate_wav(audio_tensor, 16000)
        
        return {
            'filename': os.path.basename(filepath),
            'utmos': utmos_score.item(),
            'dnsmos_p808': dnsmos_scores[0].item(),
            'dnsmos_sig': dnsmos_scores[1].item(),
            'dnsmos_bak': dnsmos_scores[2].item(),
            'dnsmos_ovrl': dnsmos_scores[3].item()
        }
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def evaluate_audio_file_torchmetrics(filepath, dnsmos_model):
    """Alternative: Calculate DNSMOS using torchmetrics (also easy)"""
    try:
        # Load audio at 16kHz
        audio, sr = librosa.load(filepath, sr=16000, mono=True)
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # DNSMOS evaluation using torchmetrics
        # Returns tensor with [p808_mos, mos_sig, mos_bak, mos_ovr]
        dnsmos_scores = dnsmos_model(audio_tensor)
        
        # UTMOS using torch.hub
        utmos_model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        audio_tensor_batch = audio_tensor.unsqueeze(0)
        utmos_score = utmos_model(audio_tensor_batch, sr)
        
        return {
            'filename': os.path.basename(filepath),
            'utmos': utmos_score.item(),
            'dnsmos_p808': dnsmos_scores[0].item(),
            'dnsmos_sig': dnsmos_scores[1].item(),
            'dnsmos_bak': dnsmos_scores[2].item(),
            'dnsmos_ovrl': dnsmos_scores[3].item()
        }
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    
    for step_count in tqdm(range(21)):
        # Configuration
        audio_folder = "/root/autodl-tmp/inferred_output_vctkdemucs_pesq/step_" + str(step_count)
        use_speechmos = True  # Set to False to use torchmetrics instead
        
        print("Processing audio files in folder:", audio_folder)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Get list of audio files
        audio_extensions = ('.wav', '.mp3', '.flac', '.m4a')
        audio_files = [f for f in os.listdir(audio_folder) 
                    if f.lower().endswith(audio_extensions)]
        
        print(f"Found {len(audio_files)} audio files to process")
        
        # Process files with progress bar
        results = []
        for filename in tqdm(audio_files, desc="Evaluating audio files"):
            filepath = os.path.join(audio_folder, filename)
            
            if use_speechmos:
                result = evaluate_audio_file_speechmos(filepath)
            else:
                result = evaluate_audio_file_torchmetrics(filepath, dnsmos_model)
            
            if result is not None:
                results.append(result)
        
        if not results:
            print("No results found. Check file paths.")
            return
        
        # Calculate mean scores
        utmos_mean = np.mean([r['utmos'] for r in results])
        dnsmos_ovrl_mean = np.mean([r['dnsmos_ovrl'] for r in results])
        dnsmos_sig_mean = np.mean([r['dnsmos_sig'] for r in results])
        dnsmos_bak_mean = np.mean([r['dnsmos_bak'] for r in results])
        dnsmos_p808_mean = np.mean([r.get('dnsmos_p808', r['dnsmos_ovrl']) for r in results])
        
        print(f"\nProcessed {len(results)} files successfully")
        print(f"Mean UTMOS: {utmos_mean:.4f}")
        print(f"Mean DNSMOS-P808 OVRL: {dnsmos_ovrl_mean:.4f}")
        print(f"Mean DNSMOS-P808 SIG: {dnsmos_sig_mean:.4f}")
        print(f"Mean DNSMOS-P808 BAK: {dnsmos_bak_mean:.4f}")
        print(f"Mean DNSMOS-P808: {dnsmos_p808_mean:.4f}")
        
        # Save detailed results to CSV
        df = pd.DataFrame(results)
        df.to_csv('utmos_dnsmos_results.csv', index=False)
        print("\nDetailed results saved to utmos_dnsmos_results.csv")
        
        # Save summary statistics
        metrics = ['UTMOS', 'DNSMOS-P808 OVRL', 'DNSMOS-P808 SIG', 'DNSMOS-P808 BAK', 'DNSMOS-P808']
        values = [
            [r['utmos'] for r in results],
            [r['dnsmos_ovrl'] for r in results],
            [r['dnsmos_sig'] for r in results],
            [r['dnsmos_bak'] for r in results],
            [r.get('dnsmos_p808', r['dnsmos_ovrl']) for r in results]
        ]
        
        summary = {
            'metric': metrics,
            'mean': [np.mean(v) for v in values],
            'std': [np.std(v) for v in values],
            'min': [np.min(v) for v in values],
            'max': [np.max(v) for v in values],
            'median': [np.median(v) for v in values]
        }
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv("vctkdemucs_pesq_dnsmos_summary_step_" + str(step_count) + ".csv", index=False)
        # print("Summary statistics saved to utmos_dnsmos_summary.csv")

if __name__ == "__main__":
    main()