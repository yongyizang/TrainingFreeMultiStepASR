
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import soundfile as sf
import torch
from src.utils.utils import load_wav, get_unique_save_path
from src.utils.omega_resolvers import get_eval_log_dir
from pathlib import Path
from tqdm import tqdm
import numpy as np
import fast_bss_eval
import dotenv
from src.evaluation.separate import separate_with_ckpt_TDF, no_overlap_inference, overlap_inference
dotenv.load_dotenv(override=True)

mode = "other"

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

@hydra.main(config_path="configs/", config_name="infer_" + mode + ".yaml", version_base='1.1')
def main(config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934

    from src.utils import utils

    model = hydra.utils.instantiate(config.model)
    
    ckpt_path = Path(config.ckpt_path)
    
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(config.device)
    
    mixtures = []

    # get all mixture.wav recursively from mixture_path
    for root, dirs, files in os.walk(config.mixture_path):
        for file in files:
            if file.endswith("mixture.wav"):
                mixtures.append(os.path.join(root, file))
                
    print(f"Found {len(mixtures)} mixtures")
    
    mixture_to_target_map = {}
    for mixture in mixtures:
        mixture_lastdirname = os.path.basename(os.path.dirname(mixture))
        if mixture_lastdirname not in mixture_to_target_map:
            mixture_to_target_map[mixture_lastdirname] = {}
        mixture_to_target_map[mixture_lastdirname]["mixture"] = mixture
        mixture_to_target_map[mixture_lastdirname]["target"] = os.path.dirname(mixture) + "/" + mode + ".wav"
        # check if target exists
        if not os.path.exists(mixture_to_target_map[mixture_lastdirname]["target"]):
            print(f"Target {mixture_to_target_map[mixture_lastdirname]['target']} does not exist")
            continue
    
    print("Found {} targets".format(len(mixture_to_target_map)))
    
    inference_steps = 20
    candidate_each_step = 10
    
    for step in tqdm(range(inference_steps)):
        all_sdrs = []
        save_path = "/root/autodl-tmp/inferred_results/" + mode + "/step_" + str(step + 1)
        prev_step = "/root/autodl-tmp/inferred_results/" + mode + "/step_" + str(step)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        
        # ratio is 0 to 0.9
        candidate_ratios = [i * 0.1 for i in range(candidate_each_step)]
                
        for item in tqdm(mixture_to_target_map):
            mixture_path = mixture_to_target_map[item]["mixture"]
            target_path = mixture_to_target_map[item]["target"]
            mixture = load_wav(mixture_path)
            target_audio = load_wav(target_path)
            prev_audio = load_wav(os.path.join(prev_step, os.path.basename(os.path.dirname(mixture_path)) + ".flac"))
            prev_sdr = sdr(target_audio, prev_audio)
            
            inferred = separate_with_ckpt_TDF(config.batch_size, model, ckpt_path, mixture, prev_audio, target_audio, candidate_ratios, config.device,
                                            config.double_chunk, config.overlap_add)
                
            mixture_lastdirname = os.path.basename(os.path.dirname(mixture_path))
            curr_save_path = os.path.join(save_path, mixture_lastdirname + ".flac")
            sf.write(curr_save_path, inferred.T, 44100)
            sdr_value = sdr(target_audio, inferred)
            all_sdrs.append(sdr_value)

if __name__ == "__main__":
    main()
