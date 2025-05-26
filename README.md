# Training-Free Multi-Step Audio Source Separation

Official implementation of **"Training-Free Multi-Step Audio Source Separation"**

We reveal that pretrained one-step audio source separation models can be leveraged for multi-step separation without additional training. Our simple yet effective inference method iteratively applies separation by optimally blending the input mixture with the previous step's separation result. At each step, we determine the optimal blending ratio by maximizing a metric.

Note that the code is for research purposes and is thus very noisy and not well-structured. We will not provide support for running the code, but we will try to answer questions related to the paper. Running the code directly without any change should yield the exact same results as reported in paper.

## Structure
You may need to change the path for your dataset.

Run experiments for speech enhancement models using:
`python ./SpeechEnhancement/infer/infer_sdr.py` for SDR-based search,
`python ./SpeechEnhancement/infer/infer_pesq.py` for PESQ-based search,
`python ./SpeechEnhancement/infer/infer_large.py` for large or xlarge model variants,
and `python ./SpeechEnhancement/infer/infer_blind_utmos.py` for blind UTMOS search.

Run experiments for music separation models using:
`python ./MusicSourceSeparation/infer/run_infer.py` (note that the code is largely a direct copy from the original DTTNet repository, with key changes in `MusicSourceSeparation/infer/src/evaluation/separate.py`)

Evaluation code is in respective folders as well (see `SpeechEnhancement/eval` and `MusicSourceSeparation/eval.py`).