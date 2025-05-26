import os
from abc import ABCMeta, ABC
from pathlib import Path

import soundfile
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from tqdm import tqdm

from src.utils.utils import load_wav
from src import utils
import numpy as np

log = utils.get_pylogger(__name__)

def check_target_name(target_name, source_names):
    try:
        assert target_name is not None
    except AssertionError:
        print('[ERROR] please identify target name. ex) +datamodule.target_name="vocals"')
        exit(-1)
    try:
        assert target_name in source_names or target_name == 'all'
    except AssertionError:
        print('[ERROR] target name should one of "bass", "drums", "other", "vocals", "all"')
        exit(-1)


def check_sample_rate(sr, sample_track):
    try:
        sample_rate = soundfile.read(sample_track)[1]
        assert sample_rate == sr
    except AssertionError:
        sample_rate = soundfile.read(sample_track)[1]
        print('[ERROR] sampling rate mismatched')
        print('\t=> sr in Config file: {}, but sr of data: {}'.format(sr, sample_rate))
        exit(-1)


class MusdbDataset(Dataset):
    __metaclass__ = ABCMeta

    def __init__(self, data_dir, chunk_size):
        self.source_names = ['bass', 'drums', 'other', 'vocals']
        self.chunk_size = chunk_size
        self.musdb_path = Path(data_dir)


class MusdbTrainDataset(MusdbDataset):
    def __init__(self, data_dir, chunk_size, target_name, aug_params, external_datasets, single_channel, epoch_size):
        super(MusdbTrainDataset, self).__init__(data_dir, chunk_size)

        self.single_channel = single_channel
        self.neg_lst = [x for x in self.source_names if x != target_name]

        self.target_name = target_name
        check_target_name(self.target_name, self.source_names)

        if not self.musdb_path.joinpath('metadata').exists():
            os.mkdir(self.musdb_path.joinpath('metadata'))

        splits = ['train']
        if external_datasets is not None:
            splits += external_datasets

        # collect paths for datasets and metadata (track names and duration)
        datasets, metadata_caches = [], []
        raw_datasets = []    # un-augmented datasets
        for split in splits:
            raw_datasets.append(self.musdb_path.joinpath(split))
            max_pitch, max_tempo = aug_params
            for p in range(-max_pitch, max_pitch+1):
                for t in range(-max_tempo, max_tempo+1, 10):
                    aug_split = split if p==t==0 else split + f'_p={p}_t={t}'
                    datasets.append(self.musdb_path.joinpath(aug_split))
                    metadata_caches.append(self.musdb_path.joinpath('metadata').joinpath(aug_split + '.pkl'))

        # collect all track names and their duration
        self.metadata = []
        raw_track_lengths = []   # for calculating epoch size
        for i, (dataset, metadata_cache) in enumerate(tqdm(zip(datasets, metadata_caches))):
            try:
                metadata = torch.load(metadata_cache)
            except FileNotFoundError:
                print('creating metadata for', dataset)
                metadata = []
                for track_name in sorted(os.listdir(dataset)):
                    track_path = dataset.joinpath(track_name)
                    track_length = load_wav(track_path.joinpath('vocals.wav')).shape[-1]
                    metadata.append((track_path, track_length))
                torch.save(metadata, metadata_cache)

            self.metadata += metadata
            if dataset in raw_datasets:
                raw_track_lengths += [length for path, length in metadata]

        self.epoch_size = sum(raw_track_lengths) // self.chunk_size if epoch_size is None else epoch_size
        log.info(f'epoch size: {self.epoch_size}')

    def __getitem__(self, _):
        sources = []
        for source_name in self.source_names:
            track_path, track_length = random.choice(self.metadata)   # random mixing between tracks
            source = load_wav(track_path.joinpath(source_name + '.wav'),
                              track_length=track_length, chunk_size=self.chunk_size) # (2, times)
            sources.append(source)

        mix = sum(sources)

        if self.target_name == 'all':
            # Targets for models that separate all four sources (ex. Demucs).
            # This adds additional 'source' dimension => batch_shape=[batch, source, channel, time]
            target = sources
        else:
            target = sources[self.source_names.index(self.target_name)]

        mix, target = torch.tensor(mix), torch.tensor(target)
        if self.single_channel:
            mix = torch.mean(mix, dim=0, keepdim=True)
            target = torch.mean(target, dim=0, keepdim=True)
        return mix, target

    def __len__(self):
        return self.epoch_size


class MusdbValidDataset(MusdbDataset):

    def __init__(self, data_dir, chunk_size, target_name, overlap, batch_size, single_channel):
        super(MusdbValidDataset, self).__init__(data_dir, chunk_size)

        self.target_name = target_name
        check_target_name(self.target_name, self.source_names)

        self.overlap = overlap
        self.batch_size = batch_size
        self.single_channel = single_channel

        musdb_valid_path = self.musdb_path.joinpath('valid')
        self.track_paths = [musdb_valid_path.joinpath(track_name)
                            for track_name in os.listdir(musdb_valid_path)]

    def __getitem__(self, index):
        mix = load_wav(self.track_paths[index].joinpath('mixture.wav')) # (2, time)

        if self.target_name == 'all':
            # Targets for models that separate all four sources (ex. Demucs).
            # This adds additional 'source' dimension => batch_shape=[batch, source, channel, time]
            target = [load_wav(self.track_paths[index].joinpath(source_name + '.wav'))
                      for source_name in self.source_names]
        else:
            target = load_wav(self.track_paths[index].joinpath(self.target_name + '.wav'))

        chunk_output_size = self.chunk_size - 2 * self.overlap
        left_pad = np.zeros([2, self.overlap])
        right_pad = np.zeros([2, self.overlap + chunk_output_size - (mix.shape[-1] % chunk_output_size)])
        mix_padded = np.concatenate([left_pad, mix, right_pad], 1)

        num_chunks = mix_padded.shape[-1] // chunk_output_size
        mix_chunks = np.array([mix_padded[:, i * chunk_output_size: i * chunk_output_size + self.chunk_size]
                      for i in range(num_chunks)])
        mix_chunk_batches = torch.tensor(mix_chunks, dtype=torch.float32).split(self.batch_size)
        target = torch.tensor(target)

        if self.single_channel:
            mix_chunk_batches = [torch.mean(t, dim=1, keepdim=True) for t in mix_chunk_batches]
            target = torch.mean(target, dim=0, keepdim=True)

        return mix_chunk_batches, target

    def __len__(self):
        return len(self.track_paths)