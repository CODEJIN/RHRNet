import torch
import numpy as np
import yaml, pickle, os, math, logging
from random import choice, randint, sample

from Audio import Audio_Prep

def Calc_RMS(audio):
    return np.sqrt(np.mean(np.square(audio), axis= -1))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, wav_paths, noise_paths, sample_rate):
        super(Dataset, self).__init__()
        self.sample_Rate = sample_rate

        self.files = []
        for path in wav_paths:
            for root, _, files in os.walk(path):
                for file in files:
                    if os.path.splitext(file)[1].lower() != '.wav':
                        continue
                    self.files.append(os.path.join(root, file))

        self.noises = []
        for path in noise_paths:
            for root, _, files in os.walk(path):
                for file in files:
                    if os.path.splitext(file)[1].lower() != '.wav':
                        continue
                    self.noises.append(os.path.join(root, file))

    def __getitem__(self, idx):
        audio = Audio_Prep(self.files[idx], sample_rate= self.sample_Rate)
        noise = Audio_Prep(choice(self.noises), sample_rate= self.sample_Rate)
        
        return audio, noise

    def __len__(self):
        return len(self.files)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(self, patterns, sample_rate):
        super(Inference_Dataset, self).__init__()
            
        self.patterns = patterns
        self.sample_Rate = sample_rate

    def __getitem__(self, idx):        
        label, file = self.patterns[idx]
        noisy = Audio_Prep(file, sample_rate= self.sample_Rate)
        
        return noisy, label

    def __len__(self):
        return len(self.patterns)


class Collater:
    def __init__(self, wav_length, samples):
        self.wav_Length = wav_length
        self.samples = samples

    def __call__(self, batch):
        audios = []
        noises = []
        noisies = []
        for audio, noise in batch:
            if any([x.shape[0] < self.wav_Length * 2 for x in [audio, noise]]):
                continue
            audio_Offsets = sample(range(0, audio.shape[0] - self.wav_Length), self.samples)
            noise_Offsets = sample(range(0, noise.shape[0] - self.wav_Length), self.samples)
            for audio_Offset, noise_Offset in zip(audio_Offsets, noise_Offsets):
                for _ in range(100):
                    audio_Sample = audio[audio_Offset:audio_Offset + self.wav_Length]
                    audio_RMS = Calc_RMS(audio_Sample)
                    if audio_RMS > 0.01:
                        break

                for _ in range(100):
                    noise_Sample = noise[noise_Offset:noise_Offset + self.wav_Length]
                    noise_RMS = Calc_RMS(noise_Sample)
                    if noise_RMS > 0.01:
                        break
                    
                if any([x < 0.01 for x in [audio_RMS, noise_RMS]]):
                    continue

                alpha = audio_RMS / noise_RMS / 10 ** (np.random.uniform(0.0, 20.0) / 20)

                noisy = audio_Sample + alpha * noise_Sample                
                max_Noisy = np.max(np.abs(noisy))
                if max_Noisy > 1.0:
                    audio_Sample /= max_Noisy * 1.01 + 1e-7
                    noise_Sample /= max_Noisy * 1.01 + 1e-7
                    noisy /= max_Noisy * 1.01 + 1e-7
                audios.append(audio_Sample)
                noises.append(noise_Sample)
                noisies.append(noisy)

        audios = torch.FloatTensor(audios)   # [Batch, Time]
        noises = torch.FloatTensor(noises)   # [Batch, Time]
        noisies = torch.FloatTensor(noisies)    # [Batch, Time]

        return audios, noises, noisies

class Inference_Collater:
    def __init__(self, reduction):
        self.reduction = reduction

    def __call__(self, batch):
        noisies, labels = zip(*batch)
        lengths = [noisy.shape[0] for noisy in noisies]
        
        max_Length = math.ceil(max(lengths) / float(self.reduction)) * self.reduction
        noisies = [np.pad(noisy, [0, max_Length - noisy.shape[0]]) for noisy in noisies]

        noisies = torch.FloatTensor(noisies)   # [Batch, Time]

        return noisies, lengths, labels