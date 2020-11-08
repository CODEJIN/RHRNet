import torch
import torch.nn.functional as F
import numpy as np
import yaml, math, logging

class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_sizes,
        shift_lengths,
        win_lengths,
        window= torch.hann_window
        ):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.layer_Dict = torch.nn.ModuleDict()

        for index, (fft_Size, shift_Length, win_Length) in enumerate(zip(
            fft_sizes,
            shift_lengths,
            win_lengths
            )):            
            self.layer_Dict['STFTLoss_{}'.format(index)] = STFTLoss(
                fft_size= fft_Size,
                shift_length= shift_Length,
                win_length= win_Length,
                window= window
                )

    def forward(self, x, y):
        spectral_Convergence_Loss = 0.0
        magnitude_Loss = 0.0
        for layer in self.layer_Dict.values():
            new_Spectral_Convergence_Loss, new_Magnitude_Loss = layer(x, y)
            spectral_Convergence_Loss += new_Spectral_Convergence_Loss
            magnitude_Loss += new_Magnitude_Loss

        spectral_Convergence_Loss /= len(self.layer_Dict)
        magnitude_Loss /= len(self.layer_Dict)

        return spectral_Convergence_Loss, magnitude_Loss

class STFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_size,
        shift_length,
        win_length,
        window= torch.hann_window
        ):
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_length = shift_length
        self.win_length = win_length
        self.window = window

    def forward(self, x, y):
        x_Magnitute = self.STFT(x)
        y_Magnitute = self.STFT(y)

        spectral_Convergence_Loss = self.SpectralConvergenceLoss(x_Magnitute, y_Magnitute)
        magnitude_Loss = self.LogSTFTMagnitudeLoss(x_Magnitute, y_Magnitute)
        
        return spectral_Convergence_Loss, magnitude_Loss

    def STFT(self, x):
        x_STFT = torch.stft(
            input= x,
            n_fft= self.fft_size,
            hop_length= self.shift_length,
            win_length= self.win_length,
            window= self.window(self.win_length)
            )
        reals, imags = x_STFT[..., 0], x_STFT[..., 1]

        return torch.sqrt(torch.clamp(reals ** 2 + imags ** 2, min= 1e-7)).transpose(2, 1)

    def LogSTFTMagnitudeLoss(self, x_magnitude, y_magnitude):
        return F.l1_loss(torch.log(x_magnitude), torch.log(y_magnitude))

    def SpectralConvergenceLoss(self, x_magnitude, y_magnitude):
        return torch.norm(y_magnitude - x_magnitude, p='fro') / torch.norm(y_magnitude, p='fro')
