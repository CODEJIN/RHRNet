import torch
import numpy as np
import yaml, logging, math

import torchaudio

class RHRNet(torch.nn.Module):
    def __init__(self, hyper_parameters):
        super(RHRNet, self).__init__()
        self.hp = hyper_parameters
        self.layer_Dict = torch.nn.ModuleDict()

        previous_Size = 1
        for index, (size, ratio, residual) in enumerate(zip(self.hp.Model.GRU_Size, self.hp.Model.Step_Ratio, self.hp.Model.Residual)):
            self.layer_Dict['GRU_{}'.format(index)] = GRU(
                input_size= previous_Size,
                hidden_size= size,
                num_layers= 1,
                batch_first= True,
                bidirectional= True
                )
            if not residual is None:
                self.layer_Dict['PReLU_{}'.format(index)] = torch.nn.PReLU()

            previous_Size = int(size * 2 / ratio)

        self.layer_Dict['Last'] = GRU(
            input_size= previous_Size,
            hidden_size= 1,
            num_layers= 1,
            batch_first= True,
            bidirectional= False
            )

    def forward(self, x):
        '''
        x: [Batch, Time]
        '''
        x = x.unsqueeze(2)   # [Batch, Time, 1]

        stacks = []
        for index, (ratio, residual) in enumerate(zip(self.hp.Model.Step_Ratio, self.hp.Model.Residual)):            
            if not residual is None:
                x = self.layer_Dict['PReLU_{}'.format(index)](x + stacks[residual])
            x = self.layer_Dict['GRU_{}'.format(index)](x)[0]
            stacks.append(x)
            x = x.reshape(x.size(0), int(x.size(1) * ratio), int(x.size(2) / ratio))

        x = self.layer_Dict['Last'](x)[0]

        return x.squeeze(2)

class GRU(torch.nn.GRU):
    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(parameter)
            elif 'bias' in name:
                torch.nn.init.zeros_(parameter)

# https://github.com/tuantle/regression-losses-pytorch
class Log_Cosh_Loss(torch.nn.Module):
    def forward(self, logits, labels):
        return torch.mean(torch.log(torch.cosh(labels - logits)))

if __name__ == "__main__":
    import yaml
    from Arg_Parser import Recursive_Parse
    hp = Recursive_Parse(yaml.load(
        open('Hyper_Parameters.yaml', encoding='utf-8'),
        Loader=yaml.Loader
        ))  
    net = RHRNet(hp)
    net(torch.randn(3, 1, 1024))