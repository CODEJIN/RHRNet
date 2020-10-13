import torch
import numpy as np
import yaml, logging, math

import torchaudio

class RHRNet(torch.nn.Module):
    def __init__(self, hyper_parameters):
        super(RHRNet, self).__init__()
        self.hp = hyper_parameters
        self.layer_Dict = torch.nn.ModuleDict()

        self.layer_Dict['Start'] = torch.nn.GRU(
            input_size= 1,
            hidden_size= self.hp.GRU_Size[0] // 4,
            num_layers= 1,
            batch_first= True,
            bidirectional= True
            )
        self.layer_Dict['PReLU_Start'] = torch.nn.PReLU()

        for index, size in enumerate(self.hp.GRU_Size):
            self.layer_Dict['GRU_{}'.format(index)] = torch.nn.GRU(
                input_size= size,
                hidden_size= size // 2,
                num_layers= 1,
                batch_first= True,
                bidirectional= True
                )
            self.layer_Dict['PReLU_{}'.format(index)] = torch.nn.PReLU()

        self.layer_Dict['Last'] = torch.nn.GRU(
            input_size= self.hp.GRU_Size[-1] // 2,
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
        x = self.layer_Dict['Start'](x)[0]   # [Batch, Time, 2]
        x = self.layer_Dict['PReLU_Start'](x)
        x = x.reshape(x.size(0), x.size(1) // 2, x.size(2) * 2) # [Batch, Time / 2, 4]
        
        stacks = []
        for index in range(0, len(self.hp.GRU_Size) // 2):
            x = self.layer_Dict['GRU_{}'.format(index)](x)[0]   # [Batch, Time, Dim / 2 * 2]
            x = self.layer_Dict['PReLU_{}'.format(index)](x)
            stacks.append(x)
            x = x.reshape(x.size(0), x.size(1) // 2, x.size(2) * 2) # [Batch, Time / 2, Dim / 2 * 2 * 2]

        x = self.layer_Dict['GRU_{}'.format(len(self.hp.GRU_Size) // 2)](x)[0]   # [Batch, Time, Dim / 2 * 2]
        x = self.layer_Dict['PReLU_{}'.format(len(self.hp.GRU_Size) // 2)](x)
        x = x.reshape(x.size(0), x.size(1) * 2, x.size(2) // 2)

        for index in range(len(self.hp.GRU_Size) // 2 + 1, len(self.hp.GRU_Size)):
            x = self.layer_Dict['GRU_{}'.format(index)](x)[0]            
            x = self.layer_Dict['PReLU_{}'.format(index)](x + stacks.pop())   # Residual
            x = x.reshape(x.size(0), x.size(1) * 2, x.size(2) // 2)

        x = self.layer_Dict['Last'](x)[0]   # [Batch, Time, 1]

        return x.squeeze(2)


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