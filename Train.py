import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import numpy as np
import logging, yaml, os, sys, argparse, time, math, importlib
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from random import sample

from Logger import Logger
from Modules import RHRNet, Log_Cosh_Loss
from Datasets import Dataset, Inference_Dataset, Collater, Inference_Collater
from Noam_Scheduler import Modified_Noam_Scheduler
from Radam import RAdam
from Arg_Parser import Recursive_Parse

try:
    from apex import amp
    is_AMP_Exist = True
except:
    logging.info('There is no apex modules in the environment. Mixed precision does not work.')
    is_AMP_Exist = False


logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

class Trainer:
    def __init__(self, hp_path, steps= 0):
        self.hp_Path = hp_path
        self.hp = Recursive_Parse(yaml.load(
            open(self.hp_Path, encoding='utf-8'),
            Loader=yaml.Loader
            ))        
        if not is_AMP_Exist:
            self.hp.Use_Mixed_Precision = False

        if not self.hp.Device is None:
            os.environ['CUDA_VISIBLE_DEVICES']= self.hp.Device

        if not torch.cuda.is_available():
            device = torch.self.device('cpu')
        else:
            self.device = torch.device('cuda:0')
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(0)

        self.steps = steps
        self.epochs = 0

        self.Datset_Generate()
        self.Model_Generate()

        self.scalar_Dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        self.writer_Dict = {
            'Train': Logger(os.path.join(self.hp.Log_Path, 'Train')),
            'Evaluation': Logger(os.path.join(self.hp.Log_Path, 'Evaluation')),
            }
        
        self.Load_Checkpoint()

    def Datset_Generate(self):
        train_Dataset = Dataset(
            wav_paths= self.hp.Train.Train_Pattern.Wav_Paths,
            noise_paths= self.hp.Train.Train_Pattern.Noise_Paths,
            sample_rate= self.hp.Sound.Sample_Rate
            )
        dev_Dataset = Dataset(
            wav_paths= self.hp.Train.Eval_Pattern.Wav_Paths,
            noise_paths= self.hp.Train.Eval_Pattern.Noise_Paths,
            sample_rate= self.hp.Sound.Sample_Rate
            )
        inference_Dataset = Inference_Dataset(
            patterns= [
                line.strip().split('\t')
                for line in open(self.hp.Train.Inference_Pattern_File_in_Train, 'r').readlines()[1:]
                ],
            sample_rate= self.hp.Sound.Sample_Rate
            )

        logging.info('The number of train patterns = {}.'.format(len(train_Dataset)))
        logging.info('The number of development patterns = {}.'.format(len(dev_Dataset)))
        logging.info('The number of inference patterns = {}.'.format(len(inference_Dataset)))

        collater = Collater(wav_length= self.hp.Train.Train_Pattern.Wav_Length, samples= self.hp.Train.Sample_per_Batch)
        inference_Collater = Inference_Collater(reduction= 2 ** (len(self.hp.GRU_Size) // 2 + 1))

        self.dataLoader_Dict = {}
        self.dataLoader_Dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_Dataset,
            shuffle= True,
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataLoader_Dict['Dev'] = torch.utils.data.DataLoader(
            dataset= dev_Dataset,
            shuffle= True,
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataLoader_Dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_Dataset,
            shuffle= False,
            collate_fn= inference_Collater,
            batch_size= self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )

    def Model_Generate(self):
        self.model = RHRNet(hyper_parameters= self.hp).to(self.device)

        self.criterion_Dict = {
            'LC': Log_Cosh_Loss().to(self.device)
            }
        self.optimizer = RAdam(
            params= self.model.parameters(),
            lr= self.hp.Train.Learning_Rate.Initial,
            betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
            eps= self.hp.Train.ADAM.Epsilon,
            weight_decay= self.hp.Train.Weight_Decay
            )
        self.scheduler = Modified_Noam_Scheduler(
            optimizer= self.optimizer,
            base = self.hp.Train.Learning_Rate.Base
            )

        if self.hp.Use_Mixed_Precision:
            self.model, self.optimizer = amp.initialize(
                models= self.model,
                optimizers=self.optimizer
                )

        logging.info(self.model)

    def Train_Step(self, audios, noisies):
        loss_Dict = {}

        audios = audios.to(self.device, non_blocking=True)
        noisies = noisies.to(self.device, non_blocking=True)
        
        predictions = self.model(noisies)
        loss_Dict['LC'] = self.criterion_Dict['LC'](audios, predictions)
        loss_Dict['Total'] = loss_Dict['LC']
        loss = loss_Dict['Total']

        self.optimizer.zero_grad()
        if self.hp.Use_Mixed_Precision:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters= amp.master_params(self.optimizer),
                max_norm= self.hp.Train.Gradient_Norm
                )
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model.parameters(),
                max_norm= self.hp.Train.Gradient_Norm
                )

        self.optimizer.step()
        self.scheduler.step()
        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_Dict.items():
            self.scalar_Dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        for audios, noisies in self.dataLoader_Dict['Train']:
            self.Train_Step(audios, noisies)
            
            if self.steps % self.hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % self.hp.Train.Logging_Interval == 0:
                self.scalar_Dict['Train'] = {
                    tag: loss / self.hp.Train.Logging_Interval
                    for tag, loss in self.scalar_Dict['Train'].items()
                    }
                self.scalar_Dict['Train']['Learning_Rate'] = self.scheduler.get_last_lr()
                self.writer_Dict['Train'].add_scalar_dict(self.scalar_Dict['Train'], self.steps)
                self.scalar_Dict['Train'] = defaultdict(float)

            if self.steps % self.hp.Train.Evaluation_Interval == 0:
                self.Evaluation_Epoch()

            if self.steps % self.hp.Train.Inference_Interval == 0:
                self.Inference_Epoch()
            
            if self.steps >= self.hp.Train.Max_Step:
                return

        self.epochs += 1

    @torch.no_grad()
    def Evaluation_Step(self, audios, noisies):
        loss_Dict = {}

        audios = audios.to(self.device, non_blocking=True)
        noisies = noisies.to(self.device, non_blocking=True)
        
        predictions = self.model(noisies)
        loss_Dict['LC'] = self.criterion_Dict['LC'](audios, predictions)
        loss_Dict['Total'] = loss_Dict['LC']
        loss = loss_Dict['Total']
        
        for tag, loss in loss_Dict.items():
            self.scalar_Dict['Evaluation']['Loss/{}'.format(tag)] += loss

        return predictions
    
    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation.'.format(self.steps))

        self.model.eval()

        for step, (audios, noisies) in tqdm(
            enumerate(self.dataLoader_Dict['Dev'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataLoader_Dict['Dev'].dataset) / self.hp.Train.Batch_Size)
            ):
            predictions = self.Evaluation_Step(audios, noisies)

        self.scalar_Dict['Evaluation'] = {
            tag: loss / step
            for tag, loss in self.scalar_Dict['Evaluation'].items()
            }
        self.writer_Dict['Evaluation'].add_scalar_dict(self.scalar_Dict['Evaluation'], self.steps)
        self.writer_Dict['Evaluation'].add_histogram_model(self.model, self.steps, delete_keywords=['layer_Dict', 'layer'])
        self.scalar_Dict['Evaluation'] = defaultdict(float)

        image_Dict = {
            'Wav/Audio': (audios[-1].cpu().numpy(), None),
            'Wav/Noise': (noisies[-1].cpu().numpy(), None),
            'Wav/Prediction': (predictions[-1].cpu().numpy(), None)
            }
        self.writer_Dict['Evaluation'].add_image_dict(image_Dict, self.steps)

        self.model.train()

    @torch.no_grad()
    def Inference_Step(self, noisies, lengths, labels, start_index= 0, tag_step= False, tag_index= False):
        noisies = noisies.to(self.device, non_blocking=True)

        predictions = self.model(noisies)

        files = []
        for index, label in enumerate(labels):
            tags = []
            if tag_step: tags.append('Step-{}'.format(self.steps))
            tags.append(label)
            if tag_index: tags.append('IDX_{}'.format(index + start_index))
            files.append('.'.join(tags))

        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG').replace('\\', '/'), exist_ok= True)
        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV').replace('\\', '/'), exist_ok= True)
        for index, (noisy, prediction, length, label, file) in enumerate(zip(
            noisies.cpu().numpy(),
            predictions.cpu().numpy(),
            lengths,
            labels,
            files,
            )):
            noisy, prediction = noisy[:length], prediction[:length]
            
            new_Figure = plt.figure(figsize=(10, 10), dpi=100)
            plt.subplot2grid((2, 1), (0, 0))
            plt.plot(noisy)
            plt.margins(x= 0)
            plt.title('Noisy audio    Label: {}'.format(label))
            plt.subplot2grid((2, 1), (1, 0))
            plt.plot(prediction)
            plt.margins(x= 0)
            plt.title('Speech enhanced audio    Label: {}'.format(label))
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG', '{}.PNG'.format(file)).replace('\\', '/'))
            plt.close(new_Figure)

            wavfile.write(
                filename= os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV', '{}.wav'.format(file)).replace('\\', '/'),
                data= (np.clip(prediction, -1.0 + 1e-7, 1.0 - 1e-7) * 32767.5).astype(np.int16),
                rate= self.hp.Sound.Sample_Rate
                )

    def Inference_Epoch(self):
        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        self.model.eval()

        for step, (noisies, lengths, labels) in tqdm(
            enumerate(self.dataLoader_Dict['Inference']),
            desc='[Inference]',
            total= math.ceil(len(self.dataLoader_Dict['Inference'].dataset) / (self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size))
            ):
            self.Inference_Step(noisies, lengths, labels, start_index= step * (self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size))

        self.model.train()

    def Load_Checkpoint(self):
        if self.steps == 0:
            paths = [
                os.path.join(root, file).replace('\\', '/')
                for root, _, files in os.walk(self.hp.Checkpoint_Path)
                for file in files
                if os.path.splitext(file)[1] == '.pt'
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
            else:
                return  # Initial training
        else:
            path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        state_Dict = torch.load(path, map_location= 'cpu')
        self.model.load_state_dict(state_Dict['Model'])
        self.optimizer.load_state_dict(state_Dict['Optimizer'])
        self.scheduler.load_state_dict(state_Dict['Scheduler'])
        self.steps = state_Dict['Steps']
        self.epochs = state_Dict['Epochs']

        if self.hp.Use_Mixed_Precision:
            if not 'AMP' in state_Dict.keys():
                logging.info('No AMP state dict is in the checkpoint. Model regards this checkpoint is trained without mixed precision.')
            else:                
                amp.load_state_dict(state_Dict['AMP'])

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    def Save_Checkpoint(self):
        os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)

        state_Dict = {
            'Model': self.model.state_dict(),
            'Optimizer': self.optimizer.state_dict(),
            'Scheduler': self.scheduler.state_dict(),
            'Steps': self.steps,
            'Epochs': self.epochs,
            }
        if self.hp.Use_Mixed_Precision:
            state_Dict['AMP'] = amp.state_dict()

        torch.save(
            state_Dict,
            os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))
            )

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

    def Train(self):
        hp_Path = os.path.join(self.hp.Checkpoint_Path, 'Hyper_Parameters.yaml').replace('\\', '/')
        if not os.path.exists(hp_Path):
            from shutil import copyfile
            os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
            copyfile(self.hp_Path, hp_Path)

        if self.steps == 0:
            self.Evaluation_Epoch()

        if self.hp.Train.Initial_Inference:
            self.Inference_Epoch()

        self.tqdm = tqdm(
            initial= self.steps,
            total= self.hp.Train.Max_Step,
            desc='[Training]'
            )

        while self.steps < self.hp.Train.Max_Step:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)
            
        self.tqdm.close()
        logging.info('Finished training.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    argParser.add_argument('-s', '--steps', default= 0, type= int)
    args = argParser.parse_args()

    new_Trainer = Trainer(hp_path= args.hyper_parameters, steps= args.steps)
    new_Trainer.Train()