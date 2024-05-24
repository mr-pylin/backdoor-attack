# dependencies
import torch
from torch import nn
from torch.utils.data import DataLoader

import torchsummary

import matplotlib.pyplot as plt

class CleanModel:
    def __init__(self, model: nn.Module, input_size: tuple[int], device: str | torch.device = 'cpu', verbose: bool = False, summary: bool = False) -> None:
        """
        This class is used to train/test/predict a given model.

        Args:
            - `model` (nn.Module): model structure + initial parameters
            - `input_size` tuple[int]: e.g. (depth, height, width)
            - `device` ( str | torch.device): move the model parameters to the device [default: 'cpu']
            - `verbose` (bool): print the model information, layer by layer [default: False]
            - `summary` (bool): print number of model's parameters, layer by layer [default: False]
        
        Returns:
            None
        """

        self.model = model.to(device)
        self.device = device

        if verbose:
            print(self.model)
        
        if summary:
            print(torchsummary.summary(model, input_size))
    
    def train(self, **kwargs):
        """
        Train & Validate a given model based on the given hyper-parameters

        Args:
            - `kwargs`: configuration dictionary
                - `trainset` (torch.utisl.data.Dataset)
                - `t_batch_size` (int): batch_size for train set
                - `t_shuffle` (bool): shuffle data for each batch in each epoch
                - `t_num_workers` (int): number of workers to make batches ready [default: 1]
                - `t_accuracy_log` (torchmetrics.classification.accuracy.MulticlassAccuracy)
                - `validationset` (torch.utisl.data.Dataset): [default: None]
                - `v_batch_size` (int): batch_size for validation set [default: None]
                - `v_shuffle` (bool): shuffle data for each batch in each epoch [default: False]
                - `v_num_workers` (int): number of workers to make batches ready [default: 1]
                - `v_accuracy_log` (torchmetrics.classification.accuracy.MulticlassAccuracy)
                - `optimizer` (torch.optim)
                - `criterion` (nn.modules.loss._Loss): loss function to be minimized by the optimizer
                - `epochs` (int): number of epochs to learn the model
                - `verbose` (bool): print accuracy and loss for trainset & validationset per epoch [default: True]
                - `plot` (bool): plot accuracy and loss per epoch to analyze the model [default: False]
        
        Returns:
            None
        """
        
        # default parameters value
        defaults = {
            't_num_workers': 1,
            'validationset': None,
            'v_batch_size': None,
            'v_shuffle': False,
            'v_num_workers': 1,
            'verbose': True,
            'plot': False,
        }
        kwargs = {**defaults, **kwargs}

        # create dataloaders + accuracy & loss metric
        trainloader = DataLoader(kwargs['trainset'], batch_size= kwargs['t_batch_size'], shuffle= kwargs['t_shuffle'], num_workers= kwargs['t_num_workers'])
        train_acc = kwargs['t_accuracy_log']
        train_acc_per_epoch  = []
        train_loss_per_epoch = []

        if kwargs['validationset']:
            validationloader = DataLoader(kwargs['validationset'], batch_size= kwargs['v_batch_size'], shuffle= kwargs['v_shuffle'], num_workers= kwargs['v_num_workers'])
            validation_acc = kwargs['v_accuracy_log']
            validation_acc_per_epoch  = []
            validation_loss_per_epoch = []
        
        # training & validating section
        for epoch in range(kwargs['epochs']):

        # train loop
            self.model.train()
            train_loss  = 0

            for x, y in trainloader:

                # send data to GPU
                x, y_true = x.to(self.device), y.to(self.device)

                # forward
                y_pred = self.model(x)
                loss = kwargs['criterion'](y_pred, y_true)

                # backward
                loss.backward()

                # update parameters
                kwargs['optimizer'].step()
                kwargs['optimizer'].zero_grad()

                # log loss & accuracy
                train_loss += loss.item() * len(x)
                train_acc.update(y_pred, y_true)

            train_loss_per_epoch.append(train_loss / len(kwargs['trainset']))
            train_acc_per_epoch.append(train_acc.compute().item())
            train_acc.reset()

            # validation loop
            if kwargs['validationset']:
                self.model.eval()
                validation_loss = 0

                with torch.no_grad():
                    for x, y in validationloader:
                        
                        # send data to GPU
                        x, y_true = x.to(self.device), y.to(self.device)

                        # forward
                        y_pred = self.model(x)
                        loss = kwargs['criterion'](y_pred, y_true)

                        # log loss & accuracy
                        validation_loss += loss.item() * len(x)
                        validation_acc.update(y_pred, y_true)

                validation_loss_per_epoch.append(validation_loss / len(kwargs['validationset']))
                validation_acc_per_epoch.append(validation_acc.compute().item())
                validation_acc.reset()

            if kwargs['verbose']:
                if kwargs['validationset']:
                    print(f"epoch {epoch:>1}  ->  train[loss: {train_loss_per_epoch[epoch]:.5f} - acc: {train_acc_per_epoch[epoch]:.2f}] | validation[loss: {validation_loss_per_epoch[epoch]:.5f} - acc: {validation_acc_per_epoch[epoch]:.2f}]")
                else:
                    print(f"epoch {epoch:>1}  ->  train[loss: {train_loss_per_epoch[epoch]:.5f} - acc: {train_acc_per_epoch[epoch]:.2f}]")
        
        # plot loss and accuracy for each epoch
        if kwargs['plot']:
            if kwargs['validationset']:
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), layout='compressed')
                fig.suptitle('Train & Validation Analysis')
                axs[0].plot(train_loss_per_epoch, label= 'train')
                axs[0].plot(validation_loss_per_epoch, label= 'validation')
                axs[0].set_title('Loss')
                axs[1].plot(train_acc_per_epoch, label= 'train')
                axs[1].plot(validation_acc_per_epoch, label= 'validation')
                axs[1].set_title('Accuracy')
                plt.legend()
            else:
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), layout='compressed')
                fig.suptitle('Train Analysis')
                axs[0].plot(train_loss_per_epoch)
                axs[0].set_title('Loss')
                axs[1].plot(train_acc_per_epoch)
                axs[1].set_title('Accuracy')
            
            plt.show()


    def test(self, **kwargs) -> tuple[list[int], list[int]]:
        """
        Test a given model

        Args:
            - `kwargs`: configuration dictionary
                - `testset` (torch.utisl.data.Dataset)
                - `batch_size` (int): batch_size for test set
                - `shuffle` (bool): shuffle data for each batch in each epoch
                - `num_workers` (int): number of workers to make batches ready [default: 1]
                - `accuracy_log` (torchmetrics.classification.accuracy.MulticlassAccuracy)
                - `criterion` (nn.modules.loss._Loss): loss function to be minimized by the optimizer
                - `verbose` (bool): print accuracy and loss for trainset & validationset per epoch [default: True]
        
        Returns:
            tuple[list[int], list[int]]
        """

        # default parameters value
        defaults = {
            'num_workers': 1,
            'verbose': True,
        }
        kwargs = {**defaults, **kwargs}

        # create dataloader + accuracy & loss metric
        testloader = DataLoader(kwargs['testset'], batch_size= kwargs['batch_size'], shuffle= kwargs['shuffle'], num_workers= kwargs['num_workers'])
        test_acc = kwargs['accuracy_log']

        # test loop
        test_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for x, y in testloader:
                
                # send data to GPU
                x, y_true = x.to(self.device), y.to(self.device)

                # forward
                y_pred = self.model(x)
                loss = kwargs['criterion'](y_pred, y_true)

                # log loss & accuracy
                test_loss += loss.item() * len(x)
                test_acc.update(y_pred, y_true)

                predictions.extend(y_pred.argmax(dim= 1).cpu())
                targets.extend(y_true.cpu())
            
            total_loss = test_loss / len(kwargs['testset'])
            total_acc  = test_acc.compute().item()
            test_acc.reset()
        
        predictions = torch.tensor(predictions)
        targets = torch.tensor(targets)
        
        if kwargs['verbose']:
            print(f"test[loss: {total_loss:.5f} - acc: {total_acc:.2f}]")
        
        return (targets, predictions)

    def predict(self):
        pass