import copy
import time
import datetime
import torch
import ray

import models
from data import SplitDataset, make_data_loader, BatchDataset
from logger import Logger
from metrics import Metric
from utils import make_optimizer, collate, to_device

@ray.remote(num_gpus=0.25)
class TransformerClient:
    def __init__(self, log_path, cfg):
        self.dataset = None
        self.local_parameters = None
        self.m = None
        self.start_time = None
        self.num_active_users = None
        self.optimizer = None
        self.model = None
        self.lr = None
        self.label_split = None
        self.data_loader = None
        self.model_rate = None
        self.client_id = None
        cfg = ray.get(cfg[0])
        self.metric = Metric()
        self.logger = Logger(log_path)
        self.cfg = cfg

    def update(self, client_id, dataset_ref, model_ref):
        dataset = ray.get(dataset_ref['dataset'])
        data_split = ray.get(dataset_ref['split'])
        label_split = ray.get(dataset_ref['label_split'])
        local_parameters = ray.get(model_ref['local_params'])
        dataset = SplitDataset(dataset, data_split[client_id])
        self.dataset = BatchDataset(dataset, self.cfg['bptt'])
        self.local_parameters = local_parameters
        self.client_id = client_id
        self.model_rate = model_ref['model_rate']
        self.label_split = label_split
        self.lr = model_ref['lr']
        self.metric = Metric()

    def step(self, m, num_active_users, start_time):
        cfg = self.cfg
        self.model = models.transformer(model_rate=self.model_rate, cfg=self.cfg).to('cuda')
        self.model.load_state_dict(self.local_parameters)
        self.model.train(True)
        self.optimizer = make_optimizer(self.model, self.lr)
        self.m = m
        self.num_active_users = num_active_users
        self.start_time = start_time
        for local_epoch in range(1, cfg['num_epochs']['local'] + 1):
            for i, step_input in enumerate(self.dataset):
                input_size = step_input['label'].size(0)
                step_input['label_split'] = torch.tensor(self.label_split[self.client_id])
                step_input = to_device(step_input, cfg['device'])
                self.model.zero_grad()
                output = self.model(step_input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                # print(step_input['label'].shape, output['score'].shape)
                evaluation = self.metric.evaluate(cfg['metric_name']['train']['Local'], step_input, output)
                self.logger.append(evaluation, 'train', n=input_size)
            self.log(local_epoch, cfg)
        return self.pull()

    def pull(self):
        model_state = {k: v.detach().clone() for k, v in self.model.to(self.cfg['device']).state_dict().items()}
        return model_state

    def log(self, epoch, cfg):
        if self.m % int((self.num_active_users * cfg['log_interval']) + 1) == 0:
            local_time = (time.time() - self.start_time) / (self.m + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (self.num_active_users - self.m - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['num_epochs']['global'] - epoch) * local_time * self.num_active_users))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * self.m / self.num_active_users),
                             'ID: {}({}/{})'.format(self.client_id, self.m + 1, self.num_active_users),
                             'Learning rate: {}'.format(self.lr),
                             'Rate: {}'.format(self.model_rate),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            self.logger.append(info, 'train', mean=False)
            self.logger.write('train', cfg['metric_name']['train']['Local'])

