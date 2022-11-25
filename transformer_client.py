import copy
import datetime
import time

import ray
import torch

import models
from datasets.lm import StackOverflowClientDataset
from logger import Logger
from metrics import Metric
from utils import make_optimizer, to_device


@ray.remote(num_gpus=0.8)
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
        dataset = dataset_ref
        label_split = dataset_ref
        local_parameters = model_ref['local_params']
        self.dataset = StackOverflowClientDataset(dataset, self.cfg['seq_length'], self.cfg['batch_size']['train'])
        self.local_parameters = copy.deepcopy(local_parameters)
        self.client_id = client_id
        self.model_rate = model_ref['model_rate']
        self.label_split = label_split
        self.lr = model_ref['lr']
        self.metric = Metric()

    def step(self, m, num_active_users, start_time):
        cfg = self.cfg
        self.model = models.transformer_nwp(model_rate=self.model_rate, cfg=self.cfg).cpu()
        self.model.load_state_dict(self.local_parameters)
        self.model = self.model.cuda()
        self.model.train(True)
        self.optimizer = make_optimizer(self.model, self.lr)
        self.m = m
        self.num_active_users = num_active_users
        self.start_time = start_time
        for local_epoch in range(1, cfg['num_epochs']['local'] + 1):
            for i, step_input in enumerate(self.dataset):
                input_size = step_input['label'].size(0)
                # step_input['label_split'] = None
                step_input = to_device(step_input, cfg['device'])
                self.optimizer.zero_grad()
                output = self.model(step_input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                evaluation = self.metric.evaluate(cfg['metric_name']['train']['Local'], step_input, output)
                self.logger.append(evaluation, 'train', n=input_size)
            self.log(local_epoch, cfg)
        return self.pull()

    def pull(self):
        model_state = {k: v.detach().clone() for k, v in self.model.cpu().state_dict().items()}
        self.model = None
        self.local_parameters = None
        return model_state

    def log(self, epoch, cfg):
        if self.m % int((self.num_active_users * cfg['log_interval']) + 1) == 0 or True:
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

    def test_model_for_user(self, m, ids):
        cfg = self.cfg
        metric = Metric()
        [dataset, model] = ray.get(ids)
        dataset = StackOverflowClientDataset(dataset, self.cfg['seq_length'], self.cfg['batch_size']['test'])
        model = model.to('cuda')
        results = []
        for _, data_input in enumerate(dataset):
            input_size = data_input['label'].size(0)
            data_input = to_device(data_input, 'cuda')
            output = model(data_input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            s = output['score'].shape
            output['score'] = output['score'].permute((0, 2, 1)).reshape((s[0] * s[2], -1))
            data_input['label'] = data_input['label'].reshape((-1,))
            evaluation = metric.evaluate(cfg['metric_name']['test']['Local'], data_input, output)
            results.append((evaluation, input_size))
        return results
