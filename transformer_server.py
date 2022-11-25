import copy
from collections import OrderedDict

import numpy as np
import ray
import torch


class TransformerServerRoll:
    def __init__(self, global_model, rate, dataset_ref, cfg_id):
        self.tau = 1e-2
        self.v_t = None
        self.beta_1 = 0.9
        self.beta_2 = 0.99
        self.eta = 1e-2
        self.m_t = None
        self.user_idx = None
        self.param_idx = None
        self.dataset_ref = dataset_ref
        self.cfg_id = cfg_id
        self.cfg = ray.get(cfg_id)
        self.global_model = global_model
        self.global_parameters = global_model.cpu().state_dict()
        self.rate = rate
        self.label_split = dataset_ref
        self.make_model_rate()
        self.num_model_partitions = 50
        self.model_idxs = {}
        self.rounds = 0
        self.tmp_counts = {}
        print("server initialized")

    def step(self, local_parameters):
        self.combine(local_parameters, self.param_idx, self.user_idx)
        self.rounds += 1

    def broadcast(self, local, lr):
        cfg = self.cfg
        self.global_model.train(True)
        num_active_users = int(np.ceil(cfg['frac'] * cfg['num_users']))
        self.user_idx = copy.deepcopy(torch.arange(cfg['num_users'])
                                      [torch.randperm(cfg['num_users'])
            [:num_active_users]].tolist())
        local_parameters, self.param_idx = self.distribute(self.user_idx)

        param_ids = [ray.put(local_parameter) for local_parameter in local_parameters]
        ray.get([client.update(self.user_idx[m],
                               self.dataset_ref,
                               {'lr': lr,
                                'model_rate': self.model_rate[self.user_idx[m]],
                                'local_params': param_ids[m]})
                 for m, client in enumerate(local)])
        return local

    def make_model_rate(self):
        cfg = self.cfg
        if cfg['model_split_mode'] == 'dynamic':
            rate_idx = torch.multinomial(torch.tensor(cfg['proportion']), num_samples=cfg['num_users'],
                                         replacement=True).tolist()
            self.model_rate = np.array(self.rate)[rate_idx]
        elif cfg['model_split_mode'] == 'fix':
            self.model_rate = np.array(self.rate)
        else:
            raise ValueError('Not valid model split mode')

    def split_model(self, user_idx):
        cfg = self.cfg
        idx_i = [None for _ in range(len(user_idx))]
        idx = [OrderedDict() for _ in range(len(user_idx))]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'embedding' in k.split('.')[-2]:
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                                local_input_size = int(np.ceil(input_size * scaler_rate))
                                roll = self.rounds % input_size
                                ridx = torch.arange(input_size, device=v.device)
                                ridx = torch.roll(ridx, roll, -1)
                                input_idx_i_m = ridx[:local_input_size]
                                idx_i[m] = input_idx_i_m
                            elif 'decoder' in k and 'linear2' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                local_output_size = int(np.ceil(output_size // cfg['transformer']['num_heads']
                                                                * scaler_rate))
                                roll = self.rounds % output_size
                                ridx = torch.arange(output_size, device=v.device)
                                ridx = torch.roll(ridx, roll, -1)
                                output_idx_i_m = (ridx.reshape(
                                    cfg['transformer']['num_heads'], -1))[:, :local_output_size].reshape(-1)
                                idx_i[m] = output_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                roll = self.rounds % output_size
                                ridx = torch.arange(output_size, device=v.device)
                                ridx = torch.roll(ridx, roll, -1)
                                output_idx_i_m = ridx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'decoder' in k and 'linear2' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                            if 'linear_v' not in k:
                                idx_i[m] = idx[m][k.replace('bias', 'weight')][1]
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                else:
                    pass
        return idx

    def distribute(self, user_idx):
        self.make_model_rate()
        param_idx = self.split_model(user_idx)
        local_parameters = [OrderedDict() for _ in range(len(user_idx))]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            local_parameters[m][k] = copy.deepcopy(v[torch.meshgrid(param_idx[m][k])])
                        else:
                            local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                    else:
                        local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                else:
                    local_parameters[m][k] = copy.deepcopy(v)
        return local_parameters, param_idx

    def combine(self, local_parameters, param_idx, user_idx):
        count = OrderedDict()
        self.global_parameters = self.global_model.state_dict()
        updated_parameters = copy.deepcopy(self.global_parameters)
        tmp_counts_cpy = copy.deepcopy(self.tmp_counts)
        for k, v in updated_parameters.items():
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
            parameter_type = k.split('.')[-1]
            count[k] = v.new_zeros(v.size(), dtype=torch.float32)
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
            for m in range(len(local_parameters)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            if k.split('.')[-2] == 'embedding':
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = list(param_idx[m][k])
                                param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                count[k][torch.meshgrid(param_idx[m][k])] += 1
                            elif 'decoder' in k and 'linear2' in k:
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = list(param_idx[m][k])
                                param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                count[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                count[k][torch.meshgrid(param_idx[m][k])] += 1
                        else:
                            tmp_v[param_idx[m][k]] += local_parameters[m][k]
                            count[k][param_idx[m][k]] += 1
                    else:
                        if 'decoder' in k and 'linear2' in k:
                            label_split = self.label_split[user_idx[m]]
                            param_idx[m][k] = param_idx[m][k][label_split]
                            tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                            count[k][param_idx[m][k]] += 1
                        else:
                            tmp_v[param_idx[m][k]] += local_parameters[m][k]
                            count[k][param_idx[m][k]] += 1
                else:
                    tmp_v += local_parameters[m][k]
                    count[k] += 1
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
            self.tmp_counts = tmp_counts_cpy
        # delta_t = {k: v - self.global_parameters[k] for k, v in updated_parameters.items()}
        # if self.rounds in self.cfg['milestones']:
        #     self.eta *= 0.5
        # if not self.m_t or self.rounds in self.cfg['milestones']:
        #     self.m_t = {k: torch.zeros_like(x) for k, x in delta_t.items()}
        # self.m_t = {
        #     k: self.beta_1 * self.m_t[k] + (1 - self.beta_1) * delta_t[k] for k in delta_t.keys()
        # }
        # if not self.v_t or self.rounds in self.cfg['milestones']:
        #     self.v_t = {k: torch.zeros_like(x) for k, x in delta_t.items()}
        # self.v_t = {
        #     k: self.beta_2 * self.v_t[k] + (1 - self.beta_2) * torch.multiply(delta_t[k], delta_t[k])
        #     for k in delta_t.keys()
        # }
        # self.global_parameters = {
        #     k: self.global_parameters[k] + self.eta * self.m_t[k] / (torch.sqrt(self.v_t[k]) + self.tau)
        #     for k in self.global_parameters.keys()
        # }
        #
        # # self.global_parameters = updated_parameters
        self.global_model.load_state_dict(self.global_parameters)
        return


class TransformerServerRollSO(TransformerServerRoll):
    def broadcast(self, local, lr):
        cfg = self.cfg
        self.global_model.train(True)
        num_active_users = cfg['active_user']
        self.user_idx = copy.deepcopy(torch.arange(cfg['num_users'])
                                      [torch.randperm(cfg['num_users'])
            [:num_active_users]].tolist())
        local_parameters, self.param_idx = self.distribute(self.user_idx)

        # param_ids = [ray.put(local_parameter) for local_parameter in local_parameters]
        param_ids = local_parameters
        configs = [[self.user_idx[m],
                    self.dataset_ref[self.user_idx[m]],
                    {'lr': lr,
                     'model_rate': self.model_rate[self.user_idx[m]],
                     'local_params': param_ids[m]}] for m, _ in enumerate(param_ids)]
        return local, configs


class TransformerServerRandomSO(TransformerServerRollSO):
    def split_model(self, user_idx):
        cfg = self.cfg
        idx_i = [None for _ in range(len(user_idx))]
        idx = [OrderedDict() for _ in range(len(user_idx))]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'embedding' in k.split('.')[-2]:
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                                local_input_size = int(np.ceil(input_size * scaler_rate))
                                ridx = torch.randperm(input_size, device=v.device)
                                input_idx_i_m = ridx[:local_input_size]
                                idx_i[m] = input_idx_i_m
                            elif 'decoder' in k and 'linear2' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                local_output_size = int(np.ceil(output_size // cfg['transformer']['num_heads']
                                                                * scaler_rate))
                                ridx = torch.randperm(output_size, device=v.device)
                                output_idx_i_m = (ridx.reshape(
                                    cfg['transformer']['num_heads'], -1))[:, :local_output_size].reshape(-1)
                                idx_i[m] = output_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                roll = self.rounds % output_size
                                ridx = torch.randperm(output_size, device=v.device)
                                output_idx_i_m = ridx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'decoder' in k and 'linear2' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                            if 'linear_v' not in k:
                                idx_i[m] = idx[m][k.replace('bias', 'weight')][1]
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                else:
                    pass
        return idx


class TransformerServerStaticSO(TransformerServerRollSO):
    def split_model(self, user_idx):
        cfg = self.cfg
        idx_i = [None for _ in range(len(user_idx))]
        idx = [OrderedDict() for _ in range(len(user_idx))]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'embedding' in k.split('.')[-2]:
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                                local_input_size = int(np.ceil(input_size * scaler_rate))
                                ridx = torch.arange(input_size, device=v.device)
                                input_idx_i_m = ridx[:local_input_size]
                                idx_i[m] = input_idx_i_m
                            elif 'decoder' in k and 'linear2' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                local_output_size = int(np.ceil(output_size // cfg['transformer']['num_heads']
                                                                * scaler_rate))
                                ridx = torch.arange(output_size, device=v.device)
                                output_idx_i_m = (ridx.reshape(
                                    cfg['transformer']['num_heads'], -1))[:, :local_output_size].reshape(-1)
                                idx_i[m] = output_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                ridx = torch.arange(output_size, device=v.device)
                                output_idx_i_m = ridx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'decoder' in k and 'linear2' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                            if 'linear_v' not in k:
                                idx_i[m] = idx[m][k.replace('bias', 'weight')][1]
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                else:
                    pass
        return idx
