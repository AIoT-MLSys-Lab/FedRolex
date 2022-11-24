import argparse
import copy
import datetime
import os
import random
import shutil
import time
from typing import List

import numpy as np
import ray
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import models
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset, GenericDataset
from logger import Logger
from metrics import Metric
from gld_client import ResnetClient
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, collate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--devices', default=None, nargs='+', type=int)
parser.add_argument('--algo', default='roll', type=str)
parser.add_argument('--weighting', default='avg', type=str)
# parser.add_argument('--lr', default=None, type=int)
parser.add_argument('--g_epochs', default=None, type=int)
parser.add_argument('--l_epochs', default=None, type=int)
parser.add_argument('--overlap', default=None, type=float)

parser.add_argument('--schedule', default=None, nargs='+', type=int)
# parser.add_argument('--exp_name', default=None, type=str)
args = vars(parser.parse_args())

cfg['overlap'] = args['overlap']
cfg['weighting'] = args['weighting']
cfg['init_seed'] = int(args['seed'])
if args['algo'] == 'roll':
    from gld_server import ResnetServerRoll as Server
elif args['algo'] == 'random':
    from gld_server import ResnetServerRandom as Server
elif args['algo'] == 'orig':
    from gld_server import ResnetServerOrig as Server
if args['devices'] is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in args['devices']])
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']])
cfg['pivot_metric'] = 'Global-Accuracy'
cfg['pivot'] = -float('inf')
cfg['metric_name'] = {'train': {'Local': ['Local-Loss', 'Local-Accuracy']},
                      'test': {'Local': ['Local-Loss', 'Local-Accuracy'], 'Global': ['Global-Loss', 'Global-Accuracy']}}

ray.init()
rates = None

def main():
    process_control()

    if args['schedule'] is not None:
        cfg['milestones'] = args['schedule']

    if args['g_epochs'] is not None and args['l_epochs'] is not None:
        cfg['num_epochs'] = {'global': args['g_epochs'], 'local': args['l_epochs']}
    cfg['init_seed'] = int(args['seed'])
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['subset'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        print('Seed: {}'.format(cfg['init_seed']))
        run_experiment()
    return


def run_experiment():
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_deterministic_debug_mode('default')
    os.environ['PYTHONHASHSEED'] = str(seed)
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
    process_dataset(dataset)
    print(cfg['model_name'])
    if 'resnet18' in cfg['model_name']:
        print("main: res18 model found")
        global_model = models.resnet18(model_rate=cfg["global_model_rate"], cfg=cfg).to(cfg['device'])
    elif 'resnet34' in cfg['model_name']:
        print("main: res34 model found")
        global_model = models.resnet34(model_rate=cfg["global_model_rate"], cfg=cfg).to(cfg['device'])
    elif 'resnet50' in cfg['model_name']:
        print("main: res50 model found")
        global_model = models.resnet50(model_rate=cfg["global_model_rate"], cfg=cfg).to(cfg['device'])
    else:
        print("main: no model found")
        global_model = models.resnet101(model_rate=cfg["global_model_rate"], cfg=cfg).to(cfg['device'])

    optimizer = make_optimizer(global_model, cfg['lr'])
    scheduler = make_scheduler(optimizer)
    last_epoch = 1
    data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])
    logger_path = os.path.join('output', 'runs', 'train_{}'.format(f'{cfg["model_tag"]}_{cfg["exp_name"]}'))
    logger = Logger(logger_path)

    cfg_id = ray.put(cfg)
    num_active_users = cfg['active_user']
    dataset_ref = {
        'split': [ray.put(split) for split in tqdm(data_split['train'])],
        'label_split': ray.put(label_split)}

    server = Server(global_model, cfg['model_rate'], dataset_ref, cfg_id)
    num_users_per_step = 8
    local = [ResnetClient.remote(logger.log_path, [cfg_id]) for _ in range(num_users_per_step)]
    for epoch in range(last_epoch, cfg['num_epochs']['global'] + 1):
        print(f"Starting {epoch}...")
        local_parameters = []
        logger.safe(True)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        t0 = time.time()
        client_configs, param_idx, user_idx = server.broadcast(lr)
        t1 = time.time()
        start_time = time.time()
        local = [ResnetClient.remote(logger.log_path, [cfg_id]) for _ in range(num_users_per_step)]
        for user_start_idx in tqdm(range(0, num_active_users, num_users_per_step)):
            idxs = list(range(user_start_idx, min(num_active_users, user_start_idx+num_users_per_step)))
            sel_cfg = [client_configs[idx] for idx in idxs]
            [client.update.remote(*config) for client, config in zip(local[0:len(sel_cfg)], sel_cfg)]
            dt = ray.get([client.step.remote(m, num_active_users, start_time)
                          for m, client in enumerate(local[0:len(sel_cfg)])])
            local_parameters += [v for _k, v in enumerate(dt)]
        torch.cuda.empty_cache()

        t2 = time.time()
        server.step(local_parameters, param_idx, user_idx)
        t3 = time.time()

        global_model = server.global_model
        test_model = global_model
        t4 = time.time()
        if epoch % 10 == 1:
            test(data_split['test'], label_split, test_model, logger, epoch, local)
        t5 = time.time()
        logger.safe(False)
        model_state_dict = global_model.state_dict()
        save_result = {
            'cfg': cfg, 'epoch': epoch + 1,
            'model_dict': model_state_dict, 'optimizer_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(), 'logger': logger}
        save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if cfg['pivot'] < logger.mean['test/{}'.format(cfg['pivot_metric'])]:
            cfg['pivot'] = logger.mean['test/{}'.format(cfg['pivot_metric'])]
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
        t6 = time.time()
        print(f'Broadcast Time      : {datetime.timedelta(seconds=t1 - t0)}')
        print(f'Client Step Time    : {datetime.timedelta(seconds=t2 - t1)}')
        print(f'Server Step Time    : {datetime.timedelta(seconds=t3 - t2)}')
        print(f'Stats Time          : {datetime.timedelta(seconds=t4 - t3)}')
        print(f'Test Time           : {datetime.timedelta(seconds=t5 - t4)}')
        print(f'Output Copy Time    : {datetime.timedelta(seconds=t6 - t5)}')
        print(f'<<Total epoch Time>>: {datetime.timedelta(seconds=t6 - t0)}')
        test_model = None
        global_model = None
        torch.cuda.empty_cache()
    [ray.kill(client) for client in local]

    logger.safe(False)
    return

def test_per_worker(model, data_split, epoch):
    with torch.no_grad():
        model.train(False)
        data_loader = make_data_loader({'test': GenericDataset(data_split)})['test']
        metric = Metric()
        model = model.cuda()
        logger_ev = []
        for i, data_input in tqdm(enumerate(data_loader)):
            data_input = collate(data_input)
            input_size = data_input['img'].size(0)
            data_input = to_device(data_input, 'cuda')
            output = model(data_input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(cfg['metric_name']['test']['Global'], data_input, output)

            logger_ev.append([evaluation, 'test', input_size])
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        return info, logger_ev

def test(data_split, label_split, model, logger, epoch, local):
    info, logger_ev = test_per_worker(model, data_split, epoch)
    [logger.append(*i) for i in logger_ev]
    logger.append(info, 'test', mean=False)
    logger.write('test', cfg['metric_name']['test']['Local'] + cfg['metric_name']['test']['Global'])
    return


if __name__ == "__main__":
    main()
