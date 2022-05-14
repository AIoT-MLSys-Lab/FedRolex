import argparse
import datetime
import os
import random
import shutil
import time

import numpy as np
import ray
import torch
import torch.backends.cudnn as cudnn

import models
from config import cfg
from data import fetch_dataset, split_dataset, BatchDataset
from logger import Logger
from metrics import Metric
from transformer_client import TransformerClient
from transformer_server import TransformerServerRoll
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler

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
# parser.add_argument('--lr', default=None, type=int)
parser.add_argument('--g_epochs', default=None, type=int)
parser.add_argument('--l_epochs', default=None, type=int)
parser.add_argument('--schedule', default=None, nargs='+', type=int)
# parser.add_argument('--exp_name', default=None, type=str)
args = vars(parser.parse_args())
cfg['init_seed'] = int(args['seed'])
if args['algo'] == 'roll':
    from transformer_server import TransformerServerRoll as Server
elif args['algo'] == 'random':
    from transformer_server import TransformerServerRandom as Server
elif args['algo'] == 'orig':
    from transformer_server import TransformerServerOrig as Server

args = vars(parser.parse_args())
cfg['init_seed'] = int(args['seed'])
if args['devices'] is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in args['devices']])

for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']])
cfg['pivot_metric'] = 'Global-Perplexity'
cfg['pivot'] = float('inf')
cfg['metric_name'] = {'train': {'Local': ['Local-Loss', 'Local-Perplexity']},
                      'test': {'Global': ['Global-Loss', 'Global-Perplexity']}}
ray.init()


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
    global_model = models.transformer(model_rate=cfg["global_model_rate"], cfg=cfg).to(cfg['device'])
    optimizer = make_optimizer(global_model, cfg['lr'])
    scheduler = make_scheduler(optimizer)
    last_epoch = 1
    data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])

    logger_path = os.path.join('output', 'runs', 'train_{}'.format(f'{cfg["model_tag"]}_{cfg["exp_name"]}'))
    logger = Logger(logger_path)

    cfg_id = ray.put(cfg)
    num_active_users = int(np.ceil(cfg['frac'] * cfg['num_users']))
    dataset_ref = {
        'dataset': ray.put(dataset['train']),
        'split': ray.put(data_split['train']),
        'label_split': ray.put(label_split)}

    server = Server(global_model, cfg['model_rate'], dataset_ref, cfg_id)
    local = [TransformerClient.remote(logger.log_path, [cfg_id]) for _ in range(num_active_users)]
    # local = [TransformerClient(logger.log_path, [cfg_id]) for _ in range(num_active_users)]

    for epoch in range(last_epoch, cfg['num_epochs']['global'] + 1):
        t0 = time.time()
        logger.safe(True)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        local = server.broadcast(local, lr)
        t1 = time.time()

        num_active_users = len(local)
        start_time = time.time()
        dt = ray.get([client.step.remote(m, num_active_users, start_time)
                      for m, client in enumerate(local)])

        local_parameters = [v for _k, v in enumerate(dt)]

        # local_parameters = [None for _ in range(num_active_users)]
        # for m in range(num_active_users):
        #     local[m].step(m, num_active_users, start_time)
        #     local_parameters[m] = local[m].pull()
        t2 = time.time()
        server.step(local_parameters)
        t3 = time.time()

        global_model = server.global_model
        test_model = global_model
        t4 = time.time()

        test(dataset['test'], test_model, logger, epoch)
        t5 = time.time()
        logger.safe(False)
        model_state_dict = global_model.state_dict()
        save_result = {
            'cfg': cfg, 'epoch': epoch + 1, 'data_split': data_split, 'label_split': label_split,
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
    logger.safe(False)
    [ray.kill(client) for client in local]
    return


def test(dataset, model, logger, epoch):
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        batch_dataset = BatchDataset(dataset, cfg['bptt'])
        for i, input in enumerate(batch_dataset):
            input_size = input['label'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(cfg['metric_name']['test']['Global'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test']['Global'])
    return


if __name__ == "__main__":
    main()
