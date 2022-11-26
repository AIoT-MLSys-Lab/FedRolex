# FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction

Code for paper:
> [FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction](https://openreview.net/forum?id=OtxyysUdBE)\
> Samiul Alam, Luyang Liu, Ming Yan, and Mi Zhang.\
> _NeurIPS 2022_.

The repository is built upon [HeteroFL](https://github.com/dem123456789/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients). 

# Overview

Most cross-device federated learning (FL) studies focus on the model-homogeneous setting where the global server model and local client models are identical. However, such constraint not only excludes low-end clients who would otherwise make unique contributions to model training but also restrains clients from training large models due to on-device resource bottlenecks. In this work, we propose FedRolex, a partial training (PT)-based approach that enables model-heterogeneous FL and can train a global server model larger than the largest client model. 

![tab:feature_comp](figures/table_overview.png) A quick overview of different model heterogeneous federated learning
algorithms is shown here. `FedRolex` is among a group of model heterogeneous algorithms that
use partial training to train a server model from
a heterogeneous federation of client devices.
![fig:overview](figures/fedrolex_overview.png) `FedRolex` trains only a sub-model extracted from the 
global server
model and sends the corresponding sub-model updates back to the server
for update aggregation. Here, is an illustration over three rounds of training on two
participating clients - one large-capacity
sub-model (left) and the other, a small-capacity one (right). The server extracts sub-models of
different capacities from the global model and separately broadcasts
them to the clients that have the corresponding capabilities. The
clients train the received sub-models on their local data and transmit
their heterogeneous sub-model updates to the server. Lastly, the server
aggregates those updates.

# Usage
## Setup
```commandline
pip install -r requirements.txt
```

## Training
Train RESNET-18 model on CIFAR-10 dataset.
```commandline
python main_resnet.py --data_name CIFAR10 \
                      --model_name resnet18 \ 
                      --control_name 1_100_0.1_non-iid-2_dynamic_a1-b1-c1-d1-e1_bn_1_1 \
                      --exp_name roll_test \
                      --algo roll \
                      --g_epoch 3200 \
                      --l_epoch 1 \
                      --lr 2e-4 \
                      --schedule 1200 \
                      --seed 31 \
                      --num_experiments 3 \
                      --devices 0 1 2
```
`data_name`: CIFAR10 or CIFAR100 \
`model_name`: resnet18 or vgg
`control_name`: 1_{num users}_{num participating users}_{iid or non-iid-{num classes}}_{dynamic or fix}
_{heterogeneity distribution}_{batch norm(bn), {group norm(gn)}}_{scalar 1 or 0}_{masked cross entropy, 1 or 0} \
`exp_name`: string value \
`algo`: roll, random or static \
`g_epoch`: num global epochs \
`l_epoch`: num local epochs \
`lr`: learning rate \
`schedule`: lr schedule, space seperated list of integers less than g_epoch \
`seed`: integer number \
`num_experiments`: integer number, will run `num_experiments` trials with `seed` incrementing each time \
`devices`: Index of GPUs to use \

To train Transformer model on StackOverflow dataset, use main_transformer.py instead.
```commandline
python main_transformer.py --data_name Stackoverflow \
                           --model_name transformer \
                           --control_name 1_100_0.1_iid_dynamic_a6-b10-c11-d18-e55_bn_1_1 \
                           --exp_name roll_so_test \
                           --algo roll \
                           --g_epoch 1500 \ 
                           --l_epoch 1 \
                           --lr 2e-4 \
                           --schedule 600 1000 \
                           --seed 31 \
                           --num_experiments 3 \
                           --devices 0 1 2 3 4 5 6 7
```
To train a data and model homogeneous the command would look like this.
```commandline
python main_resnet.py --data_name CIFAR10 \
                      --model_name resnet18 \
                      --control_name 1_100_0.1_iid_dynamic_a1_bn_1_1 \ 
                      --exp_name homogeneous_largest_low_heterogeneity \
                      --algo static \
                      --g_epoch 3200 \
                      --l_epoch 1 \
                      --lr 2e-4 \
                      --schedule 800 1200 \ 
                      --seed 31 \
                      --num_experiments 3 \
                      --devices 0 1 2
```

## Citation
If you find this useful for your work, please consider citing:

```
@InProceedings{alam2022fedrolex,
  title = {FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction},
  author = {Alam, Samiul and Liu, Luyang and Yan, Ming and Zhang, Mi},
  booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
  year = {2022}
}
```


