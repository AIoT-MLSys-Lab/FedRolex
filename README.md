# FedRolex
This is the official implementation of [*"FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model 
Extraction"](https://openreview.net/forum?id=OtxyysUdBE).

# Overview 
![tab:feature_comp](figures/table_overview.png) Overview of different model heterogeneous federated learning algorithms

![fig:overview](figures/fedrolex_overview.png) `FedRolex` trains only a sub-model extracted from the global server
model and sends the corresponding sub-model updates back to the server
for update aggregation. Here, is an illustration over three rounds of training on two
participating clients - one large-capacity
sub-model (left) and the other, a small-capacity one (right). The server extracts sub-models of
different capacities from the global model and separately broadcasts
them to the clients that have the corresponding capabilities. The
clients train the received sub-models on their local data and transmit
their heterogeneous sub-model updates to the server. Lastly, the server
aggregates those updates.


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
