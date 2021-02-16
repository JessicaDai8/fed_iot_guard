from typing import Dict, Tuple, List

import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data import mirai_attacks, gafgyt_attacks, split_client_data, ClientData, FederationData


def get_benign_dataset(train_data: ClientData) -> Dataset:
    data_list = [torch.tensor(device_data['benign']).float() for device_data in train_data]
    dataset = TensorDataset(torch.cat(data_list, dim=0))
    return dataset


def get_test_datasets(test_data: ClientData) -> Dict[str, Dataset]:
    data_dict = {**{'benign': []},
                 **{'mirai_' + attack: [] for attack in mirai_attacks},
                 **{'gafgyt_' + attack: [] for attack in gafgyt_attacks}}

    for device_data in test_data:
        for key, arr in device_data.items():
            data_dict[key].append(torch.tensor(arr).float())

    datasets_test = {key: TensorDataset(torch.cat(data_dict[key], dim=0)) for key in data_dict.keys() if len(data_dict[key]) > 0}
    return datasets_test


def get_train_dl(client_train_data: ClientData, train_bs: int) -> DataLoader:
    dataset_train = get_benign_dataset(client_train_data)
    train_dl = DataLoader(dataset_train, batch_size=train_bs, shuffle=True)
    return train_dl


def get_val_dl(client_val_data: ClientData, test_bs: int) -> DataLoader:
    dataset_val = get_benign_dataset(client_val_data)
    val_dl = DataLoader(dataset_val, batch_size=test_bs)
    return val_dl


def get_test_dls_dict(client_test_data: ClientData, test_bs: int) -> Dict[str, DataLoader]:
    datasets = get_test_datasets(client_test_data)
    test_dls = {key: DataLoader(dataset, batch_size=test_bs) for key, dataset in datasets.items()}
    return test_dls


def get_train_val_test_dls(train_data: FederationData, val_data: FederationData, local_test_data: FederationData, train_bs: int, test_bs: int) \
        -> Tuple[List[DataLoader], List[DataLoader], List[Dict[str, DataLoader]]]:

    clients_dl_train = [get_train_dl(client_train_data, train_bs) for client_train_data in train_data]
    clients_dl_val = [get_val_dl(client_val_data, test_bs) for client_val_data in val_data]
    clients_dls_test = [get_test_dls_dict(client_test_data, test_bs) for client_test_data in local_test_data]

    return clients_dl_train, clients_dl_val, clients_dls_test


def get_client_unsupervised_initial_splitting(client_data: ClientData, p_test: float, p_unused: float) -> Tuple[ClientData, ClientData]:
    # Separate the data of the clients between benign and attack
    client_benign_data = [{'benign': device_data['benign']} for device_data in client_data]
    client_attack_data = [{key: device_data[key] for key in device_data.keys() if key != 'benign'} for device_data in client_data]
    client_train_val, client_benign_test = split_client_data(client_benign_data, p_test=p_test, p_unused=p_unused)
    client_test = [{**device_benign_test, **device_attack_data}
                   for device_benign_test, device_attack_data in zip(client_benign_test, client_attack_data)]

    return client_train_val, client_test