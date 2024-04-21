import os, sys
sys.path.append(os.getcwd())

from copy import deepcopy
from types import SimpleNamespace
from typing import List, Set, Tuple, Callable, Optional

import matplotlib.pyplot as plt
import sklearn.datasets
import tensorflow as tf
import tflib as lib
import tflib.plot
import tflib.mnist_fed
import tflib.sn as sn
import time
import numpy as np
import torch
from context_printer import ContextPrinter as Ctp, Color
# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from architectures import NormalizingModel
from ml import set_models_sub_divs


def federated_averaging(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    with torch.no_grad():
        state_dict_mean = global_model.state_dict()
        for key in state_dict_mean:
            state_dict_mean[key] = torch.stack([model.state_dict()[key] for model in models], dim=-1).mean(dim=-1)
        global_model.load_state_dict(state_dict_mean)


# For 8 clients this is equivalent to federated trimmed mean 3
def federated_median(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    n_excluded_down = (len(models) - 1) // 2
    n_included = 2 if (len(models) % 2 == 0) else 1

    with torch.no_grad():
        state_dict_median = global_model.state_dict()
        for key in state_dict_median:
            # It seems that it's much faster to compute the median by manually sorting and narrowing onto the right element
            # rather than using torch.median.
            sorted_tensor, _ = torch.sort(torch.stack([model.state_dict()[key] for model in models], dim=-1), dim=-1)
            trimmed_tensor = torch.narrow(sorted_tensor, -1, n_excluded_down, n_included)
            state_dict_median[key] = trimmed_tensor.mean(dim=-1)
        global_model.load_state_dict(state_dict_median)


def federated_min_max(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    subs = torch.stack([model.sub for model in models])
    sub, _ = torch.min(subs, dim=0)
    divs = torch.stack([model.div for model in models])
    max_values = divs + subs
    max_value, _ = torch.max(max_values, dim=0)
    div = max_value - sub
    global_model.set_sub_div(sub, div)


# Shortcut for __federated_trimmed_mean(global_model, models, 1) so that it's easier to set the aggregation function as a single param
def federated_trimmed_mean_1(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    __federated_trimmed_mean(global_model, models, 1)


# Shortcut for __federated_trimmed_mean(global_model, models, 2) so that it's easier to set the aggregation function as a single param
def federated_trimmed_mean_2(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    __federated_trimmed_mean(global_model, models, 2)


def __federated_trimmed_mean(global_model: torch.nn.Module, models: List[torch.nn.Module], trim_num_up: int) -> None:
    n = len(models)
    n_remaining = n - 2 * trim_num_up

    with torch.no_grad():
        state_dict_result = global_model.state_dict()
        for key in state_dict_result:
            sorted_tensor, _ = torch.sort(torch.stack([model.state_dict()[key] for model in models], dim=-1), dim=-1)
            trimmed_tensor = torch.narrow(sorted_tensor, -1, trim_num_up, n_remaining)
            state_dict_result[key] = trimmed_tensor.mean(dim=-1)
        global_model.load_state_dict(state_dict_result)


# As defined in https://arxiv.org/pdf/2006.09365.pdf
def s_resampling(models: List[torch.nn.Module], s: int) -> Tuple[List[torch.nn.Module], List[List[int]]]:
    T = len(models)
    c = [0 for _ in range(T)]
    output_models = []
    output_indexes = []
    for t in range(T):
        j = [-1 for _ in range(s)]
        for i in range(s):
            while True:
                j[i] = np.random.randint(T)
                if c[j[i]] < s:
                    c[j[i]] += 1
                    break
        output_indexes.append(j)
        with torch.no_grad():
            g_t_bar = deepcopy(models[0])
            sampled_models = [models[j[i]] for i in range(s)]
            federated_averaging(g_t_bar, sampled_models)
            output_models.append(g_t_bar)

    return output_models, output_indexes


def model_update_scaling(global_model: torch.nn.Module, malicious_clients_models: List[torch.nn.Module], factor: float) -> None:
    with torch.no_grad():
        for model in malicious_clients_models:
            new_state_dict = {}
            for key, original_param in global_model.state_dict().items():
                param_delta = model.state_dict()[key] - original_param
                param_delta = param_delta * factor
                new_state_dict.update({key: original_param + param_delta})
            model.load_state_dict(new_state_dict)


def model_canceling_attack(global_model: torch.nn.Module, malicious_clients_models: List[torch.nn.Module], n_honest: int) -> None:
    factor = - n_honest / len(malicious_clients_models)
    with torch.no_grad():
        for normalizing_model in malicious_clients_models:
            new_state_dict = {}
            for key, original_param in global_model.model.state_dict().items():
                new_state_dict.update({key: original_param * factor})
            normalizing_model.model.load_state_dict(new_state_dict)
            # We only change the internal model of the NormalizingModel. That way we do not actually attack the normalization values
            # because they are not supposed to change throughout the training anyway.


def select_mimicked_client(params: SimpleNamespace) -> Optional[int]:
    honest_client_ids = [client_id for client_id in range(len(params.clients_devices)) if client_id not in params.malicious_clients]
    if params.model_poisoning == 'mimic_attack':
        mimicked_client_id = np.random.choice(honest_client_ids)
        Ctp.print('The mimicked client is {}'.format(mimicked_client_id))
    else:
        mimicked_client_id = None
    return mimicked_client_id


# Attack in which all malicious clients mimic the model of a single good client. The mimicked client should always be the same throughout
# the federation rounds.
def mimic_attack(models: List[torch.nn.Module], malicious_clients: Set[int], mimicked_client: int) -> None:
    with torch.no_grad():
        for i in malicious_clients:
            models[i].load_state_dict(models[mimicked_client].state_dict())


def init_federated_models(train_dls: List[DataLoader], params: SimpleNamespace, architecture: Callable):
    # Initialization of a global model
    n_clients = len(params.clients_devices)
    global_model = NormalizingModel(architecture(activation_function=params.activation_fn, hidden_layers=params.hidden_layers),
                                    sub=torch.zeros(params.n_features), div=torch.ones(params.n_features))

    if params.cuda:
        global_model = global_model.cuda()

    models = [deepcopy(global_model) for _ in range(n_clients)]
    set_models_sub_divs(params.normalization, models, train_dls, color=Color.RED)

    if params.normalization == 'min-max':
        federated_min_max(global_model, models)
    else:
        federated_averaging(global_model, models)

    models = [deepcopy(global_model) for _ in range(n_clients)]
    return global_model, models


def model_poisoning(global_model: torch.nn.Module, models: List[torch.nn.Module], params: SimpleNamespace,
                    mimicked_client_id: Optional[int] = None, verbose: bool = False) -> List[torch.nn.Module]:
    malicious_clients_models = [model for client_id, model in enumerate(models) if client_id in params.malicious_clients]
    n_honest = len(models) - len(malicious_clients_models)

    # Model poisoning attacks
    if params.model_poisoning is not None:
        if params.model_poisoning == 'cancel_attack':
            model_canceling_attack(global_model=global_model, malicious_clients_models=malicious_clients_models, n_honest=n_honest)
            if verbose:
                Ctp.print('Performing cancel attack')
        elif params.model_poisoning == 'mimic_attack':
            mimic_attack(models, params.malicious_clients, mimicked_client_id)
            if verbose:
                Ctp.print('Performing mimic attack on client {}'.format(mimicked_client_id))
        else:
            raise ValueError('Wrong value for model_poisoning: ' + str(params.model_poisoning))

    # Rescale the model updates of the malicious clients (if any)
    model_update_scaling(global_model=global_model, malicious_clients_models=malicious_clients_models, factor=params.model_update_factor)
    return models


    neural_net = "AlexNet" #Options: AlexNet,Inception,ResNet
    BATCH_SIZE = 50 # Batch size
    TEST_BATCH_SIZE = 1000
    ITERS = 10000 # How many generator iterations to train for 
    INPUT_DIM = 784 # Number of pixels in MNIST (28*28)
    nodes = 100
    maximize_iters = 1
    test_iters = 100
    noise_std = 0.01
    
    address = 'mnist_'+neural_net
    
    if not os.path.exists(address):
        os.makedirs(address)
                        
    #incept functions code (adapted from https://github.com/farzanfarnia/RobustFL/tree/main)
    def incept(input_x, input_filters, ch1_filters, ch3_filters, spectral_norm=True, tighter_sn=True,
               scope_name='incept', update_collection=None, beta=1., bn=True, reuse=None, training=False):
        """Inception module"""
            
        with tf.variable_scope(scope_name, reuse=reuse):
            ch1_output = tf.nn.relu(sn.conv2d(input_x, [1, 1, input_filters, ch1_filters],
                                              scope_name='conv_ch1', spectral_norm=spectral_norm,
                                              xavier=True, bn=bn, beta=beta, tighter_sn=tighter_sn,
                                              update_collection=update_collection, reuse=reuse, training=training))
            ch3_output = tf.nn.relu(sn.conv2d(input_x, [3, 3, input_filters, ch3_filters],
                                              scope_name='conv_ch3', spectral_norm=spectral_norm,
                                              xavier=True, bn=bn, beta=beta, tighter_sn=tighter_sn,
                                              update_collection=update_collection, reuse=reuse, training=training))
            return tf.concat([ch1_output, ch3_output], axis=-1)
    
    
    def downsample(input_x, input_filters, ch3_filters, spectral_norm=True, tighter_sn=True,
                   scope_name='downsamp', update_collection=None, beta=1., bn=True, reuse=None, training=False):
        """Downsample module"""
            
        with tf.variable_scope(scope_name, reuse=reuse):
            ch3_output = tf.nn.relu(sn.conv2d(input_x, [3, 3, input_filters, ch3_filters], tighter_sn=tighter_sn,
                                              scope_name='conv_ch3', spectral_norm=spectral_norm,
                                              xavier=True, bn=bn, stride=2, beta=beta, reuse=reuse,
                                              update_collection=update_collection, training=training))
            pool_output = tf.nn.max_pool(input_x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                         padding='SAME', name='pool')
            return tf.concat([ch3_output, pool_output], axis=-1)
    
        
    def inception(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None, training=False):
        """Mini-inception architecture (note that we do batch norm in absence of spectral norm)"""
        
        snconv_kwargs = dict(spectral_norm=False, reuse=reuse, training=training, bn=True)
        input_data_reshaped = tf.reshape(input_data,[-1,28,28,1])
        layer1 = tf.nn.relu(sn.conv2d(input_data_reshaped, [3, 3, 1, 96], scope_name='conv1', **snconv_kwargs))
        layer2 = incept(layer1, 96, 32, 32, scope_name='incept2', **snconv_kwargs)
        layer3 = incept(layer2, 32+32, 32, 48, scope_name='incept3', **snconv_kwargs)
        layer4 = downsample(layer3, 32+48, 80, scope_name='downsamp4', **snconv_kwargs)
        layer5 = incept(layer4, 80+32+48, 112, 48, scope_name='incept5', **snconv_kwargs)
        layer6 = incept(layer5, 112+48, 96, 64, scope_name='incept6', **snconv_kwargs)
        layer7 = incept(layer6, 96+64, 80, 80, scope_name='incept7', **snconv_kwargs)
        layer8 = incept(layer7, 80+80, 48, 96, scope_name='incept8', **snconv_kwargs)
        layer9 = downsample(layer8, 48+96, 96, scope_name='downsamp9', **snconv_kwargs)
        layer10 = incept(layer9, 96+48+96, 176, 160, scope_name='incept10', **snconv_kwargs)
        layer11 = incept(layer10, 176+160, 176, 160, scope_name='incept11', **snconv_kwargs)
        layer12 = tf.nn.pool(layer11, window_shape=[7, 7], pooling_type='AVG', 
                             padding='SAME', strides=[1, 1], name='mean_pool12')
        
        fc = sn.linear(layer12, num_classes, scope_name='fc', spectral_norm=False, xavier=True, reuse=reuse)
            
        return fc
    
    def alexnet(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None, training=False):
        """AlexNet architecture
            two [convolution 5x5 -> max-pool 3x3 -> local-response-normalization] modules 
            followed by two fully connected layers with 384 and 192 hidden units, respectively. 
            Finally a NUM_CLASSES-way linear layer is used for prediction
        """
        input_data_reshaped = tf.reshape(input_data,[-1,28,28,1])
        conv = sn.conv2d(input_data_reshaped, [5, 5, 1, 96], scope_name='conv1', spectral_norm=False, reuse=reuse)
        conv1 = tf.nn.relu(conv, name='conv1_relu')
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        #norm1= pool1
        
        conv = sn.conv2d(norm1, [5, 5, 96, 256], scope_name='conv2', spectral_norm=False, reuse=reuse)
        conv2 = tf.nn.relu(conv, name='conv2_relu')
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool2')
        norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        #norm2 = pool2
        
        reshape = tf.reshape(norm2, [-1, 6*6*256])
        lin = sn.linear(reshape, 384, scope_name='linear1', spectral_norm=False, reuse=reuse)
        lin1 = tf.nn.relu(lin, name='linear1_relu')
    
        lin = sn.linear(lin1, 192, scope_name='linear2', spectral_norm=False, reuse=reuse)
        lin2 = tf.nn.relu(lin, name='linear2_relu')
    
        fc = sn.linear(lin2, num_classes, scope_name='fc', spectral_norm=False, reuse=reuse)
            
        return fc
                            
        stepsize_adv_delta = 0.02
        stepsize_adv_gamma = 0.001
        adv_lambda = 0.1
        LAMBDA_0 = 4.0
        LAMBDA_1 = 100.0
        norm_cons = 2.5
        norm_train = 2.5
        adv_stepsize = 0.05
        norm_max_0 = 1.0
        norm_max_1 = 5.0
        
        real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, INPUT_DIM])
        label = tf.placeholder(tf.int64, shape=[BATCH_SIZE])
        delta = tf.placeholder(tf.float32, shape=[1, INPUT_DIM])
        Gamma = tf.placeholder(tf.float32, shape=[INPUT_DIM, INPUT_DIM])
        data_perturbed = tf.matmul(real_data,Gamma) + delta
        
        if neural_net=="AlexNet":
            NN_out_perturbed = alexnet(data_perturbed )
        elif neural_net=="Inception":
            NN_out_perturbed = inception(data_perturbed )
            
        train_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(NN_out_perturbed,axis=1),label),dtype=tf.float32))
        
        train_loss= tf.reduce_mean( tf.log(tf.reduce_sum(tf.exp(NN_out_perturbed),reduction_indices=[1]))
                                   - tf.diag_part(tf.gather(NN_out_perturbed,label,axis=1)))
        attack_power = tf.reduce_sum(delta**2)
        Gamma_power = tf.reduce_sum( (tf.eye(num_rows=INPUT_DIM,dtype=tf.float32)-Gamma)**2 )
        train_loss = BATCH_SIZE*train_loss - LAMBDA_0 * attack_power - LAMBDA_1 * Gamma_power
        
        gradients_delta = tf.gradients(train_loss,delta)[0]
        delta_update = delta + stepsize_adv_delta*gradients_delta 
        gradients_gamma = tf.gradients(train_loss,Gamma)[0]
        Gamma_update = Gamma + stepsize_adv_gamma*gradients_gamma
        
        
        real_data_agg = tf.placeholder(tf.float32, shape=[BATCH_SIZE*nodes, INPUT_DIM])
        label_agg = tf.placeholder(tf.int64, shape=[BATCH_SIZE*nodes])
        
        if neural_net=="AlexNet":
            NN_out_agg = alexnet(real_data_agg,reuse=True)
        elif neural_net=="Inception":
            NN_out_agg = inception(real_data_agg,reuse=True)
            
        
        train_loss_agg= tf.reduce_mean( tf.log(tf.reduce_sum(tf.exp(NN_out_agg),reduction_indices=[1]))
                                   - tf.diag_part(tf.gather(NN_out_agg,label_agg,axis=1)))
        train_acc_agg = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(NN_out_agg,axis=1),label_agg),dtype=tf.float32))
        
        
        valid_data = tf.placeholder(tf.float32, shape=[TEST_BATCH_SIZE, INPUT_DIM])
        delta_valid = tf.placeholder(tf.float32, shape=[1, INPUT_DIM])
        Gamma_valid = tf.placeholder(tf.float32, shape=[INPUT_DIM, INPUT_DIM])
        valid_label = tf.placeholder(tf.int64, shape=[TEST_BATCH_SIZE])
        
        if neural_net=="AlexNet":
            valid_NN_out = alexnet(tf.matmul(valid_data,Gamma_valid) + delta_valid,reuse=True)
        elif neural_net=="Inception":
            valid_NN_out = inception(tf.matmul(valid_data,Gamma_valid) + delta_valid,reuse=True)
        
        valid_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(valid_NN_out,axis=1),valid_label),dtype=tf.float32))
        valid_loss= tf.reduce_mean( tf.log(tf.reduce_sum(tf.exp(valid_NN_out),reduction_indices=[1]))
                                  - tf.diag_part(tf.gather(valid_NN_out,valid_label,axis=1))  )
                     
        delta_valid_update = delta_valid + adv_stepsize * tf.gradients(valid_loss,delta_valid)[0]
        delta_valid_update = delta_valid_update / tf.transpose([tf.maximum(tf.norm(delta_valid_update)/norm_max_0,1.)]) 
        Gamma_valid_update = Gamma_valid + adv_stepsize * tf.gradients(valid_loss,Gamma_valid)[0]
        Gamma_valid_update = tf.eye(num_rows=INPUT_DIM,dtype=tf.float32) + ((Gamma_valid_update-tf.eye(num_rows=INPUT_DIM,dtype=tf.float32))
                               / tf.transpose([tf.maximum(tf.norm(Gamma_valid_update-tf.eye(num_rows=INPUT_DIM,dtype=tf.float32))/norm_max_1,1.)])  )
        
        
        saver = tf.train.Saver(max_to_keep=10)
        nn_params = tf.trainable_variables()
        Classifier_train_op = tf.train.GradientDescentOptimizer(
                learning_rate=1e-3
            ).minimize(train_loss_agg, var_list=nn_params)


#robustFL helper training function code. Adapted from https://github.com/farzanfarnia/RobustFL
train_gen, dev_gen, test_gen = lib.mnist_fed.load(BATCH_SIZE, TEST_BATCH_SIZE, k= nodes)
def inf_train_gen():
    while True:
        for elements in train_gen():
            for (images,targets) in elements:
                yield images,targets
            
def inf_test_gen():
    while True:
        for elements in test_gen():
            for (images,targets) in elements:
                yield images,targets
            


# Aggregates the model according to params.aggregation_function, potentially using s-resampling, and distributes the global model back to the clients
def model_aggregation(global_model: torch.nn.Module, models: List[torch.nn.Module], params: SimpleNamespace, verbose: bool = False)\
        -> Tuple[torch.nn.Module, List[torch.nn.Module]]:

    if params.resampling is not None:
        models, indexes = s_resampling(models, params.resampling)
        if verbose:
            Ctp.print(indexes)
            
    # insert robustFL code. Adapted from https://github.com/farzanfarnia/RobustFL

    train_loss_arr = []
    train_acc_arr= []
    train_loss_perturbed_arr = []
    train_acc_perturbed_arr= []
    valid_acc_arr = []
    valid_acc_perturbed_arr = []

    np.random.seed(1)
    perturbation_add_train = noise_std*np.random.normal(size=[nodes,INPUT_DIM])
    matrix_mult_train = (noise_std/np.sqrt(INPUT_DIM))*np.random.normal(size=[nodes,INPUT_DIM,INPUT_DIM])

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.45))) as session:


        session.run(tf.initialize_all_variables())
        gen = inf_train_gen()
        gen_test = inf_test_gen()
        delta_np=np.array([np.random.normal(size=[1,INPUT_DIM ])/(1e12)]*nodes)
        gamma_np=np.array([np.eye(INPUT_DIM,dtype=np.float32)]*nodes)
        _data_agg = np.zeros([BATCH_SIZE*nodes,INPUT_DIM],dtype=np.float32)
        _data_perturbed_agg = np.zeros([BATCH_SIZE*nodes,INPUT_DIM],dtype=np.float32)
        _labels_agg = np.zeros([BATCH_SIZE*nodes],dtype=np.int64)
        for iteration in range(ITERS):
            
            start_time = time.time()
            
            for k in range(nodes):    
                
                data_inf = next(gen)
                _data = data_inf[0]
                _data_agg[k*BATCH_SIZE:(k+1)*BATCH_SIZE,:] = _data
                _data_agg[k*BATCH_SIZE:(k+1)*BATCH_SIZE,:] += np.matmul(_data_agg[k*BATCH_SIZE:(k+1)*BATCH_SIZE,:],
                                                                       np.squeeze(matrix_mult_train[k,:,:]))
                _data_agg[k*BATCH_SIZE:(k+1)*BATCH_SIZE,:] +=  perturbation_add_train[k,:]
                _labels = data_inf[1]
                _labels_agg[k*BATCH_SIZE:(k+1)*BATCH_SIZE] = _labels
                
                for _ in range(maximize_iters):
                    delta_np[k],gamma_np[k],_data_perturbed = session.run([delta_update,Gamma_update,data_perturbed],
                                                                          feed_dict={real_data: _data,label: _labels
                                                                  ,delta: delta_np[k], Gamma:gamma_np[k]})
                    
                _data_perturbed_agg[k*BATCH_SIZE:(k+1)*BATCH_SIZE,:] = _data_perturbed
                    #_data_agg[k*BATCH_SIZE:(k+1)*BATCH_SIZE,:]
                
            _,_train_loss_perturbed,_train_acc_perturbed = session.run([Classifier_train_op,train_loss_agg,train_acc_agg],
                                                                  feed_dict={real_data_agg: _data_perturbed_agg,
                                                                             label_agg: _labels_agg})                             
            _train_loss,_train_acc = session.run([train_loss_agg,train_acc_agg],
                                                 feed_dict={real_data_agg: _data_agg,
                                                          label_agg: _labels_agg})                             
        
            lib.plot.plot(address+'/train_loss_perturbed', _train_loss_perturbed)
            lib.plot.plot(address+'/train_acc_perturbed', _train_acc_perturbed)
            lib.plot.plot(address+'/train_loss', _train_loss)
            lib.plot.plot(address+'/train_acc', _train_acc)
            train_loss_perturbed_arr.append(_train_loss_perturbed)
            train_acc_perturbed_arr.append(_train_acc_perturbed)
            train_loss_arr.append(_train_loss)
            train_acc_arr.append(_train_acc)
    
            
            
            # Write logs every 500 iters
            
            if iteration % 1000 == 0:
                test_data_inf = next(gen_test)
                _data_valid = test_data_inf[0]
                _labels_valid = test_data_inf[1]      
    
                delta_valid_np=np.random.normal(size=[1,INPUT_DIM ])/(1e12)
                gamma_valid_np=np.eye(INPUT_DIM,dtype=np.float32)
                
                _valid_acc  = session.run(valid_acc, feed_dict={valid_data: _data_valid,valid_label:_labels_valid,
                                                               Gamma_valid: gamma_valid_np, delta_valid: delta_valid_np})      
                for _ in range(test_iters):
                     gamma_valid_np,delta_valid_np= session.run([Gamma_valid_update,delta_valid_update],
                                                                feed_dict={valid_data: _data_valid,valid_label:_labels_valid,
                                                               Gamma_valid: gamma_valid_np, delta_valid: delta_valid_np}) 
                
                _valid_acc_perturbed  = session.run(valid_acc, feed_dict={valid_data: _data_valid,valid_label:_labels_valid,
                                                               Gamma_valid: gamma_valid_np, delta_valid: delta_valid_np})      
    
                valid_acc_arr.append(_valid_acc)
                valid_acc_perturbed_arr.append(_valid_acc_perturbed)
                
                lib.plot.plot(address+'/valid_acc_nonadversarial', _valid_acc)
                lib.plot.plot(address+'/valid_acc_perturbed', _valid_acc_perturbed)
                
                
                np.save(address+'/train_loss_arr',train_loss_arr)
                np.save(address+'/train_acc_arr',train_acc_arr)
                np.save(address+'/train_loss_perturbed_arr',train_loss_perturbed_arr)
                np.save(address+'/train_acc_perturbed_arr',train_acc_perturbed_arr)
                np.save(address+'/valid_acc_arr',valid_acc_arr)
                np.save(address+'/valid_acc_perturbed_arr',valid_acc_perturbed_arr)
                
            if iteration % 1000 == 0 and iteration>0:
                saver.save(session, address+"/model_"+str(iteration))
                
            if iteration % 50 == 0 or iteration<10:
                lib.plot.flush()
    
            lib.plot.tick()
    
    params.aggregation_function(global_model, models)

    # Distribute the global model back to each client
    models = [deepcopy(global_model) for _ in range(len(params.clients_devices))]

    return global_model, models
