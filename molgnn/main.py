import json
import torch
import os
import numpy
import gcnetwork
import time
import random
import tensorboardX
import torch.optim as optim
import train_test_gcn as train_test_gcn
import glob
import tqdm
import datasets
from torch.utils.data import DataLoader
def view_model_param(MODEL_NAME, net_params):
    model = gcnetwork.gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += numpy.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    t0 = time.time()
    per_epoch_time = []
    DATASET_NAME = dataset.name
    if net_params['self_loop']:
        print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
        dataset._add_self_loops()
    
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = tensorboardX.SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    numpy.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    
    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))

    model = gcnetwork.gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=params['lr_reduce_factor'],patience=params['lr_schedule_patience'],verbose=True)
    
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_MAEs, epoch_val_MAEs = [], [] 
        
    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=False, collate_fn=dataset.collate)
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=False, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=False, collate_fn=dataset.collate)
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, epoch_train_mae, optimizer = train_test_gcn.train_epoch(model, optimizer, device, train_loader, epoch)
                 
                epoch_val_loss, epoch_val_mae = train_test_gcn.evaluate_network(model, device, val_loader, epoch)
                _, epoch_test_mae = train_test_gcn.evaluate_network(model, device, test_loader, epoch)
                
                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_MAEs.append(epoch_train_mae)
                epoch_val_MAEs.append(epoch_val_mae)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train/_mae', epoch_train_mae, epoch)
                writer.add_scalar('val/_mae', epoch_val_mae, epoch)
                writer.add_scalar('test/_mae', epoch_test_mae, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_MAE=epoch_train_mae, val_MAE=epoch_val_mae,
                              test_MAE=epoch_test_mae)
                
                per_epoch_time.append(time.time()-start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
                
                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    _, test_mae = train_test_gcn.evaluate_network(model, device, test_loader, epoch)
    _, train_mae = train_test_gcn.evaluate_network(model, device, train_loader, epoch)
    print("Test MAE: {:.4f}".format(test_mae))
    print("Train MAE: {:.4f}".format(train_mae))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(numpy.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST MAE: {:.4f}\nTRAIN MAE: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  test_mae, train_mae, epoch, (time.time()-t0)/3600, numpy.mean(per_epoch_time)))

def main():    
    """
        USER CONTROLS
    """
    
    
    config = {}
    # device
    config['gpu'] = {}
    config['gpu']['use'] = False
    config['gpu']['id'] = 0
    device = torch.device("cpu")
    # model, dataset, out_dir
    MODEL_NAME = "GCN"
    DATASET_NAME = "AQSOL"
    dataset = datasets.MoleculeDataset(DATASET_NAME)
    out_dir = "results"
    # parameters
    params = {}
    params['seed'] = 42
    params['epochs'] = 20
    params['batch_size'] = 128
    params['init_lr'] = 0.001
    params['lr_reduce_factor'] = 0.5
    params['lr_schedule_patience'] = 10
    params['min_lr'] = 1e-5
    params['weight_decay'] = 0.0
    params['print_epoch_interval'] = 1
    params['max_time'] = 12

    net_params = {
        "L": 4,
        "hidden_dim": 145,
        "out_dim": 145,
        "residual": True,
        "readout": "mean",
        "in_feat_dropout": 0.0,
        "dropout": 0.0,
        "batch_norm": True,
        "self_loop": False
    }
    # network parameters
    net_params['device'] = device
    net_params['gpu_id'] = 0
    net_params['batch_size'] = params['batch_size']    
    
    # ZINC
    net_params['num_atom_type'] = dataset.num_atom_type
    net_params['num_bond_type'] = dataset.num_bond_type
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)

main()