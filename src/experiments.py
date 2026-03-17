"""
    Main training and evaluation script for the model across all datasets.
    
    Implements a rolling-window experiment:
    For each country, prediction shift, and test day:
        1.  Data is split into training, validation and test sets.
        2.  Model is trained.
        3.  Per-region forecast errors and metrics are recorded.
"""

# === IMPORTS ===

import os
import time
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from math import ceil
from utils import generate_new_batches, AverageMeter, read_datasets
from src.models import ATMGNN, ATMGNN_Diff
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === FUNCTIONS === 

# Weight controlling the relative importance of the SEIR biological consistency penalty.
SEIR_LAMBDA = 0.1

def train(adj, features, y):
    """
    Runs one forward pass and backward pass and updates the model weights.
    Supports both the deterministic ATMGNN head and the DDPM diffusion decoder.

    ARGS:
        adj      (torch.sparse_coo_tensor): Batch adjacency matrix.
        features (torch.FloatTensor): Batch node feature matrix.
        y        (torch.FloatTensor): Ground-truth target values for this batch.

    RETURNS:
        output     (torch.Tensor): Model predictions for this batch.
        loss_train (torch.Tensor): Training loss scalar.
    """
    
    optimizer.zero_grad()

    if isinstance(model, ATMGNN_Diff):
        
        # Diffusion training: epsilon-prediction loss on encoder conditioning.
        loss_train = model.compute_diffusion_loss(adj, features, y)
        loss_train.backward(retain_graph=True)
        optimizer.step()
        
        # Return target as size placeholder
        output = y
    else:
        output = model(adj, features)
        loss_mse = F.mse_loss(output, y)
        
        # SEIR consistency penalty (predicted case counts cannot be negative).
        seir_penalty = F.relu(-output).mean()
        loss_train = loss_mse + SEIR_LAMBDA * seir_penalty
        loss_train.backward(retain_graph=True)
        optimizer.step()

    return output, loss_train


def test(adj, features, y):
    """
        Runs a forward pass without updating weights, used for validation and testing.

        ARGS:
            adj      (torch.sparse_coo_tensor): Batch adjacency matrix.
            features (torch.FloatTensor): Batch node feature matrix.
            y        (torch.FloatTensor): Ground-truth target values for this batch.

        RETURNS:
            output    (torch.Tensor): Model predictions for this batch.
            loss_test (torch.Tensor): MSE loss scalar.
    """
    
    output = model(adj, features)
    loss_test = F.mse_loss(output, y)
    return output, loss_test

# === MAIN ===

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Starting learning rate.')
    parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
    parser.add_argument('--batch-size', type=int, default=128, help='Size of batch.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--window', type=int, default=7, help='Size of window for features.')
    parser.add_argument('--graph-window', type=int, default=7, help='Size of window for graphs.')
    parser.add_argument('--early-stop', type=int, default=100, help='How many epochs to wait before stopping.')
    parser.add_argument('--start-exp', type=int, default=15, help='The first day to start the predictions.')
    parser.add_argument('--ahead', type=int, default=21, help='The number of days ahead of the train set the predictions should reach.')
    parser.add_argument('--sep', type=int, default=10, help='Seperator for validation and train set.')
    parser.add_argument('--rand-weights', type=bool, default=False, help="True or False. Enable ablation where weights in the adjacency matrix are shuffled.")
    parser.add_argument('--rand-seed', type=int, default=0, help="Specify the random seeds for reproducibility.")
    parser.add_argument('--edge-decay', type=float, default=0.5, help='Exponential time decay for edge weights across the graph window (Set to 0.0 to disable decay).')
    parser.add_argument('--model-name', type=str, default='ATMGNN_Diff', choices=['ATMGNN', 'ATMGNN_Diff'], help='Model architecture to train (ATMGNN = deterministic head, ATMGNN_Diff = diffusion decoder).')
    parser.add_argument('--diffusion-steps', type=int, default=8, help='Number of DDPM denoising steps T (only used with ATMGNN_Diff).')
    parser.add_argument('--num-samples', type=int, default=50, help='Number of diffusion samples at inference for uncertainty estimation (only used with ATMGNN_Diff).')
    
    args = parser.parse_args()
    
    # Fix all random seeds so results are reproducible across runs.
    torch.manual_seed(args.rand_seed)
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    
    # Use GPU if available, otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    
    # Load graphs, features, labels, and targets for all four countries at once.
    meta_labs, meta_graphs, meta_features, meta_y = read_datasets(args.window, args.rand_weights)
    
    # Iterate over each country dataset in order
    for country in ["IT", "ES", "EN", "FR"]:
        
        if(country=="IT"):  # Italy
            idx = 0

        elif(country=="ES"):    # Spain 
            idx = 1

        elif(country=="EN"):    # Great Britain
            idx = 2

        elif(country=="FR"):    # France
            idx = 3

        # Extract this country's data from the shared meta-lists.
        labels = meta_labs[idx]
        gs_adj = meta_graphs[idx]
        features = meta_features[idx]
        y = meta_y[idx]
        n_samples = len(gs_adj)             # total number of days available
        nfeat = meta_features[idx][0].shape[1]  # number of input features per node
        
        n_nodes = gs_adj[0].shape[0]
        print(n_nodes)
        
        # Create output directories if they don't exist yet.
        if not os.path.exists('../results'):
            os.makedirs('../results')
        if not os.path.exists('../Checkpoints'):
            os.makedirs('../Checkpoints')
        if not os.path.exists('../Predictions'):
            os.makedirs('../Predictions')

        
        for args.model in [args.model_name]:   # Selects model architecture via --model-name arg
			# Predicts 0, 1, ..., 'ahead' - 1 days into the future.
            for shift in list(range(0,args.ahead)):

                result = []                                 # stores mean absolute error (MAE) per test day
                y_pred = np.empty((n_nodes, 0), dtype=int)  # accumulates predictions column by column
                y_true = np.empty((n_nodes, 0), dtype=int)  # accumulates ground-truth column by column
                y_uncert = np.empty((n_nodes, 0), dtype=float)  # accumulates per-node uncertainty (std. deviation) for diffusion model
                y_val = []
                exp = 0
                fw = open("../results/results_"+country+"_temporal.csv","a")

                # Rolling-window loop where each iteration moves the test day one step forward.
                for test_sample in range(args.start_exp,n_samples-shift):
                    exp+=1
                    print(test_sample)

                    # === DATA SPLITTING ===
                    # Training indices: all days from the start up to 'sep' days before the test day.
                    idx_train = list(range(args.window-1, test_sample-args.sep))
                    
                    # Validation indices: even-offset days in the 'sep'-day buffer before the test day.
                    idx_val = list(range(test_sample-args.sep,test_sample,2)) 
                    
                    # Add odd-offset days in the same buffer to training (interleaved split).
                    idx_train = idx_train+list(range(test_sample-args.sep+1,test_sample,2))

                    # Convert index lists into batched tensors for train, val, and test.
                    adj_train, features_train, y_train = generate_new_batches(gs_adj, features, y, idx_train, args.graph_window, shift, args.batch_size,device,test_sample, decay=args.edge_decay)
                    adj_val, features_val, y_val = generate_new_batches(gs_adj, features, y, idx_val, args.graph_window,  shift,args.batch_size, device,test_sample, decay=args.edge_decay)
                    
                    # Test is always a single day, i.e., the current test_sample.
                    adj_test, features_test, y_test = generate_new_batches(gs_adj, features, y,  [test_sample], args.graph_window,shift, args.batch_size, device,test_sample, decay=args.edge_decay)

                    n_train_batches = ceil(len(idx_train)/args.batch_size)
                    n_val_batches = 1   # validation is always one batch (one day)
                    n_test_batches = 1  # test is always one batch (one day)


                    # === TRAINING ===
                    
                    # Re-initialise model and optimizer fresh for each (test_sample, shift) pair.
                    stop = False
                    while(not stop):
                        if args.model == "ATMGNN_Diff":
                            model = ATMGNN_Diff(nfeat=nfeat, nhidden=args.hidden, nout=1, n_nodes=n_nodes, window=args.graph_window, dropout=args.dropout, nhead=1, diffusion_steps=args.diffusion_steps).to(device)
                        else:
                            model = ATMGNN(nfeat=nfeat, nhidden=args.hidden, nout=1, n_nodes=n_nodes, window=args.graph_window, dropout=args.dropout, nhead=1).to(device)

                        optimizer = optim.Adam(model.parameters(), lr=args.lr)
                        
                        # Reduce learning rate when validation loss stops improving.
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

                        best_val_acc = 1e8      # tracks the lowest validation loss seen so far
                        val_among_epochs = []
                        train_among_epochs = []
                        stop = False

                        for epoch in range(args.epochs):    
                            start = time.time()

                            model.train()
                            train_loss = AverageMeter()

                            # Train for one epoch
                            for batch in range(n_train_batches):
                                output, loss = train(adj_train[batch], features_train[batch], y_train[batch])
                                train_loss.update(loss.data.item(), output.size(0))

                            # Evaluate on validation set
                            model.eval()

                            output, val_loss = test(adj_val[0], features_val[0], y_val[0])
                            val_loss = float(val_loss.detach().cpu().numpy())


                            # Print results
                            if(epoch%50==0):
                                print("Epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg),"val_loss=", "{:.5f}".format(val_loss), "time=", "{:.5f}".format(time.time() - start))

                            train_among_epochs.append(train_loss.avg)
                            val_among_epochs.append(val_loss)

                            # Early-stop if validation loss hasn't changed in the last 20 epochs (training stalled).
                            if(epoch<30 and epoch>10):
                                if(len(set([round(val_e) for val_e in val_among_epochs[-20:]])) == 1 ):
                                    stop = False
                                    break

                            # Early-stop if validaion loss hasn't changed in the last 50 epochs (fully converged).
                            if( epoch>args.early_stop):
                                if(len(set([round(val_e) for val_e in val_among_epochs[-50:]])) == 1):
                                    print("training has ended.")
                                    break

                            stop = True

                            # Save a checkpoint whenever the model achieves a new best validation loss.
                            if val_loss < best_val_acc:
                                best_val_acc = val_loss
                                torch.save({
                                    'state_dict': model.state_dict(),
                                    'optimizer' : optimizer.state_dict(),
                                    'edge_decay' : args.edge_decay,
                                }, '../Checkpoints/model_best_{}_shift{}_{}_RW_{}_seed{}_AG.pth.tar'.format(args.model, shift, country, args.rand_weights, args.rand_seed))

                            scheduler.step(val_loss)


                    print("validation") 
                    
                    # === TESTING ===
                    
                    test_loss = AverageMeter()

                    # Reload the best checkpoint saved during training before running on the test day.
                    checkpoint = torch.load('../Checkpoints/model_best_{}_shift{}_{}_RW_{}_seed{}_AG.pth.tar'.format(args.model, shift, country, args.rand_weights, args.rand_seed))
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    model.eval()

                    # Multi-sample inference for diffusion model 
                    if args.model == "ATMGNN_Diff" and args.num_samples > 1:
                        with torch.no_grad():
                            samples = model(adj_test[0], features_test[0], n_samples=args.num_samples)
                            output = samples.mean(dim=0)
                            uncertainty = samples.std(dim=0)
                    else:
                        output, loss = test(adj_test[0], features_test[0], y_test[0])
                        uncertainty = None

                    o = output.cpu().detach().numpy()
                    l = y_test[0].cpu().numpy()
                    
                    # Mean absolute error (MAE) averaged over all regions for this test day.
                    error = np.sum(abs(o-l))/n_nodes
                    print("Shape of error: {}".format(o.shape))
			
                    print("test error=", "{:.5f}".format(error))
                    result.append(error)
                    
                    # Append this day's predictions and ground-truth as a new column.
                    y_pred = np.append(y_pred, o.reshape(-1,1), axis=1)
                    y_true = np.append(y_true, l.reshape(-1,1), axis=1)

                    # Append per-node uncertainty (standard deviation across samples).
                    if uncertainty is not None:
                        y_uncert = np.append(y_uncert, uncertainty.cpu().numpy().reshape(-1,1), axis=1)

                # Print summary metrics across all test days for this (country, shift) pair.
                print("{:.5f}".format(np.mean(result))+",{:.5f}".format(np.std(result))+",{:.5f}".format(  np.sum(labels.iloc[:,args.start_exp:test_sample].mean(1))))
                print("Aux metrics: {:.5f}".format(mean_absolute_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred, squared=False))+",{:.5f}".format(r2_score(y_true, y_pred)))

                # Write metrics to CSV and save raw prediction/truth arrays to disk.
                fw.write(str(args.model)+"_AGW_MMR_"+str(args.rand_weights)+","+str(shift)+",{:.5f}".format(np.mean(result))+",{:.5f}".format(np.std(result))+",{:.5f}".format(mean_absolute_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred, squared=False))+",{:.5f}".format(r2_score(y_true, y_pred))+"\n")
                fw.close()
                np.savetxt("../Predictions/predict_{}_shift{}_{}.csv".format(args.model, shift, country), y_pred, fmt="%.5f", delimiter=',')
                np.savetxt("../Predictions/truth_{}_shift{}_{}.csv".format(args.model, shift, country), y_true, fmt="%.5f", delimiter=',')
                if y_uncert.shape[1] > 0:
                    np.savetxt("../Predictions/uncertainty_{}_shift{}_{}.csv".format(args.model, shift, country), y_uncert, fmt="%.5f", delimiter=',')