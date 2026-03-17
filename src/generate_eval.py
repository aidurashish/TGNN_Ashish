"""
    Script for loading pre-trained model checkpoints and evaluating them autoregressively on data for each country and shift.
"""

# === IMPORTS === 

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from utils import generate_new_batches, read_datasets
from src.models import ATMGNN, ATMGNN_Diff
import warnings

# === FUNCTION ===

warnings.simplefilter(action='ignore', category=FutureWarning)

def output_val(gs_adj, features, y, model, checkpoint_name, shift, n_samples=1):
    """
        Loads a saved checkpoint, runs the model on one evaluation day, and returns predictions.
        For diffusion models, draws multiple samples to produce uncertainty estimates.

        ARGS:
            gs_adj          (list[sp.spmatrix]): Sparse adjacency matrices, one per day.
            features        (list[np.ndarray]): Node feature matrices, one per day.
            y               (list): Ground-truth case counts, one list per day.
            model           (nn.Module): Uninitialised model instance to load weights into.
            checkpoint_name (str): File path of the saved '.pth.tar' checkpoint to load.
            shift           (int): How many days ahead this checkpoint was trained to predict.
            n_samples       (int): Number of diffusion samples for uncertainty estimation.

        RETURNS:
            o           (np.ndarray): Predicted case counts of shape: [n_nodes,].
            l           (np.ndarray): Ground-truth case counts of shape: [n_nodes,].
            uncertainty (np.ndarray | None): Per-node std across samples, or None for deterministic models.
    """

    # Loads the checkpoint first so we can read the decay value that was used at training time.
    checkpoint = torch.load(checkpoint_name, map_location=torch.device('cpu'))
    decay = checkpoint.get('edge_decay', 0.0)   # 0.0 (equal weights) to preserve backward compatibility.

    # Build a single-day batch using the training-time decay value retrieved from the checkpoint.
    adj_test, features_test, y_test = generate_new_batches(gs_adj, features, y, [args.eval_start], args.graph_window, shift, args.batch_size, device, -1, decay=decay)

    # Restore weights and optimizer state from the checkpoint.
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()

    uncertainty = None
    if isinstance(model, ATMGNN_Diff) and n_samples > 1:
        with torch.no_grad():
            samples = model(adj_test[0], features_test[0], n_samples=n_samples)
            output = samples.mean(dim=0)
            uncertainty = samples.std(dim=0).cpu().detach().numpy()
    else:
        output = model(adj_test[0], features_test[0])

    # Move results to CPU numpy arrays for saving.
    o = output.cpu().detach().numpy()
    l = y_test[0].cpu().numpy()

    return o, l, uncertainty

# === MAIN === 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Starting learning rate.')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--batch-size', type=int, default=64, help='Size of batch.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--window', type=int, default=7, help='Size of window for features.')
    parser.add_argument('--graph-window', type=int, default=7, help='Size of window for graphs.')
    parser.add_argument('--early-stop', type=int, default=100, help='How many epochs to wait before stopping.')
    parser.add_argument('--start-exp', type=int, default=15, help='The first day to start the predictions.')
    parser.add_argument('--ahead', type=int, default=14, help='The number of days ahead of the train set the predictions should reach.')
    parser.add_argument('--sep', type=int, default=10, help='Seperator for validation and train set.')
    parser.add_argument('--eval-start', type=int, default=7, help='Start day offset for evaluation on new data.')
    parser.add_argument('--rand-weights', type=bool, default=False, help="True or False. Enable ablation where weights in the adjacency matrix are shuffled.")
    parser.add_argument('--model-name', type=str, default='ATMGNN_Diff', choices=['ATMGNN', 'ATMGNN_Diff'], help='Model architecture to evaluate (must match trained checkpoint).')
    parser.add_argument('--diffusion-steps', type=int, default=8, help='Number of DDPM denoising steps T (only used with ATMGNN_Diff).')
    parser.add_argument('--num-samples', type=int, default=50, help='Number of diffusion samples at inference for uncertainty estimation.')

    args = parser.parse_args()
    
    # Use GPU if available, otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    
    # Load graphs, features, labels, and targets for all four countries at once.
    meta_labs, meta_graphs, meta_features, meta_y = read_datasets(args.window)

    # Iterate over each country dataset in order.
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
        if not os.path.exists('../eval'):
            os.makedirs('../eval')


        for args.model in [args.model_name]:   # Selects model architecture via --model-name arg
            # Pre-allocate matrices to collect predictions and ground-truth for all shifts at once.
            prediction_set = np.empty((args.ahead, n_nodes), np.float64)
            truth_set = np.empty((args.ahead, n_nodes), np.float64)
            uncertainty_set = np.empty((args.ahead, n_nodes), np.float64) if args.model == "ATMGNN_Diff" else None

            # Loop over each prediction horizon (e.g.: 0 = next day, 1 = two days ahead, etc.).
            for shift in list(range(0,args.ahead)):

                result = []
                y_pred = np.empty((n_nodes, 0), dtype=int)
                y_true = np.empty((n_nodes, 0), dtype=int)

                print("Evaluating {} at shift {}...".format(args.model, shift))

                # === INITIALIZATION ===
                
                # Build a fresh model and optimizer to receive the loaded checkpoint weights.
                if args.model == "ATMGNN_Diff":
                    model = ATMGNN_Diff(nfeat=nfeat, nhidden=args.hidden, nout=1, n_nodes=n_nodes, window=args.graph_window, dropout=args.dropout, nhead=1, diffusion_steps=args.diffusion_steps).to(device)
                else:
                    model = ATMGNN(nfeat=nfeat, nhidden=args.hidden, nout=1, n_nodes=n_nodes, window=args.graph_window, dropout=args.dropout, nhead=1).to(device)
                optimizer = optim.Adam(model.parameters(), lr=args.lr)

                # === TESTING === 
                
                # Run inference using the best checkpoint for this (country, shift) pair.
                n_eval_samples = args.num_samples if args.model == "ATMGNN_Diff" else 1
                o, l, u = output_val(gs_adj, features, y, model, '../Checkpoints/model_best_{}_shift{}_{}.pth.tar'.format(args.model, shift, country), shift, n_samples=n_eval_samples)
                
                # Store this shift's results in the pre-allocated matrices.
                prediction_set[shift] = o
                truth_set[shift] = l
                if uncertainty_set is not None and u is not None:
                    uncertainty_set[shift] = u
                
                # Save per-shift predictions and ground-truth as individual CSV files.
                np.savetxt("../eval/predict_{}_shift{}_{}.csv".format(args.model, shift, country), o.reshape(1, -1), fmt="%.5f", delimiter=',')
                np.savetxt("../eval/truth_{}_shift{}_{}.csv".format(args.model, shift, country), l.reshape(1, -1), fmt="%.5f", delimiter=',')
                if u is not None:
                    np.savetxt("../eval/uncertainty_{}_shift{}_{}.csv".format(args.model, shift, country), u.reshape(1, -1), fmt="%.5f", delimiter=',')