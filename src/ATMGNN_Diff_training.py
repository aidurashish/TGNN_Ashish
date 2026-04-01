"""
    Training and evaluation script for the ATMGNN_Diff model.

    The ATMGNN encoder backbone is warm-started from an already-trained ATMGNN
    checkpoint (produced by ATMGNN_training.py) when one is available.

    Implements a rolling-window experiment:
    For each country, prediction shift, and test day:
        1.  Data is split into training, validation and test sets.
        2.  Model is trained.
        3.  Per-region forecast errors, metrics, and uncertainty estimates are recorded.
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
from models import ATMGNN_Diff
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')   # non-interactive backend; safe for scripts with no display
import matplotlib.pyplot as plt

# === FUNCTIONS ===

# Weight controlling the relative importance of the SEIR biological consistency penalty
# applied to the auxiliary direct-head loss during diffusion training.
SEIR_LAMBDA = 0.1

def train(adj, features, y, node_weights=None):
    """
    Runs one forward pass (diffusion denoising loss) and backward pass and updates
    the model weights.

    ARGS:
        adj          (torch.sparse_coo_tensor): Batch adjacency matrix.
        features     (torch.FloatTensor): Batch node feature matrix.
        y            (torch.FloatTensor): Ground-truth target values for this batch.
        node_weights (torch.FloatTensor | None): Per-node loss weights for scale balancing.

    RETURNS:
        output     (torch.Tensor): Ground-truth y (size placeholder; predictions come from the fc head at test time).
        loss_train (torch.Tensor): Diffusion training loss scalar.
    """

    optimizer.zero_grad()

    # Diffusion training: epsilon-prediction loss on encoder conditioning.
    loss_train = model.compute_diffusion_loss(adj, features, y, node_weights=node_weights)
    loss_train.backward()
    for p in model.parameters():
        if p.grad is not None:
            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Return target as size placeholder (output.size(0) is used for AverageMeter).
    output = y
    return output, loss_train


def test(adj, features, y, node_weights=None):
    """
    Runs a forward pass without updating weights, used for validation and testing.
    Uses the inherited deterministic fc head for a stable point forecast.

    ARGS:
        adj          (torch.sparse_coo_tensor): Batch adjacency matrix.
        features     (torch.FloatTensor): Batch node feature matrix.
        y            (torch.FloatTensor): Ground-truth target values for this batch.
        node_weights (torch.FloatTensor | None): Per-node loss weights for scale balancing.

    RETURNS:
        output    (torch.Tensor): Model predictions for this batch.
        loss_test (torch.Tensor): MSE loss scalar.
    """

    with torch.no_grad():
        output = model(adj, features)   # n_samples=1 -> deterministic fc head
        if node_weights is not None:
            w = node_weights.repeat(output.size(0) // node_weights.size(0))
            loss_test = (w * (output - y) ** 2).mean()
        else:
            loss_test = F.mse_loss(output, y)
    return output, loss_test


def _plot_loss_curve(train_losses, val_losses, model_name, country, out_dir):
    """
    Saves a train/val loss vs epoch curve for one (model, country) pair.
    Uses the final training run (shift=0, last test_sample — most training data).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train loss', linewidth=1.5)
    ax.plot(epochs, val_losses,   label='Val loss',   linewidth=1.5, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('{} — {} — Loss / Epoch'.format(model_name, country))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, '{}_{}_loss_curve.png'.format(model_name, country))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('  [PLOT] Loss curve saved to {}'.format(path))


def _plot_predictions_vs_actuals(pred_store, model_name, country, out_dir):
    """Saves a grid of subplots (one per shift) comparing mean predicted vs mean actual daily
    case counts (averaged across all regions) for one (model, country) pair."""
    shifts = sorted(pred_store.keys())
    if not shifts:
        return
    ncols = min(len(shifts), 4)
    nrows = (len(shifts) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    for i, shift in enumerate(shifts):
        ax = axes[i // ncols][i % ncols]
        mean_pred, mean_true = pred_store[shift]
        days = range(1, len(mean_true) + 1)
        ax.plot(days, mean_true, label='Actual',    linewidth=1.5, color='steelblue')
        ax.plot(days, mean_pred, label='Predicted', linewidth=1.5, color='tomato', linestyle='--')
        ax.set_title('Shift +{} day{}'.format(shift, '' if shift == 1 else 's'))
        ax.set_xlabel('Test day')
        ax.set_ylabel('Mean case count')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
    # Hide any unused subplot panels
    for j in range(len(shifts), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle('{} — {} — Predictions vs Actuals'.format(model_name, country), fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, '{}_{}_predictions_vs_actuals.png'.format(model_name, country))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('  [PLOT] Predictions vs actuals saved to {}'.format(path))


# === MAIN ===

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Starting learning rate.')
    parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
    parser.add_argument('--batch-size', type=int, default=16, help='Size of batch.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--window', type=int, default=7, help='Size of window for features.')
    parser.add_argument('--graph-window', type=int, default=7, help='Size of window for graphs.')
    parser.add_argument('--early-stop', type=int, default=100, help='How many epochs to wait before stopping.')
    parser.add_argument('--start-exp', type=int, default=15, help='The first day to start the predictions.')
    parser.add_argument('--ahead', type=int, default=7, help='The number of days ahead of the train set the predictions should reach.')
    parser.add_argument('--sep', type=int, default=10, help='Seperator for validation and train set.')
    parser.add_argument('--rand-weights', type=bool, default=False, help="True or False. Enable ablation where weights in the adjacency matrix are shuffled.")
    parser.add_argument('--rand-seed', type=int, default=0, help="Specify the random seeds for reproducibility.")
    parser.add_argument('--edge-decay', type=float, default=0.5, help='Exponential time decay for edge weights across the graph window (Set to 0.0 to disable decay).')
    parser.add_argument('--diffusion-steps', type=int, default=100, help='Number of DDPM denoising steps T.')
    parser.add_argument('--num-samples', type=int, default=30, help='Number of diffusion samples at inference for uncertainty estimation.')

    args = parser.parse_args()

    # Fix all random seeds so results are reproducible across runs.
    torch.manual_seed(args.rand_seed)
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)

    # Use GPU if available, otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    print("\n" + "="*60)
    print("  ATMGNN_Diff Training Run")
    print("="*60)
    print("  Device          : {}".format(device))
    print("  Models          : ATMGNN_Diff")
    print("  Countries       : IT, EN, FR")
    print("  Shifts          : 0 to {}".format(args.ahead - 1))
    print("  Epochs          : {} (early stop after {})".format(args.epochs, args.early_stop))
    print("  Diffusion steps : {}".format(args.diffusion_steps))
    print("  Inference smpls : {}".format(args.num_samples))
    print("  Rand seed       : {}".format(args.rand_seed))
    print("="*60 + "\n")

    print("[SETUP] Loading datasets...")
    
    # Load graphs, features, labels, and targets for all four countries at once.
    meta_labs, meta_graphs, meta_features, meta_y = read_datasets(args.window, args.rand_weights)
    print("[SETUP] Datasets loaded.\n")

    for country in ["IT", "EN", "FR"]:

        if country == "IT":     # Italy
            idx = 0
        elif country == "EN":   # Great Britain
            idx = 2
        elif country == "FR":   # France
            idx = 3

        # Extract this country's data from the shared meta-lists.
        labels   = meta_labs[idx]
        gs_adj   = meta_graphs[idx]
        features = meta_features[idx]
        y        = meta_y[idx]
        n_samples = len(gs_adj)                        # total number of days available
        nfeat     = meta_features[idx][0].shape[1]     # number of input features per node

        n_nodes = gs_adj[0].shape[0]
        print("\n" + "-"*60)
        print("  Country: {}  |  Nodes: {}  |  Days available: {}".format(country, n_nodes, n_samples))
        print("-"*60)

        # Per-node inverse-frequency weights: regions with larger mean case counts receive less
        # weight so the loss is balanced across all scales.
        mean_cases = labels.values.astype(float).mean(axis=1)  # mean daily cases per region
        inv_weights = 1.0 / (np.log1p(mean_cases) + 1.0)       # inverse of log-scale magnitude
        inv_weights = inv_weights / inv_weights.mean()           # normalise so mean weight = 1
        node_weights = torch.FloatTensor(inv_weights).to(device)

        # Create output directories if they don't exist yet.
        for _dir in ['../results', '../Checkpoints', '../Predictions', '../figures/training']:
            if not os.path.exists(_dir):
                os.makedirs(_dir)

        for args.model in ['ATMGNN_Diff']:
            print("\n[MODEL] Starting training: {} on {}".format(args.model, country))
            _loss_history = None    # (train_losses, val_losses) from shift=0 last test_sample
            _pred_store = {}        # shift -> (mean_pred_per_day, mean_true_per_day)

            # Predicts 0, 1, ..., 'ahead' - 1 days into the future.
            for shift in list(range(0, args.ahead)):
                print("\n  [SHIFT {}/{}] Model={} Country={}".format(shift + 1, args.ahead, args.model, country))

                # Resume check: skip this shift if all three outputs already exist. ---
                _pred_path    = "../Predictions/predict_{}_shift{}_{}.csv".format(args.model, shift, country)
                _truth_path   = "../Predictions/truth_{}_shift{}_{}.csv".format(args.model, shift, country)
                _results_path = "../results/results_{}_temporal.csv".format(country)
                _results_row  = "{}_AGW_MMR_{},{},".format(args.model, args.rand_weights, shift)
                _results_done = False
                if os.path.exists(_results_path):
                    with open(_results_path) as _f:
                        _results_done = any(line.startswith(_results_row) for line in _f)
                if os.path.exists(_pred_path) and os.path.exists(_truth_path) and _results_done:
                    print("    [SKIP] Outputs already exist, skipping.")
                    continue

                result  = []                                   # stores MAE per test day
                y_pred  = np.empty((n_nodes, 0), dtype=int)   # accumulates predictions column by column
                y_true  = np.empty((n_nodes, 0), dtype=int)   # accumulates ground-truth column by column
                y_uncert = np.empty((n_nodes, 0), dtype=float) # per-node uncertainty estimates
                y_val   = []
                exp     = 0

                n_test_days = n_samples - shift - args.start_exp
                print("    Rolling window: {} test days ({} to {})".format(
                    n_test_days, args.start_exp, n_samples - shift - 1))

                # Rolling-window loop: each iteration moves the test day one step forward.
                for test_sample in range(args.start_exp, n_samples - shift):
                    exp += 1
                    print("    [Day {}/{} | test_sample={}]".format(exp, n_test_days, test_sample),
                        end=" ", flush=True)

                    # === DATA SPLITTING ===
                    
                    idx_train = list(range(args.window - 1, test_sample - args.sep))
                    idx_val   = list(range(test_sample - args.sep, test_sample, 2))
                    idx_train = idx_train + list(range(test_sample - args.sep + 1, test_sample, 2))

                    # Augment training with time-reversed samples.
                    _augment = True
                    adj_train, features_train, y_train = generate_new_batches(
                        gs_adj, features, y, idx_train, args.graph_window, shift,
                        args.batch_size, device, test_sample,
                        decay=args.edge_decay, augment_reverse=_augment)
                    adj_val, features_val, y_val = generate_new_batches(
                        gs_adj, features, y, idx_val, args.graph_window, shift,
                        args.batch_size, device, test_sample, decay=args.edge_decay)
                    adj_test, features_test, y_test = generate_new_batches(
                        gs_adj, features, y, [test_sample], args.graph_window, shift,
                        args.batch_size, device, test_sample, decay=args.edge_decay)

                    n_train_batches = ceil((len(idx_train) * (2 if _augment else 1)) / args.batch_size)

                    # === TRAINING ===

                    # Re-initialise model and optimizer fresh for each (test_sample, shift) pair.
                    best_val_acc  = float('inf')
                    max_restarts  = 3
                    restart_count = 0
                    stop          = False

                    while not stop:
                        restart_count += 1
                        if restart_count > max_restarts:
                            print("\n    [WARN] Max restarts ({}) exceeded for test_sample={}. "
                                "Using best checkpoint found.".format(max_restarts, test_sample))
                            stop = True
                            break

                        model = ATMGNN_Diff(
                            nfeat=nfeat, nhidden=args.hidden, nout=1, n_nodes=n_nodes,
                            window=args.graph_window, dropout=args.dropout, nhead=1,
                            diffusion_steps=args.diffusion_steps, decoder_hidden=128
                        ).to(device)

                        # Warm-start the encoder from the already-trained ATMGNN checkpoint.
                        _atmgnn_ckpt = '../Checkpoints/model_best_ATMGNN_shift{}_{}_RW_{}_seed{}_AG.pth.tar'.format(
                            shift, country, args.rand_weights, args.rand_seed)
                        if os.path.exists(_atmgnn_ckpt):
                            _src = torch.load(_atmgnn_ckpt, weights_only=False)['state_dict']
                            _dst = model.state_dict()
                            _dst.update({k: v for k, v in _src.items() if not k.startswith('diffusion.')})
                            model.load_state_dict(_dst)
                
                            # Freeze only the heavy GCN backbone; let mix, attention, and fc layers fine-tune alongside the diffusion decoder for better conditioning.
                            _frozen_prefixes = ('bottom_encoder.', 'middle_encoder.', 'middle_linear.')
                            for name, param in model.named_parameters():
                                if name.startswith(_frozen_prefixes):
                                    param.requires_grad_(False)

                        optimizer = optim.Adam(
                            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

                        val_among_epochs   = []
                        train_among_epochs = []
                        stop               = False

                        for epoch in range(args.epochs):
                            start = time.time()

                            model.train()
                            train_loss = AverageMeter()
                            for batch in range(n_train_batches):
                                output, loss = train(
                                    adj_train[batch], features_train[batch], y_train[batch],
                                    node_weights=node_weights)
                                train_loss.update(loss.data.item(), output.size(0))

                            model.eval()
                            # Use diffusion loss for validation to give a fair training signal.
                            with torch.no_grad():
                                val_loss = float(model.compute_diffusion_loss(
                                    adj_val[0], features_val[0], y_val[0],
                                    node_weights=node_weights).item())

                            if epoch % 50 == 0:
                                print("\n      Epoch {:03d}/{:d}  train_loss={:.5f}  "
                                    "val_loss={:.5f}  time={:.3f}s".format(
                                        epoch + 1, args.epochs, train_loss.avg,
                                        val_loss, time.time() - start),
                                    end="", flush=True)

                            train_among_epochs.append(train_loss.avg)
                            val_among_epochs.append(val_loss)

                            # Save a checkpoint whenever the model achieves a new best val loss.
                            if val_loss < best_val_acc:
                                best_val_acc = val_loss
                                torch.save({
                                    'state_dict': model.state_dict(),
                                    'optimizer' : optimizer.state_dict(),
                                    'edge_decay' : args.edge_decay,
                                }, '../Checkpoints/model_best_{}_shift{}_{}_RW_{}_seed{}_AG.pth.tar'.format(
                                    args.model, shift, country, args.rand_weights, args.rand_seed))

                            # Restart if training has diverged.
                            if not np.isfinite(val_loss) or val_loss > 1e12:
                                print("\n    [WARN] Diverged val_loss ({:.3e}) at epoch {} "
                                    "=> restarting with fresh model.".format(val_loss, epoch + 1))
                                stop = False
                                break

                            # Early-stop if val loss stalled in the first 30 epochs.
                            if epoch < 30 and epoch > 10:
                                if len(set([round(v, 2) for v in val_among_epochs[-20:]])) == 1:
                                    stop = False
                                    break

                            # Early-stop if val loss stalled for args.early_stop epochs.
                            if epoch > args.early_stop:
                                if len(set([round(v, 2) for v in val_among_epochs[-args.early_stop:]])) == 1:
                                    print("\n      [EARLY STOP] Converged at epoch {}.".format(epoch + 1))
                                    break

                            stop = True
                            scheduler.step(val_loss)

                    # Capture loss curves from shift=0 for the loss-curve plot.
                    if shift == 0 and 'train_among_epochs' in dir() and len(train_among_epochs) > 0:
                        _loss_history = (list(train_among_epochs), list(val_among_epochs))

                    # === TESTING ===

                    _ckpt_path = '../Checkpoints/model_best_{}_shift{}_{}_RW_{}_seed{}_AG.pth.tar'.format(
                        args.model, shift, country, args.rand_weights, args.rand_seed)
                    if not os.path.exists(_ckpt_path):
                        print("\n    [WARN] No checkpoint was saved for test_sample={} "
                            "(all restarts diverged). Skipping this day.".format(test_sample))
                        del adj_train, features_train, y_train
                        del adj_val, features_val, y_val
                        del adj_test, features_test, y_test
                        torch.cuda.empty_cache()
                        continue

                    checkpoint = torch.load(_ckpt_path, weights_only=False)
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    model.eval()

                    # Point forecast via the deterministic fc head (stable, fast).
                    output, loss = test(adj_test[0], features_test[0], y_test[0])

                    # Diffusion sampling for uncertainty estimation.
                    uncertainty = None
                    if args.num_samples > 1:
                        with torch.no_grad():
                            diff_samples = model(adj_test[0], features_test[0], n_samples=args.num_samples)
                            uncertainty = diff_samples.std(dim=0)

                    o_log = output.cpu().detach().numpy()
                    l     = y_test[0].cpu().numpy()
                    o     = np.expm1(np.clip(o_log, 0.0, 10.0))
                    l     = np.expm1(l)

                    error = np.sum(abs(o - l)) / n_nodes
                    print("\nShape of error: {}".format(o.shape))
                    print("Test error=", "{:.5f}".format(error))
                    result.append(error)

                    y_pred = np.append(y_pred, o.reshape(-1, 1), axis=1)
                    y_true = np.append(y_true, l.reshape(-1, 1), axis=1)

                    # Append per-node uncertainty (in original case-count space).
                    if uncertainty is not None:
                        u_log    = uncertainty.cpu().numpy()
                        u_cases  = (np.expm1(np.clip(o_log + u_log, 0.0, 10.0))
                                    - np.expm1(np.clip(o_log - u_log, 0.0, 10.0))) / 2.0
                        y_uncert = np.append(y_uncert, u_cases.reshape(-1, 1), axis=1)

                    del adj_train, features_train, y_train
                    del adj_val, features_val, y_val
                    del adj_test, features_test, y_test
                    torch.cuda.empty_cache()

                # Print summary metrics across all test days for this (country, shift) pair.
                if len(result) == 0:
                    print("\n  [SHIFT {} SUMMARY] Model={} Country={} — no valid predictions "
                        "(all days diverged), skipping.".format(shift, args.model, country))
                    continue

                print("\n  [SHIFT {} SUMMARY] Model={} Country={}".format(shift, args.model, country))
                print("    MAE={:.5f}  std={:.5f}  MSE={:.5f}  RMSE={:.5f}  R2={:.5f}".format(
                    mean_absolute_error(y_true, y_pred),
                    np.std(result),
                    mean_squared_error(y_true, y_pred),
                    np.sqrt(mean_squared_error(y_true, y_pred)),
                    r2_score(y_true, y_pred)))

                if y_pred.shape[1] > 0:
                    _pred_store[shift] = (y_pred.mean(axis=0), y_true.mean(axis=0))

                with open("../results/results_" + country + "_temporal.csv", "a") as fw:
                    fw.write(
                        str(args.model) + "_AGW_MMR_" + str(args.rand_weights) + ","
                        + str(shift)
                        + ",{:.5f}".format(np.mean(result))
                        + ",{:.5f}".format(np.std(result))
                        + ",{:.5f}".format(mean_absolute_error(y_true, y_pred))
                        + ",{:.5f}".format(mean_squared_error(y_true, y_pred))
                        + ",{:.5f}".format(np.sqrt(mean_squared_error(y_true, y_pred)))
                        + ",{:.5f}".format(r2_score(y_true, y_pred)) + "\n")

                np.savetxt("../Predictions/predict_{}_shift{}_{}.csv".format(args.model, shift, country),
                        y_pred, fmt="%.5f", delimiter=',')
                np.savetxt("../Predictions/truth_{}_shift{}_{}.csv".format(args.model, shift, country),
                        y_true, fmt="%.5f", delimiter=',')
                if y_uncert.shape[1] > 0:
                    np.savetxt("../Predictions/uncertainty_{}_shift{}_{}.csv".format(
                        args.model, shift, country), y_uncert, fmt="%.5f", delimiter=',')
                print("    Predictions saved to ../Predictions/")

            # === POST-TRAINING PLOTS ===

            print("\n[PLOT] Generating training plots for {} on {}...".format(args.model, country))
            if _loss_history is not None:
                _plot_loss_curve(_loss_history[0], _loss_history[1], args.model, country,
                                '../figures/training')
            else:
                print('  [PLOT] No loss history captured (all runs may have been skipped).')
            if _pred_store:
                _plot_predictions_vs_actuals(_pred_store, args.model, country, '../figures/training')
            else:
                print('  [PLOT] No predictions captured for plot.')
