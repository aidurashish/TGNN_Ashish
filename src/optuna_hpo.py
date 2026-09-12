"""
    Optuna hyperparameter optimisation for DiffATMGNN for three selected hyperparameters (lr, hidden, dropout).
    All completed trials are persisted to a SQLite database.
"""

# === IMPORTS ===

import os   # File discovery and management
import sys  # Import local files from the src/ directory of this project
import json # Save HPO results as JSON  
import argparse # For passing custom arguments when running the HPO script
import random   # Random seed generation
from math import ceil   # Rounding/estimating number of batches for optimization

import numpy as np  # For arithmetic operations
import torch    # Core deep-learning framework 
import torch.nn.functional as F # NN functions with no memory
import torch.optim as optim # Adam optimizer
import optuna   # Core hyperparameter optimization framework for automating hyperparameter search
import optuna.visualization # For plotting hyperparameter optimization results and summary
from optuna.pruners import MedianPruner # To kill bad HPO trials early instead of wasting time

# Navigate within project directory 
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)

from utils import generate_new_batches, read_datasets   # Retrieve and process datasets for the model(s)
from models import ATMGNN_Diff  # Imports main DiffATMGNN model for HPO

# === CONSTANTS ===

SEIR_LAMBDA = 0.1          # Disease biology penalty weight 
COUNTRIES   = ["IT", "EN", "FR", "ES"]
COUNTRY_IDX = {"IT": 0, "ES": 1, "EN": 2, "FR": 3}

# Fixed untuned settings used during HPO trials.
_SHIFT        = 0
_WINDOW       = 7
_GRAPH_WINDOW = 7
_SEP          = 10
_BATCH_SIZE   = 32
_EDGE_DECAY   = 0.5
_START_EXP    = 15


# === TRAINING HELPERS ===

def _train_step_diff(model, optimizer, adj, features, y, node_weights):
    """
        Performs one forward + backward pass for DiffATMGNN
    """
    optimizer.zero_grad()   # Clear old gradients so that each step if fresh
    loss = model.compute_diffusion_loss(adj, features, y, node_weights=node_weights)    # Compute MSE loss for diffusion 
    loss.backward() # Backpropagation
    
    # Per-parameter loop
    for p in model.parameters():
        if p.grad is not None:  # Only parameters with a gradient
            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN/inf values with 0 to avoid malfunctioning
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    # Cap total gradient size at 1.0 max to prevent exploding
    optimizer.step()    # Update model weights with cleaned up gradients
    return loss.item()  # Return loss as a plain number 


def _val_loss(model, adj, features, y, node_weights):
    """
        Performs a weighted MSE validation loss calculation
    """
    with torch.no_grad():   # Disable tracking of gradients
        output = model(adj, features)   # Get model output (predictions)
        w      = node_weights.repeat(output.size(0) // node_weights.size(0))    # Apply node weights
        return (w * (output - y) ** 2).mean().item()    # Calculate weighted MSE val_loss and return a float


# === MAIN ===

def objective(trial, model_name, dataset, device, hpo_epochs):
    """
        Optuna's objective function which optimizes the selected hyperparameters, builds a model per country, trains for
        a certain amount of epochs using a fixed mid-range split, and returns the mean best validation loss across all four countries.

        NOTE: Uses MedianPruner to kill bad trials early.
    """
    # Define search space for the three hyperparameters
    lr      = trial.suggest_float("lr",      1e-4, 1e-2, log=True)
    hidden  = trial.suggest_categorical("hidden",  [32, 64, 128, 256])  # Discrete values instead of a range since lr is usually designated into powers of 2
    dropout = trial.suggest_float("dropout", 0.2,  0.6)

    meta_labs, meta_graphs, meta_features, meta_y = dataset # Split the loaded dataset tuple into 4 parts, each indexed by country.

    # Build per-country data and model before the epoch loop
    contexts = []
    for country in COUNTRIES:
        cidx = COUNTRY_IDX[country]

        labels   = meta_labs[cidx]
        gs_adj   = meta_graphs[cidx]
        feats    = meta_features[cidx]
        y        = meta_y[cidx]
        n_samp   = len(gs_adj)
        nfeat    = feats[0].shape[1]
        n_nodes  = gs_adj[0].shape[0]

        # Node weights 
        mean_cases   = labels.values.astype(float).mean(axis=1) # Avg. case count per node per time
        inv_w        = 1.0 / (np.log1p(mean_cases) + 1.0)   # Inverse-frequency: low-case nodes get higher weight
        inv_w        = inv_w / inv_w.mean() # Normalize to mean = 1
        node_weights = torch.FloatTensor(inv_w).to(device)  # Convert to GPU Tensor(s)

        # Fixed mid-range split: picks one fixed midpoint in the timeline instead of testing every rolling window
        test_sample = _START_EXP + (n_samp - _START_EXP) // 2

        # Split logic 
        idx_train = list(range(_WINDOW - 1, test_sample - _SEP))
        idx_val   = list(range(test_sample - _SEP, test_sample, 2))
        idx_train = idx_train + list(range(test_sample - _SEP + 1, test_sample, 2))

        if len(idx_train) == 0 or len(idx_val) == 0:    # If a country has little training data, mark as 'None' and move on to avoid crashing.
            contexts.append(None)
            continue

        # Generate training and validation batches
        adj_train, feat_train, y_train = generate_new_batches(
            gs_adj, feats, y, idx_train, _GRAPH_WINDOW, _SHIFT,
            _BATCH_SIZE, device, test_sample,
            decay=_EDGE_DECAY, augment_reverse=True)
        
        adj_val, feat_val, y_val = generate_new_batches(
            gs_adj, feats, y, idx_val, _GRAPH_WINDOW, _SHIFT,
            _BATCH_SIZE, device, test_sample, decay=_EDGE_DECAY)    # NOTE: No time-reversal augmentation in validation as model should know to identify reverse patterns by now from training batches

        # Estimate number of batches 
        n_batches = ceil(len(idx_train) * 2 / _BATCH_SIZE)

        # Build the model
        # Left as a conditional so as to optimize other potential models, if necessary.
        if model_name == "DiffATMGNN":
            model = ATMGNN_Diff(
                nfeat=nfeat, nhidden=hidden, nout=1, n_nodes=n_nodes,
                window=_GRAPH_WINDOW, dropout=dropout, nhead=1,
            ).to(device)

            # Warm-start encoder weights from an existing ATMGNN checkpoint (if exists).
            # NOTE: This section is no longer relevant due to the abscence of ATMGNN, which was used initially during DiffATMGNN's training.
            _ckpt_path = os.path.join(
                _ROOT_DIR, 'checkpoints',
                'model_best_ATMGNN_shift0_{}_RW_False_seed0_AG.pth.tar'.format(country))
            if os.path.exists(_ckpt_path):
                _src_state = torch.load(_ckpt_path, map_location=device,weights_only=False)['state_dict']   # Load saved weights
                _dst_state = model.state_dict() # Fresh DiffATMGNN model weights
                
                # Copy weights that exist in Diff model, aren't diffusion-specific layers, and have matching tensor shape.
                _dst_state.update({
                    k: v for k, v in _src_state.items()
                    if k in _dst_state
                    and not k.startswith('diffusion.')
                    and v.shape == _dst_state[k].shape
                })
                model.load_state_dict(_dst_state)   # Apply merged weights

        # Adam Optimizer used only on the three selected hyperparameters using sampled 'lr'
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        
        # Scheduler: it lowers the learning rate if validation loss stops improving after 5 checks
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        contexts.append(dict(
            country      = country,
            model        = model,
            optimizer    = optimizer,
            scheduler    = scheduler,
            adj_train    = adj_train,
            feat_train   = feat_train,
            y_train      = y_train,
            adj_val      = adj_val,
            feat_val     = feat_val,
            y_val        = y_val,
            n_batches    = n_batches,
            node_weights = node_weights,
            best_val     = float('inf'),
        ))

    # Filter out any skipped countries.
    active = [c for c in contexts if c is not None]
    if not active:
        raise optuna.exceptions.TrialPruned()

    # Select the correct training function
    # Left as a variable, so as to accomodate additional training functions for other future models for HPO
    train_fn = _train_step_diff

    # Per-epoch loop for training all countries 
    for epoch in range(hpo_epochs):
        epoch_val_losses = []

        # Per-country loop
        for ctx in active:
            # Train one epoch
            ctx['model'].train()
            for b in range(ctx['n_batches']):
                train_fn(
                    ctx['model'], ctx['optimizer'],
                    ctx['adj_train'][b], ctx['feat_train'][b], ctx['y_train'][b],
                    ctx['node_weights'],
                )

            # Validate
            ctx['model'].eval() # Disables dropout
            vl = _val_loss(
                ctx['model'],
                ctx['adj_val'][0], ctx['feat_val'][0], ctx['y_val'][0],
                ctx['node_weights'],
            )
            if not np.isfinite(vl):
                vl = 1e9   # If loss is NaN/Inf (model diverged), replace with a large penalty instead.

            ctx['best_val'] = min(ctx['best_val'], vl)  # track current country's best/lowest validation loss seen so far.
            ctx['scheduler'].step(vl)   # Adjust learning rate based on above validation loss
            epoch_val_losses.append(vl) # Store current epoch loss for calculating the averages later

        mean_val = float(np.mean(epoch_val_losses)) # Calculate average loss across all 4 countries this epoch.

        trial.report(mean_val, epoch)   # Report to Optuna the current progress, so it can compare against other trials.
        if trial.should_prune():    # if current trial is worse than the median of past trials (via MedianPruner), kill it early to save compute.
            raise optuna.exceptions.TrialPruned()   

    # Return the mean of each country's best validation loss
    return float(np.mean([ctx['best_val'] for ctx in active]))


# === SAVE BEST PARAMS ===

def save_best_params(study, model_name, results_dir):
    """
        Saves the best hyperparameters from completed trials to a JSON file.
        It is called after every completed trial and at the end of the HPO run.
    """
    os.makedirs(results_dir, exist_ok=True)

    completed = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return  # Nothing to save yet.

    try:
        best_val    = study.best_value
        best_params = study.best_params
    except ValueError:
        return  # No completed trials with a valid value.

    out = {
        "model"              : model_name,
        "n_trials_completed" : len(completed),
        "best_value"         : best_val,
        "best_params"        : best_params,
    }
    path = os.path.join(results_dir, "{}_best_params.json".format(model_name))
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print("[HPO] Best params saved -> {}".format(path))
    print("      lr={lr:.6f}  hidden={hidden}  dropout={dropout:.4f}".format(**best_params))


# === PLOTS ===

def plot_study(study, model_name, figures_dir):
    """
        Generates and saves three interactive HTML plots using the optuna.visualization library:

            1) Optimisation history:    trial values and the running best.
            2) Parameter importance:    which of the three hyperparameters mattered most.
            3) Parallel coordinates:    overview of every trial.

        All plots are saved to figures/hpo/.
    """
    os.makedirs(figures_dir, exist_ok=True)

    completed = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        print("[HPO] Fewer than 2 completed trials — skipping plots.")
        return

    plots = [
        (optuna.visualization.plot_optimization_history,  '{}_opt_history.html'),
        (optuna.visualization.plot_param_importances,     '{}_param_importance.html'),
        (optuna.visualization.plot_parallel_coordinate,   '{}_parallel_coords.html'),
    ]

    for plot_fn, filename_template in plots:
        try:
            fig  = plot_fn(study)
            path = os.path.join(figures_dir, filename_template.format(model_name))
            fig.write_html(path)
            print('[HPO] Saved → {}'.format(path))
        except Exception as exc:
            print('[HPO] Could not generate {}: {}'.format(filename_template, exc))


# === MAIN ===

def main():
    parser = argparse.ArgumentParser(description='Optuna HPO for DiffATMGNN.')
    parser.add_argument('--model', type=str, required=True, choices=['DiffATMGNN'], help='Which model to optimise.')
    parser.add_argument('--n-trials', type=int, default=20, help='Number of Optuna trials to run.')
    parser.add_argument('--hpo-epochs', type=int, default=50, help='Training epochs per trial.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for the sampler and data splits.')
    args = parser.parse_args()

    # Random seed for Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\n' + '=' * 60)
    print('  Optuna HPO — {}'.format(args.model))
    print('=' * 60)
    print('  Device     : {}'.format(device))
    print('  Countries  : {}'.format(', '.join(COUNTRIES)))
    print('  Shift      : {} (HPO only)'.format(_SHIFT))
    print('  Epochs     : {} per trial'.format(args.hpo_epochs))
    print('  Trials     : {}'.format(args.n_trials))
    print('  Seed       : {}'.format(args.seed))
    print('=' * 60 + '\n')

    # Output paths 
    results_dir = os.path.join(_ROOT_DIR, 'results')
    figures_dir = os.path.join(_ROOT_DIR, 'figures', 'hpo')
    db_path     = os.path.join(results_dir, '{}_hpo.db'.format(args.model))
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Load all four country datasets once
    print('[HPO] Loading datasets...')
    dataset = read_datasets(_WINDOW, rand_weight=False)
    print('[HPO] Datasets loaded.\n')

    # Create or resume the Optuna study, if interrupted
    storage_url = 'sqlite:///{}'.format(db_path)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(
        study_name    = '{}_hpo'.format(args.model),
        direction     = 'minimize',
        storage       = storage_url,
        load_if_exists= True,
        pruner        = pruner,
        sampler       = sampler,
    )

    n_existing = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_existing > 0:
        print('[HPO] Resuming study — {} trial(s) already completed.\n'.format(
            n_existing))

    # Persist best params JSON after every trial
    # NOTE: Even a mid-run interrupt leaves a valid JSON on disk.
    def _after_trial_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            save_best_params(study, args.model, results_dir)

    # Run optimisation
    print('[HPO] Starting optimisation ({} trials × {} epochs)...\n'.format(
        args.n_trials, args.hpo_epochs))
    try:
        # Optuna objective function
        study.optimize(
            lambda trial: objective(trial, args.model, dataset, device, args.hpo_epochs),
            n_trials   = args.n_trials,
            callbacks  = [_after_trial_callback],
        )
    except KeyboardInterrupt:
        # Interrupt safety
        print('\n[HPO] Interrupted by user.')
        print('[HPO] All completed trials are saved in:\n      {}'.format(db_path))
        save_best_params(study, args.model, results_dir)
        print('[HPO] Generating plots from completed trials...')
        plot_study(study, args.model, figures_dir)
        print('[HPO] Exiting safely.')
        return

    # Final save + plots
    save_best_params(study, args.model, results_dir)

    print('\n[HPO] Generating plots...')
    plot_study(study, args.model, figures_dir)

    # Print command to run full training with best params
    try:
        p = study.best_params
        print('\n' + '=' * 60)
        print('  HPO complete.')
        print('  Best mean val loss : {:.6f}'.format(study.best_value))
        print('  Best params        : lr={lr:.6f}  hidden={hidden}  '
            'dropout={dropout:.4f}'.format(**p))
        print('=' * 60)
        print('\n  Run full training with these values:\n')
        _script = ('DiffATMGNN_training.py')
        print('  python src/{script} --lr {lr:.6f} --hidden {hidden}'
              ' --dropout {dropout:.4f}'.format(script=_script, **p))
        print()
    except ValueError:
        print('[HPO] No completed trials — no best params to report.')


if __name__ == '__main__':
    main()
