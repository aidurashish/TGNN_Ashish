"""
Optuna hyperparameter optimisation for ATMGNN and ATMGNN_Diff for three hyperparameters (lr, hidden, dropout).
All completed trials are persisted to a SQLite database.
"""

# === IMPORTS ===

import os
import sys
import json
import argparse
import random
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import optuna
import optuna.visualization
from optuna.pruners import MedianPruner

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)

from utils import generate_new_batches, read_datasets
from models import ATMGNN, ATMGNN_Diff

# === CONSTANTS ===

SEIR_LAMBDA = 0.1          # Disease biology penalty weight 
COUNTRIES   = ["IT", "EN", "FR", "ES"]
COUNTRY_IDX = {"IT": 0, "ES": 1, "EN": 2, "FR": 3}

# Fixed non-tuned settings used during HPO trials.
_SHIFT        = 0
_WINDOW       = 7
_GRAPH_WINDOW = 7
_SEP          = 10
_BATCH_SIZE   = 32
_EDGE_DECAY   = 0.5
_START_EXP    = 15


# === TRAINING HELPERS ===

def _train_step_atmgnn(model, optimizer, adj, features, y, node_weights):
    """One forward + backward pass for ATMGNN (MSE + SEIR non-negativity penalty)."""
    optimizer.zero_grad()
    output = model(adj, features)
    w      = node_weights.repeat(output.size(0) // node_weights.size(0))
    loss   = (w * (output - y) ** 2).mean() + SEIR_LAMBDA * F.relu(-output).mean()
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def _train_step_diff(model, optimizer, adj, features, y, node_weights):
    """One forward + backward pass for ATMGNN_Diff (diffusion denoising loss)."""
    optimizer.zero_grad()
    loss = model.compute_diffusion_loss(adj, features, y, node_weights=node_weights)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def _val_loss(model, adj, features, y, node_weights):
    """Weighted MSE validation loss (same for both model types - uses FC head)."""
    with torch.no_grad():
        output = model(adj, features)
        w      = node_weights.repeat(output.size(0) // node_weights.size(0))
        return (w * (output - y) ** 2).mean().item()


# === MAIN ===

def objective(trial, model_name, dataset, device, hpo_epochs):
    """
        Optuna objective function.

        Suggests three hyperparameters, builds a model per country, trains for
        hpo_epochs epochs using a fixed mid-range split, and returns the mean
        best validation loss across all four countries.

        Calls trial.report() after every epoch so MedianPruner can kill bad trials early.
    """
    lr      = trial.suggest_float("lr",      1e-4, 1e-2, log=True)
    hidden  = trial.suggest_categorical("hidden",  [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.2,  0.6)

    meta_labs, meta_graphs, meta_features, meta_y = dataset

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

        # Per-node inverse-frequency weights 
        mean_cases   = labels.values.astype(float).mean(axis=1)
        inv_w        = 1.0 / (np.log1p(mean_cases) + 1.0)
        inv_w        = inv_w / inv_w.mean()
        node_weights = torch.FloatTensor(inv_w).to(device)

        # Fixed mid-range split: avoids running the full rolling window.
        test_sample = _START_EXP + (n_samp - _START_EXP) // 2

        idx_train = list(range(_WINDOW - 1, test_sample - _SEP))
        idx_val   = list(range(test_sample - _SEP, test_sample, 2))
        idx_train = idx_train + list(range(test_sample - _SEP + 1, test_sample, 2))

        if len(idx_train) == 0 or len(idx_val) == 0:
            # Degenerate split, i.e., skip this country and treat as neutral loss.
            contexts.append(None)
            continue

        adj_train, feat_train, y_train = generate_new_batches(
            gs_adj, feats, y, idx_train, _GRAPH_WINDOW, _SHIFT,
            _BATCH_SIZE, device, test_sample,
            decay=_EDGE_DECAY, augment_reverse=True)
        adj_val, feat_val, y_val = generate_new_batches(
            gs_adj, feats, y, idx_val, _GRAPH_WINDOW, _SHIFT,
            _BATCH_SIZE, device, test_sample, decay=_EDGE_DECAY)

        n_batches = ceil(len(idx_train) * 2 / _BATCH_SIZE)

        # Build model
        if model_name == "ATMGNN":
            model = ATMGNN(
                nfeat=nfeat, nhidden=hidden, nout=1, n_nodes=n_nodes,
                window=_GRAPH_WINDOW, dropout=dropout, nhead=1,
            ).to(device)

        else:  # DiffATMGNN
            model = ATMGNN_Diff(
                nfeat=nfeat, nhidden=hidden, nout=1, n_nodes=n_nodes,
                window=_GRAPH_WINDOW, dropout=dropout, nhead=1,
            ).to(device)

            # Warm-start encoder weights from an existing ATMGNN checkpoint.
            _ckpt_path = os.path.join(
                _ROOT_DIR, 'checkpoints',
                'model_best_ATMGNN_shift0_{}_RW_False_seed0_AG.pth.tar'.format(country))
            if os.path.exists(_ckpt_path):
                _src_state = torch.load(_ckpt_path, map_location=device,
                                        weights_only=False)['state_dict']
                _dst_state = model.state_dict()
                # Copy all ATMGNN keys that exist in the Diff model (skip diffusion decoder).
                _dst_state.update({
                    k: v for k, v in _src_state.items()
                    if k in _dst_state and not k.startswith('diffusion.')
                })
                model.load_state_dict(_dst_state)

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
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

    # Select the correct per-step training function
    train_fn = _train_step_atmgnn if model_name == "ATMGNN" else _train_step_diff

    # Epoch loop: train all countries one epoch at a time
    for epoch in range(hpo_epochs):
        epoch_val_losses = []

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
            ctx['model'].eval()
            vl = _val_loss(
                ctx['model'],
                ctx['adj_val'][0], ctx['feat_val'][0], ctx['y_val'][0],
                ctx['node_weights'],
            )
            if not np.isfinite(vl):
                vl = 1e9   # Treat diverged trials as very bad but not fatal.

            ctx['best_val'] = min(ctx['best_val'], vl)
            ctx['scheduler'].step(vl)
            epoch_val_losses.append(vl)

        mean_val = float(np.mean(epoch_val_losses))

        # Report intermediate value so MedianPruner can stop bad trials early.
        trial.report(mean_val, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Return the mean of each country's best validation loss during the trial.
    return float(np.mean([ctx['best_val'] for ctx in active]))


# === SAVE BEST PARAMS ===

def save_best_params(study, model_name, results_dir):
    """
    Saves the best hyperparameters from completed trials to a JSON file.
    Called after every completed trial (via callback) and at the end of the run.
    Safe to call even if no trials have completed yet.
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

    print("[HPO] Best params saved → {}".format(path))
    print("      lr={lr:.6f}  hidden={hidden}  dropout={dropout:.4f}".format(
        **best_params))


# === PLOTS ===

def plot_study(study, model_name, figures_dir):
    """
    Generates and saves three interactive HTML plots using optuna.visualization:

        1. Optimisation history  — trial values and the running best.
        2. Parameter importance  — which of the three hyperparameters mattered most.
        3. Parallel coordinates  — a bird's-eye view of every trial.

    All plots are saved as interactive HTML to figures/hpo/.
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
    parser = argparse.ArgumentParser(
        description='Optuna HPO for ATMGNN / DiffATMGNN.')
    parser.add_argument('--model', type=str, required=True, choices=['ATMGNN', 'DiffATMGNN'], help='Which model to optimise.')
    parser.add_argument('--n-trials', type=int, default=20, help='Number of Optuna trials to run (default: 20).')
    parser.add_argument('--hpo-epochs', type=int, default=50, help='Training epochs per trial (default: 50).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for the sampler and data splits (default: 42).')
    args = parser.parse_args()

    # Reproducibility
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

    # Output paths (all absolute so chdir in read_datasets doesn't break them)
    results_dir = os.path.join(_ROOT_DIR, 'results')
    figures_dir = os.path.join(_ROOT_DIR, 'figures', 'hpo')
    db_path     = os.path.join(results_dir, '{}_hpo.db'.format(args.model))
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Load all four country datasets once
    print('[HPO] Loading datasets...')
    dataset = read_datasets(_WINDOW, rand_weight=False)
    print('[HPO] Datasets loaded.\n')

    # Create or resume the Optuna study
    # load_if_exists=True => the study resumes from the DB if previously
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

    n_existing = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE])
    if n_existing > 0:
        print('[HPO] Resuming study — {} trial(s) already completed.\n'.format(
            n_existing))

    # After-trial callback: persist best params JSON after every trial
    # This means even a mid-run interrupt leaves a valid JSON on disk.
    def _after_trial_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            save_best_params(study, args.model, results_dir)

    # Run optimisation
    print('[HPO] Starting optimisation ({} trials × {} epochs)...\n'.format(
        args.n_trials, args.hpo_epochs))
    try:
        study.optimize(
            lambda trial: objective(
                trial, args.model, dataset, device, args.hpo_epochs),
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
        _script = ('ATMGNN_training.py' if args.model == 'ATMGNN'
                else 'ATMGNN_Diff_training.py')
        print('  python src/{script} --lr {lr:.6f} --hidden {hidden}'
              ' --dropout {dropout:.4f}'.format(script=_script, **p))
        print()
    except ValueError:
        print('[HPO] No completed trials — no best params to report.')


if __name__ == '__main__':
    main()
