"""
    Ablation study for ATMGNN and DiffATMGNN.

    Models:
        1.  ARIMA                  - statistical baseline
        2.  BiLSTM                 - temporal deep-learning baseline
        3.  STAN  (MPNN_LSTM)      - spatio-temporal GNN baseline
        4.  ATMGNN     (no SEIR)   - ablates SEIR node features
        5.  DiffATMGNN (no SEIR)   - ablates SEIR in the probabilistic setting

    Outputs (ablation_results/):
        results_ablation_summary.csv  - one row per model, averaged over countries and shifts
        results_ablation_full.csv     - one row per (model, country, shift)
"""

# === IMPORTS ===

import os
import sys
import warnings
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
from torch_geometric.nn.conv import GCNConv

warnings.filterwarnings("ignore")

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_SRC_DIR, "..")
OUT_DIR   = os.path.join(_ROOT_DIR, "ablation_results")
PRED_DIR  = os.path.join(_ROOT_DIR, "predictions")
CKPT_DIR  = os.path.join(_ROOT_DIR, "ablation_checkpoints")  

# === CONFIGURATION ===

WINDOW          = 7
GRAPH_WINDOW    = 7
AHEAD           = 7       
START_EXP       = 15
SEP             = 10
EPOCHS          = 300     
EARLY_STOP      = 100     
LR              = 0.001
HIDDEN          = 64
DROPOUT         = 0.5
BATCH_SIZE      = 32
EDGE_DECAY      = 0.5
RAND_SEED       = 0
SEIR_LAMBDA     = 0.1     
DIFFUSION_STEPS = 50      
DECODER_HIDDEN  = 128     
COUNTRIES       = ["IT", "EN", "FR", "ES"]
COUNTRY_IDX     = {"IT": 0, "ES": 1, "EN": 2, "FR": 3}

# Hyperparameters for BiLSTM and STAN (no HPO was run for these baselines).
LR              = 0.001
HIDDEN          = 64
DROPOUT         = 0.5

# Hyperparameters for ATMGNN_noSEIR and DiffATMGNN_noSEIR, derived from the
# same HPO run used to train the full ATMGNN and DiffATMGNN models.
# Using identical HPs ensures RQ2 isolates the SEIR contribution only.
GNN_LR          = 0.0048
GNN_HIDDEN      = 128
GNN_DROPOUT     = 0.3       


# === UTILITY FUNCTIONS ===

def _strip_seir(features, window):
    """Removes the four fitted SEIR compartment columns from node feature matrices."""
    return [np.concatenate([H[:, :window], H[:, window + 4 : window + 5]], axis=1)
            for H in features]


def _node_weights(labels, device):
    """Builds per-node inverse-frequency loss weights."""
    mean_cases  = labels.values.astype(float).mean(axis=1)
    inv_weights = 1.0 / (np.log1p(mean_cases) + 1.0)
    inv_weights = inv_weights / inv_weights.mean()
    return torch.FloatTensor(inv_weights).to(device)


def _compute_metrics(y_pred, y_true, result=None):
    """
    Computes the full metric set matching the main training scripts:
        mean_result : mean of per-test-day MAE values
        std_result  : std  of per-test-day MAE values
        MAE, MSE, RMSE, R2 : aggregated over all predictions
    'result' is the list of per-test-day MAE values collected during the
    rolling loop.  Pass None when it is unavailable (e.g. loaded from files).
    """
    yp = np.asarray(y_pred).flatten()
    yt = np.asarray(y_true).flatten()
    mask = np.isfinite(yp) & np.isfinite(yt)
    yp, yt = yp[mask], yt[mask]
    nan6 = {'mean_result': float('nan'), 'std_result': float('nan'),
            'MAE': float('nan'), 'MSE': float('nan'),
            'RMSE': float('nan'), 'R2': float('nan')}
    if len(yp) == 0:
        return nan6
    return {
        'mean_result': float(np.mean(result)) if result else float('nan'),
        'std_result':  float(np.std(result))  if result else float('nan'),
        'MAE':  float(mean_absolute_error(yt, yp)),
        'MSE':  float(mean_squared_error(yt, yp)),
        'RMSE': float(np.sqrt(mean_squared_error(yt, yp))),
        'R2':   float(r2_score(yt, yp)),
    }


def _average_metrics(per_country_per_shift):
    """Averages metrics across all (country, shift) pairs."""
    buckets = {'mean_result': [], 'std_result': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'R2': []}
    for shift_map in per_country_per_shift.values():
        for m in shift_map.values():
            if np.isfinite(m['MAE']):
                for k in buckets:
                    if np.isfinite(m.get(k, float('nan'))):
                        buckets[k].append(m[k])
    return {k: float(np.mean(v)) if v else float('nan') for k, v in buckets.items()}


# === MODELS ===

class BiLSTM(nn.Module):
    """Bidirectional LSTM temporal baseline (no graph)."""

    def __init__(self, nfeat, nhid, n_nodes, dropout):
        super().__init__()
        self.n_nodes = n_nodes
        self.lstm = nn.LSTM(nfeat, nhid, num_layers=2,
                            bidirectional=True, dropout=dropout)
        self.fc   = nn.Linear(nhid * 2, 1)

    def forward(self, adj, x):
        # x: [batch * n_nodes, nfeat]; treat each row as an independent sequence.
        x   = x.unsqueeze(0)       # [1, batch*n_nodes, nfeat]
        out, _ = self.lstm(x)
        return self.fc(out[-1]).view(-1)


class MPNN_LSTM(nn.Module):
    """
        Message-Passing GCN + LSTM (STAN baseline)."""

    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout):
        super().__init__()
        self.window  = window
        self.n_nodes = n_nodes
        self.nhid    = nhid
        self.nfeat   = nfeat

        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid,  nhid)
        self.bn1   = nn.BatchNorm1d(nhid)
        self.bn2   = nn.BatchNorm1d(nhid)
        self.rnn1  = nn.LSTM(2 * nhid, nhid, 1)
        self.rnn2  = nn.LSTM(nhid, nhid, 1)
        self.fc1   = nn.Linear(2 * nhid + window * nfeat, nhid)
        self.fc2   = nn.Linear(nhid, nout)
        self.drop  = nn.Dropout(dropout)
        self.relu  = nn.ReLU()

    def forward(self, adj, x):
        weight  = adj.coalesce().values()
        adj_idx = adj.coalesce().indices()

        skip = (x.view(-1, self.window, self.n_nodes, self.nfeat)
                .transpose(1, 2)
                .reshape(-1, self.window, self.nfeat))

        x  = self.relu(self.conv1(x, adj_idx, edge_weight=weight))
        x  = self.bn1(x); x = self.drop(x); gx1 = x
        x  = self.relu(self.conv2(x, adj_idx, edge_weight=weight))
        x  = self.bn2(x); x = self.drop(x); gx2 = x

        gc = torch.cat([gx1, gx2], dim=1)
        gc = (gc.view(-1, self.window, self.n_nodes, gc.size(1))
                .transpose(0, 1).contiguous()
                .view(self.window, -1, gc.size(1)))

        out,  (hn1, _) = self.rnn1(gc)
        _,    (hn2, _) = self.rnn2(out)

        x    = torch.cat([hn1[0], hn2[0]], dim=1)
        skip = skip.reshape(skip.size(0), -1)
        x    = torch.cat([x, skip], dim=1)
        x    = self.relu(self.fc1(x)); x = self.drop(x)
        return self.relu(self.fc2(x)).squeeze().view(-1)


# === MULTIPROCESSOR ===

def _country_worker(packed_args):
    """Processes one (model_name, country) pair across all shifts."""

    import os, sys, warnings, random
    import numpy as np
    import torch, torch.nn.functional as F, torch.optim as optim
    from math import ceil
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    warnings.filterwarnings("ignore")

    # Re-add src/ to the path so utils and models are importable.
    _src = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _src)
    from utils import generate_new_batches, AverageMeter
    from models import ATMGNN, ATMGNN_Diff

    (model_name, country,
    gs_adj, features_full, y, labels,
    n_nodes,
    _WINDOW, _GRAPH_WINDOW, _AHEAD, _START_EXP, _SEP,
    _EPOCHS, _EARLY_STOP, _LR, _HIDDEN, _DROPOUT,
    _GNN_LR, _GNN_HIDDEN, _GNN_DROPOUT,
    _BATCH_SIZE, _EDGE_DECAY, _RAND_SEED, _SEIR_LAMBDA,
    _DIFFUSION_STEPS, _DECODER_HIDDEN,
    _CKPT_DIR) = packed_args

    # Device and seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(_RAND_SEED)
    np.random.seed(_RAND_SEED)
    random.seed(_RAND_SEED)

    # Feature preparation (model-specific)
    if model_name in ("ATMGNN_noSEIR", "DiffATMGNN_noSEIR"):
        # Strip SEIR columns; keep case history + growth rate.
        features = [np.concatenate([H[:, :_WINDOW],
                                    H[:, _WINDOW + 4 : _WINDOW + 5]], axis=1)
                    for H in features_full]
        nfeat = _WINDOW + 1
    else:
        # BiLSTM and STAN: case-count history only (no SEIR, no growth rate).
        features = [H[:, :_WINDOW] for H in features_full]
        nfeat = _WINDOW

    n_samples = len(gs_adj)

    # Node weights (matches both main training scripts) 
    mean_cases  = labels.values.astype(float).mean(axis=1)
    inv_weights = 1.0 / (np.log1p(mean_cases) + 1.0)
    inv_weights = inv_weights / inv_weights.mean()
    node_weights = torch.FloatTensor(inv_weights).to(device)

    # Select hyperparameters: HPO-derived values for ATMGNN family,
    # defaults for BiLSTM and STAN which had no HPO run.
    _use_gnn_hp = model_name in ("ATMGNN_noSEIR", "DiffATMGNN_noSEIR")
    _active_lr      = _GNN_LR      if _use_gnn_hp else _LR
    _active_hidden  = _GNN_HIDDEN  if _use_gnn_hp else _HIDDEN
    _active_dropout = _GNN_DROPOUT if _use_gnn_hp else _DROPOUT

    # Helpers local to this worker

    def _make_model():
        """Instantiates a fresh model according to model_name."""
        if model_name == "BiLSTM":
            return BiLSTM(nfeat=nfeat, nhid=_active_hidden,
                        n_nodes=n_nodes, dropout=_active_dropout).to(device)
        elif model_name == "STAN":
            return MPNN_LSTM(nfeat=nfeat, nhid=_active_hidden, nout=1,
                            n_nodes=n_nodes, window=_GRAPH_WINDOW,
                            dropout=_active_dropout).to(device)
        elif model_name == "ATMGNN_noSEIR":
            return ATMGNN(nfeat=nfeat, nhidden=_active_hidden, nout=1,
                        n_nodes=n_nodes, window=_GRAPH_WINDOW,
                        dropout=_active_dropout, nhead=1).to(device)
        elif model_name == "DiffATMGNN_noSEIR":
            return ATMGNN_Diff(nfeat=nfeat, nhidden=_active_hidden, nout=1,
                            n_nodes=n_nodes, window=_GRAPH_WINDOW,
                            dropout=_active_dropout, nhead=1,
                            diffusion_steps=_DIFFUSION_STEPS,
                            decoder_hidden=_DECODER_HIDDEN).to(device)
        raise ValueError("Unknown model: {}".format(model_name))

    def _train_step(model, optimizer, adj, feat, y_batch):
        optimizer.zero_grad()

        if model_name == "ATMGNN_noSEIR":
            output   = model(adj, feat)
            loss_mse = F.mse_loss(output, y_batch)
            # Non-negativity constraint from ATMGNN_training.py
            seir_penalty = F.relu(-output).mean()
            loss = loss_mse + _SEIR_LAMBDA * seir_penalty

        elif model_name == "DiffATMGNN_noSEIR":
            cond = model.encode(adj, feat)
            y_2d = y_batch.unsqueeze(-1) if y_batch.dim() == 1 else y_batch
            diff_loss = model.diffusion.compute_loss(y_2d, cond)

            direct = model.relu(model.fc1(cond))
            direct = model.dropout(direct)
            direct = model.relu(model.fc2(direct)).squeeze(-1).view(-1)
            if node_weights is not None:
                w = node_weights.repeat(direct.size(0) // node_weights.size(0))
                aux_loss = (w * (direct - y_batch.view(-1)) ** 2).mean()
            else:
                aux_loss = F.mse_loss(direct, y_batch.view(-1))

            loss   = diff_loss + 0.1 * aux_loss
            output = y_batch   # size placeholder for AverageMeter

        else:  # BiLSTM, STAN
            output = model(adj, feat)
            loss   = F.mse_loss(output, y_batch)

        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return output, loss

    def _val_loss(model, adj, feat, y_batch):
        with torch.no_grad():
            if model_name == "ATMGNN_noSEIR":
                output = model(adj, feat)
                w      = node_weights.repeat(output.size(0) // node_weights.size(0))
                return float((w * (output - y_batch) ** 2).mean().item()), output

            elif model_name == "DiffATMGNN_noSEIR":
                cond = model.encode(adj, feat)
                y_2d = y_batch.unsqueeze(-1) if y_batch.dim() == 1 else y_batch
                diff_loss = model.diffusion.compute_loss(y_2d, cond)
                direct = model.relu(model.fc1(cond))
                direct = model.dropout(direct)
                direct = model.relu(model.fc2(direct)).squeeze(-1).view(-1)
                if node_weights is not None:
                    w = node_weights.repeat(direct.size(0) // node_weights.size(0))
                    aux_loss = (w * (direct - y_batch.view(-1)) ** 2).mean()
                else:
                    aux_loss = F.mse_loss(direct, y_batch.view(-1))
                return float((diff_loss + 0.1 * aux_loss).item()), direct

            else:  # BiLSTM, STAN
                output = model(adj, feat)
                return float(F.mse_loss(output, y_batch).item()), output

    def _test_output(model, adj, feat):
        """Deterministic point forecast."""
        with torch.no_grad():
            return model(adj, feat)   

    # Graph window: BiLSTM uses gw=1 (no temporal graph stacking needed)
    _gw = 1 if model_name == "BiLSTM" else _GRAPH_WINDOW

    # Rolling-window loop
    shift_metrics = {}

    for shift in range(_AHEAD):
        print("  [{}][{}] Shift {}/{}".format(model_name, country, shift + 1, _AHEAD),
            flush=True)

        y_pred = np.empty((n_nodes, 0), dtype=float)
        y_true = np.empty((n_nodes, 0), dtype=float)
        result = []   # per-test-day MAE, matches main training scripts

        for test_sample in range(_START_EXP, n_samples - shift):

            # Data splitting
            idx_train = list(range(_WINDOW - 1, test_sample - _SEP))
            idx_val   = list(range(test_sample - _SEP, test_sample, 2))
            idx_train = idx_train + list(range(test_sample - _SEP + 1, test_sample, 2))

            if len(idx_train) < 2:
                continue

            # Augment with time-reversed samples for ATMGNN variants
            _augment = model_name in ("ATMGNN_noSEIR", "DiffATMGNN_noSEIR")

            adj_train, feat_train, y_train = generate_new_batches(
                gs_adj, features, y, idx_train, _gw, shift,
                _BATCH_SIZE, device, test_sample,
                decay=_EDGE_DECAY, augment_reverse=_augment)
            adj_val, feat_val, y_val = generate_new_batches(
                gs_adj, features, y, idx_val, _gw, shift,
                _BATCH_SIZE, device, test_sample, decay=_EDGE_DECAY)
            adj_test, feat_test, y_test = generate_new_batches(
                gs_adj, features, y, [test_sample], _gw, shift,
                _BATCH_SIZE, device, test_sample, decay=_EDGE_DECAY)

            n_batches = ceil((len(idx_train) * (2 if _augment else 1)) / _BATCH_SIZE)

            # Training with restart logic 
            best_val_acc   = float('inf')
            best_state_dict = None   
            max_restarts   = 3
            restart_count  = 0
            stop           = False

            while not stop:
                restart_count += 1
                if restart_count > max_restarts:
                    print("    [WARN] Max restarts exceeded for test_sample={}.".format(
                        test_sample), flush=True)
                    stop = True
                    break

                model     = _make_model()
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=_active_lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

                val_history = []
                stop        = False

                for epoch in range(_EPOCHS):
                    model.train()
                    train_loss_meter = AverageMeter()
                    for b in range(n_batches):
                        out, loss = _train_step(model, optimizer,
                                                adj_train[b], feat_train[b], y_train[b])
                        train_loss_meter.update(loss.item(), out.size(0))

                    model.eval()
                    vl, _ = _val_loss(model, adj_val[0], feat_val[0], y_val[0])

                    val_history.append(vl)

                    if not np.isfinite(vl) or vl > 1e12:
                        # Diverged — restart (matches both main scripts)
                        stop = False
                        break

                    if vl < best_val_acc:
                        best_val_acc    = vl
                        best_state_dict = {k: v.cpu().clone()
                                           for k, v in model.state_dict().items()}

                    # Early stop: stalled in first 30 epochs (matches main scripts)
                    if 10 < epoch < 30:
                        if len(set(round(v, 2) for v in val_history[-20:])) == 1:
                            stop = False
                            break

                    # Early stop: stalled for EARLY_STOP epochs (matches main scripts)
                    if epoch > _EARLY_STOP:
                        if len(set(round(v, 2) for v in val_history[-_EARLY_STOP:])) == 1:
                            break

                    stop = True
                    scheduler.step(vl)

            # Testing
            if best_state_dict is None:
                del adj_train, feat_train, y_train
                del adj_val,   feat_val,   y_val
                del adj_test,  feat_test,  y_test
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue

            model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
            best_state_dict = None   # free RAM immediately after restore
            model.eval()

            output = _test_output(model, adj_test[0], feat_test[0])

            o_log = output.cpu().detach().numpy()
            l_log = y_test[0].cpu().numpy()
            o     = np.expm1(np.clip(o_log, 0.0, 10.0))
            l     = np.expm1(l_log)

            # Per-test-day MAE (matches main training scripts: error = sum|o-l|/n_nodes)
            error = float(np.sum(abs(o - l)) / n_nodes)
            result.append(error)

            y_pred = np.append(y_pred, o.reshape(-1, 1), axis=1)
            y_true = np.append(y_true, l.reshape(-1, 1), axis=1)

            del adj_train, feat_train, y_train
            del adj_val,   feat_val,   y_val
            del adj_test,  feat_test,  y_test
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # Compute full metric set for this shift
        if y_pred.size > 0:
            yp, yt = y_pred.flatten(), y_true.flatten()
            mask   = np.isfinite(yp) & np.isfinite(yt)
            yp, yt = yp[mask], yt[mask]
            shift_metrics[shift] = {
                'mean_result': float(np.mean(result)),
                'std_result':  float(np.std(result)),
                'MAE':  float(mean_absolute_error(yt, yp)),
                'MSE':  float(mean_squared_error(yt, yp)),
                'RMSE': float(np.sqrt(mean_squared_error(yt, yp))),
                'R2':   float(r2_score(yt, yp)),
            }
        else:
            shift_metrics[shift] = {
                'mean_result': float('nan'), 'std_result': float('nan'),
                'MAE': float('nan'), 'MSE': float('nan'),
                'RMSE': float('nan'), 'R2': float('nan')}

        print("  [{}][{}] Shift {} -> MAE={:.4f} MSE={:.4f} RMSE={:.4f} R2={:.4f}".format(
            model_name, country, shift,
            shift_metrics[shift]['MAE'], shift_metrics[shift]['MSE'],
            shift_metrics[shift]['RMSE'], shift_metrics[shift]['R2']), flush=True)

    return shift_metrics


# ARIMA (CPU-only, run sequentially)

def _run_arima_all_countries(meta_labs, meta_graphs, full_results):
    """Runs ARIMA for all countries"""
    results = {}
    for country in COUNTRIES:
        idx       = COUNTRY_IDX[country]
        labels    = meta_labs[idx]
        n_samples = len(meta_graphs[idx])
        n_regions = labels.shape[0]
        print("\n  [ARIMA][{}]  ({} regions, {} test days)".format(
            country, n_regions, n_samples - START_EXP), flush=True)

        if _is_done(full_results, "ARIMA", country):
            print("    [SKIP] Already complete.", flush=True)
            results[country] = {s: full_results[("ARIMA", country, s)] for s in range(AHEAD)}
            continue

        # Accumulators indexed by shift.
        preds  = {s: [] for s in range(AHEAD)}   # flat predicted values
        truths = {s: [] for s in range(AHEAD)}   # flat ground-truth values
        daily  = {s: [] for s in range(AHEAD)}   # per-test-day mean abs error

        for test_sample in range(START_EXP, n_samples):
            day_preds = {s: [] for s in range(AHEAD)}
            day_truth = {s: [] for s in range(AHEAD)}

            for j in range(n_regions):
                series = labels.iloc[j, :test_sample].values.astype(float)

                if series.sum() == 0:
                    yhats = [0.0] * AHEAD
                else:
                    try:
                        fit   = ARIMAModel(series, order=(2, 0, 2)).fit()
                        yhats = [float(abs(v)) for v in fit.forecast(steps=AHEAD)]
                    except Exception:
                        try:
                            fit   = ARIMAModel(series, order=(1, 0, 0)).fit()
                            yhats = [float(abs(v)) for v in fit.forecast(steps=AHEAD)]
                        except Exception:
                            yhats = [float(np.mean(series[-WINDOW:]))] * AHEAD

                for s in range(AHEAD):
                    target_idx = test_sample + s
                    if target_idx >= n_samples:
                        continue
                    target = float(labels.iloc[j, target_idx])
                    day_preds[s].append(max(yhats[s], 0.0))
                    day_truth[s].append(max(target,   0.0))

            for s in range(AHEAD):
                if day_preds[s]:
                    preds[s].extend(day_preds[s])
                    truths[s].extend(day_truth[s])
                    daily[s].append(float(np.mean(
                        abs(np.array(day_preds[s]) - np.array(day_truth[s])))))

            if (test_sample - START_EXP + 1) % 10 == 0:
                print("    test_sample {}/{}".format(
                    test_sample - START_EXP + 1, n_samples - START_EXP), flush=True)

        shift_metrics = {}
        for s in range(AHEAD):
            shift_metrics[s] = _compute_metrics(
                np.array(preds[s]), np.array(truths[s]), result=daily[s])
            print("    Shift {} -> MAE={:.4f} MSE={:.4f} RMSE={:.4f} R2={:.4f}".format(
                s, shift_metrics[s]['MAE'], shift_metrics[s]['MSE'],
                shift_metrics[s]['RMSE'], shift_metrics[s]['R2']), flush=True)

        results[country] = shift_metrics
    return results


# === LOAD RESULTS ===

def _load_saved_predictions(model_prefix):
    """Loads pre-computed predictions from predictions/ and recomputes metrics."""
    results = {}
    for country in COUNTRIES:
        shift_metrics = {}
        for shift in range(AHEAD):
            pred_path  = os.path.join(
                PRED_DIR, "predict_{}_shift{}_{}.csv".format(model_prefix, shift, country))
            truth_path = os.path.join(
                PRED_DIR, "truth_{}_shift{}_{}.csv".format(model_prefix, shift, country))

            if not os.path.exists(pred_path) or not os.path.exists(truth_path):
                print("    [WARN] Missing: {} shift={} {}".format(
                    model_prefix, shift, country), flush=True)
                shift_metrics[shift] = {
                    'mean_result': float('nan'), 'std_result': float('nan'),
                    'MAE': float('nan'), 'MSE': float('nan'),
                    'RMSE': float('nan'), 'R2': float('nan')}
                continue

            y_pred = np.loadtxt(pred_path,  delimiter=',')
            y_true = np.loadtxt(truth_path, delimiter=',')

            shift_metrics[shift] = _compute_metrics(y_pred.flatten(), y_true.flatten())
            print("  [{}][{}] Shift {} -> MAE={:.4f} MSE={:.4f} RMSE={:.4f} R2={:.4f}".format(
                model_prefix, country, shift,
                shift_metrics[shift]['MAE'], shift_metrics[shift]['MSE'],
                shift_metrics[shift]['RMSE'], shift_metrics[shift]['R2']), flush=True)

        results[country] = shift_metrics
    return results


# === OUTPUT ===

_FULL_CSV    = os.path.join(OUT_DIR, "results_ablation_full.csv")
_CSV_HEADER  = "model,country,shift,mean_result,std_result,MAE,MSE,RMSE,R2\n"


def _load_checkpoint_csv():
    existing_full = {}
    existing_all  = {}
    if not os.path.exists(_FULL_CSV):
        return existing_full, existing_all
    with open(_FULL_CSV, 'r') as f:
        f.readline()   # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
            model, country, shift = parts[0], parts[1], int(parts[2])
            m = {
                'mean_result': float(parts[3]), 'std_result': float(parts[4]),
                'MAE':  float(parts[5]), 'MSE':  float(parts[6]),
                'RMSE': float(parts[7]), 'R2':   float(parts[8]),
            }
            existing_full[(model, country, shift)] = m
            existing_all.setdefault(model, {}).setdefault(country, {})[shift] = m
    n = len(existing_full)
    if n > 0:
        print("[RESUME] Loaded {} previously computed rows.".format(n), flush=True)
    return existing_full, existing_all


def _is_done(full_results, model_name, country):
    """Returns True if all AHEAD shifts for (model, country) are already in full_results."""
    return all((model_name, country, s) in full_results for s in range(AHEAD))


def _append_to_csv(model_name, country, shift_metrics):
    """Appends one completed (model, country) block to the full CSV immediately."""
    write_header = not os.path.exists(_FULL_CSV)
    with open(_FULL_CSV, 'a') as f:
        if write_header:
            f.write(_CSV_HEADER)
        for shift, m in shift_metrics.items():
            f.write("{},{},{},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n".format(
                model_name, country, shift,
                m['mean_result'], m['std_result'],
                m['MAE'], m['MSE'], m['RMSE'], m['R2']))


def _print_table(summary):
    header = "{:<35} {:>10} {:>10} {:>10} {:>10}".format("Model", "MAE", "MSE", "RMSE", "R2")
    print("\n" + "=" * 80)
    print("  ABLATION RESULTS  (averaged across all countries and shifts)")
    print("=" * 80)
    print(header)
    print("-" * 80)
    for name, m in summary.items():
        print("{:<35} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            name, m['MAE'], m['MSE'], m['RMSE'], m['R2']))
    print("=" * 80 + "\n")


def _save_results(summary, full_results):
    os.makedirs(OUT_DIR, exist_ok=True)

    # Summary CSV construction
    with open(os.path.join(OUT_DIR, "results_ablation_summary.csv"), "w") as f:
        f.write("model,mean_result,std_result,MAE,MSE,RMSE,R2\n")
        for name, m in summary.items():
            f.write("{},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n".format(
                name, m['mean_result'], m['std_result'],
                m['MAE'], m['MSE'], m['RMSE'], m['R2']))

    # Full CSV is written incrementally via _append_to_csv — do not overwrite here.
    print("[SAVE] Summary written to {}/".format(OUT_DIR))


# === MAIN ===

if __name__ == '__main__':

    import random
    torch.manual_seed(RAND_SEED)
    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  ABLATION STUDY")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("  Device      : {}".format(device))
    print("  Countries   : {}".format(", ".join(COUNTRIES)))
    print("  Shifts      : 0 to {}".format(AHEAD - 1))
    print("  Epochs      : {} (early stop after {})".format(EPOCHS, EARLY_STOP))
    print("=" * 70 + "\n")

    print("[SETUP] Loading datasets...")
    sys.path.insert(0, _SRC_DIR)
    from utils import read_datasets
    meta_labs, meta_graphs, meta_features, meta_y = read_datasets(WINDOW, rand_weight=False)
    print("[SETUP] Done.\n")

    # Prepackage per-country data for the worker
    country_data = {
        c: (meta_graphs[COUNTRY_IDX[c]],
            meta_features[COUNTRY_IDX[c]],
            meta_y[COUNTRY_IDX[c]],
            meta_labs[COUNTRY_IDX[c]],
            meta_graphs[COUNTRY_IDX[c]][0].shape[0])            # n_nodes
        for c in COUNTRIES
    }

    # Shared config tuple passed to every worker.
    _cfg = (WINDOW, GRAPH_WINDOW, AHEAD, START_EXP, SEP,
            EPOCHS, EARLY_STOP, LR, HIDDEN, DROPOUT,
            GNN_LR, GNN_HIDDEN, GNN_DROPOUT,
            BATCH_SIZE, EDGE_DECAY, RAND_SEED, SEIR_LAMBDA,
            DIFFUSION_STEPS, DECODER_HIDDEN, CKPT_DIR)

    # Load any results saved before the last interruption.
    full_results, all_results = _load_checkpoint_csv()

    # Model 1: ARIMA (CPU, sequential)
    print("\n" + "-" * 70)
    print("  MODEL: ARIMA")
    print("-" * 70)
    all_results["ARIMA"] = _run_arima_all_countries(meta_labs, meta_graphs, full_results)
    for country, sm in all_results["ARIMA"].items():
        for shift, m in sm.items():
            full_results[("ARIMA", country, shift)] = m

    # Models 2-5: GNN-based models (sequential)
    GNN_MODELS = ["BiLSTM", "STAN", "ATMGNN_noSEIR", "DiffATMGNN_noSEIR"]

    for model_name in GNN_MODELS:
        print("\n" + "-" * 70)
        print("  MODEL: {}".format(model_name))
        print("-" * 70)

        all_results.setdefault(model_name, {})
        for country in COUNTRIES:
            if _is_done(full_results, model_name, country):
                print("  [SKIP] {} {} already complete.".format(model_name, country), flush=True)
                all_results[model_name][country] = {
                    s: full_results[(model_name, country, s)] for s in range(AHEAD)}
                continue
            packed_args = (model_name, country) + country_data[country] + _cfg
            shift_metrics = _country_worker(packed_args)
            all_results[model_name][country] = shift_metrics
            for shift, m in shift_metrics.items():
                full_results[(model_name, country, shift)] = m
            _append_to_csv(model_name, country, shift_metrics)

    # Models 6 & 7: Load existing predictions (no retraining)               
    for model_name, prefix in [("ATMGNN_withSEIR",     "ATMGNN"),
                                ("DiffATMGNN_withSEIR", "ATMGNN_Diff")]:
        print("\n" + "-" * 70)
        print("  MODEL: {} (loading from predictions/)".format(model_name))
        print("-" * 70)
        all_results.setdefault(model_name, {})
        if not _is_done(full_results, model_name, list(COUNTRIES)[-1]):
            loaded = _load_saved_predictions(prefix)
            for country, sm in loaded.items():
                all_results[model_name][country] = sm
                if not _is_done(full_results, model_name, country):
                    for shift, m in sm.items():
                        full_results[(model_name, country, shift)] = m
                    _append_to_csv(model_name, country, sm)
                else:
                    for shift, m in sm.items():
                        full_results.setdefault((model_name, country, shift), m)
        else:
            for country in COUNTRIES:
                all_results[model_name][country] = {
                    s: full_results[(model_name, country, s)] for s in range(AHEAD)}

    # Aggregate and report                                                 
    ORDERED = [
        "ARIMA", "BiLSTM", "STAN",
        "ATMGNN_noSEIR", "DiffATMGNN_noSEIR",
        "ATMGNN_withSEIR", "DiffATMGNN_withSEIR",
    ]

    summary = {name: _average_metrics(all_results[name])
            for name in ORDERED if name in all_results}

    _print_table(summary)
    _save_results(summary, full_results)
