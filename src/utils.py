"""
    Core functions for loading, preprocessing, and batching data used to train the DiffATMGNN model.
"""

# === IMPORTS === 

import torch    # ore deep-learning framework
import networkx as nx   # Create and analyse graphs and networks
import numpy as np  # Numerical operations
import scipy.sparse as sp   # For storing large sparse matrices
import pandas as pd # Read and process CSV files
from datetime import date, timedelta    # Handle calendar dates and time differences
import os   # Handle files and directories
from scipy.integrate import odeint  # For solving ordinary differential equations (for SEIR)
from scipy.optimize import minimize # To find parameter values that minimize the SEIR mathematical objective function.

# === FUNCTIONS === 

# Fixed COVID-19 mean incubation period (days) which is held constant to reduce fitting complexity.
# σ = 1.0 / Mean incuabtion period (days)
COVID_SIGMA = 1.0 / 5.1 # Value obtained from study published in Annals of Internal Medicine (Lauer et al., 2020) [https://doi.org/10.7326/M20-0504]

def _seir_odes(compartments, t, beta, sigma, gamma):
    """
        Defines the SEIR ODE system, where the population is split into four compartments.

        ARGS:
            compartments (list): Current [S, E, I, R] state values.
            t            (float): Current time (unused directly, but required by odeint).
            beta         (float): Transmission rate (fitted).
            sigma        (float): Incubation rate = 1 / mean_incubation_days (fixed).
            gamma        (float): Recovery rate (fitted).

        RETURNS:
            list: Derivatives [dS/dt, dE/dt, dI/dt, dR/dt].
    """
    S, E, I, R = compartments
    N = S + E + I + R   # equals 1.0
    dS = -beta * S * I / N
    dE =  beta * S * I / N - sigma * E
    dI =  sigma * E - gamma * I
    dR =  gamma * I
    return [dS, dE, dI, dR]


def _fit_seir_to_region(case_counts):
    """
        Fits an SEIR model to a single region's observed case curve using
        Nelder-Mead optimisation, then returns the smoothed compartment trajectory.

        ARGS:
            case_counts (array): Observed daily case counts for one region, ordered by date.

        RETURNS:
            np.ndarray: Fitted trajectory of shape: [n_days, 4], columns = [S, E, I, R], expressed as proportions of the inferred total population (N=1).
    """
    case_arr = np.array(case_counts, dtype=float).clip(min=0)   # Convert input case array to float values for consistency and turn negative case counts to zero (if any).
    n = len(case_arr)   # Number of days in observation period

    # Normalisation
    total = max(case_arr.sum(), 1.0)    # Total observed cases
    i_obs = (case_arr / total).clip(0, 1)

    # Set initial compartment fractions from the first observed data point.
    I0  = float(i_obs[0])   # First normalzied case value
    E0  = min(I0 * 2.0, 1.0)   # Exposed (E) pool is roughly twice the infected pool early in an outbreak
    R0  = 0.0   # Because nobody recovers early in the pandemic
    S0  = max(1.0 - I0 - E0 - R0, 0.0)  # Assign remaining population to Susceptible pool
    y0  = [S0, E0, I0, R0]  # Initial SEIR state

    t_span = np.arange(n, dtype=float)  # Creates one time value per observation.

    # Residual (error) function to be minimized by the optimizer
    def _residuals(params):
        beta, gamma = params    # Transmission and Recovery rates
        if beta <= 0 or gamma <= 0: # Reject invalid negative or zero rates
            return 1e6  # A very large value to tell the optimizer that the value is unacceptable
        try:
            # Compare simulated curve with observed curve
            sol = odeint(_seir_odes, y0, t_span, args=(beta, COVID_SIGMA, gamma))   # Shape: [n_days, 4_states]
            return float(np.mean((sol[:, 2] - i_obs) ** 2))
        except Exception:
            return 1e6

    # Optimise starting from typical COVID-19 parameter estimates.
    result = minimize(
        _residuals,
        x0=[0.3, 0.1],  # Begin with beta = 0.3 and gamma = 0.1
        method='Nelder-Mead',
        options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000},
    )
    # OPTIONS: 
    # (1) xatol => Stop when parameter changes are extremely small like 1e-4 
    # (2) fatol => Stop when error changes are extremely small like 1e-4
    # (3) maxiter => Allow 2000 iterations at the most

    # Extract optimized parameters
    # max() ensures values are slightly positive
    beta_fit  = max(result.x[0], 1e-6)
    gamma_fit = max(result.x[1], 1e-6)

    # Run SEIR one last time with optimized/fitted parameters
    trajectory = odeint(_seir_odes, y0, t_span, args=(beta_fit, COVID_SIGMA, gamma_fit))
    return np.clip(trajectory, 0.0, 1.0)    # Force output to range [0,1]


def _backfill_late_starters(labels):
    """
        For regions whose first nonzero report arrives after day 0, 
        this function backfills the silent leading period using an exponential decay estimated from the regions that were already reporting during the same window.
    """
    fixed = labels.copy().astype(float) # Copy and convert input values to float (prevent changes to original dataset)
    vals  = fixed.values.copy()

    # Mean daily log-growth rate of regions that reported from day 0.
    early_rates = []    # Collect rates of early-reporting regions 
    # Per-region loop
    for row in vals:
        nz = np.where(row > 0)[0]   # Find the positions of positive (non-zero) case values.
        
        # Find consecutive non-zero positive case reports from day 0 
        if len(nz) > 0 and nz[0] == 0 and nz[-1] > 0:
            log_seg = np.log1p(row[:nz[-1] + 1])    # Take logarithm [log(1+x)] of reporting period
            # NOTE: We take log1p because log(0) is unidentified
            if len(log_seg) > 1:
                early_rates.append(np.mean(np.diff(log_seg)))
                
    # If no suitable early-reporting regions exist, return the data unchanged.
    if not early_rates:
        return fixed  
    mean_growth = np.mean(early_rates) # Estimates the average daily log-growth rate.

    # Process each late-starting region
    for i, region in enumerate(fixed.index):
        row = vals[i].copy()
        nz  = np.where(row > 0)[0]
        
        # Skip regions without late-start
        if len(nz) == 0 or nz[0] == 0:
            continue  # already starts from day 0
        first_nz = nz[0]    # Store first reporting day of late-starting region
        
        # Backfilling: move backward from the day before the first positive report to day 0.
        for day in range(first_nz - 1, -1, -1):
            row[day] = max(1.0, np.round(row[day + 1] * np.exp(-mean_growth)))
        fixed.iloc[i] = row

    return fixed    # Return corrected dataset


def _interpolate_reporting_gaps(labels):
    """
        Linearly interpolates interior zero values in each region's time series.
    """
    fixed = labels.copy().astype(float) # Create a copy and convert to float values 
    
    # Per-region loop 
    for region in fixed.index:
        row = fixed.loc[region].values.copy()   # Get region's case values
        nonzero_idx = np.where(row > 0)[0]  # Find days with positive (non-zero) reports
        
        # Skip regions with fewer than two positive values, as interpolation needs a starting and ending known value.
        if len(nonzero_idx) < 2:
            continue
        
        first_nz, last_nz = nonzero_idx[0], nonzero_idx[-1] # Finds the first and last positive-reporting days.
        x = np.arange(len(row)) # Create array of day positions
        interior_zeros = (row == 0) & (x > first_nz) & (x < last_nz)    # Select zeros between the first and last positive values.. 
        if not interior_zeros.any():
            continue
        row_interp = np.interp(x, x[row > 0], row[row > 0]) # Estimates values between known positive points using straight lines by one-dimensional linear interpolation
        row[interior_zeros] = np.round(row_interp[interior_zeros])  # Rounds the estimated values to whole case counts.
        fixed.loc[region] = np.maximum(row, 0)  # Ensure no value is negative
    return fixed    # Return corrected dataset


def _smooth_batch_reporting(labels, window=3):
    """
        Applies a rolling median filter to smooth batch-reporting spikes.
    """
    fixed = labels.copy().astype(float) # Create a copy and convert to float values 
    
    # Per-region loop
    for region in fixed.index:
        row = pd.Series(fixed.loc[region].values.copy())    # Create a pandas time series out of region's values
        smoothed = row.rolling(window, center=True, min_periods=1).median() # Aply rolling-median
        fixed.loc[region] = np.maximum(np.round(smoothed.values), 0)    # Prevent negative values and round up values
    return fixed    # Return corrected dataset


def read_datasets(window, rand_weight=False):
    """
        Loads graph and label data for all four countries in the dataset.

        ARGS:
            window      (int): Number of past days to include as node features.
            rand_weight (bool): Whether to replace real edge weights with uniform weights of 1 (FOR TESTING).

        RETURNS:
            final_labels     (list): DataFrames of daily case counts per region per country.
            final_graphs     (list): Lists of adjacency matrices (one per day) per country.
            final_features   (list): Lists of node feature matrices (one per day) per country.
            final_targets    (list): Lists of target case counts (one list per day) per country.
    """
    
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data"))
    final_labels = []
    final_graphs = []
    final_features = []
    final_targets = []

    # Italy
    os.chdir("Italy [COVID-19]")
    labels = pd.read_csv("italy_labels.csv")
    labels = labels.set_index("name")   # rows = regions, columns = dates

    # Builds a list of date strings
    sdate = date(2020, 3, 13)
    edate = date(2020, 5, 12)
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    
    
    # Build one graph per day, keeping only regions and dates present in both the graph and labels.
    Gs =generate_graphs(dates,"IT",rand_weight) 
    labels = labels.loc[list(Gs[0].nodes()),:]
    labels = labels.loc[:,dates]
    labels = _smooth_batch_reporting(labels)
    
    final_labels.append(labels)
    
    # Convert each NetworkX graph to a plain numpy adjacency matrix.
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    final_graphs.append(gs_adj)

    # Build a feature matrix per day using the past 'window' days of case counts.
    features = generate_new_features(Gs ,labels ,dates ,window )

    final_features.append(features)

    # Collect the ground-truth case count for every region on every day.
    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])

    final_targets.append(y)

    # Spain
    os.chdir("../Spain [COVID-19]")
    labels = pd.read_csv("spain_labels.csv")
    labels = labels.set_index("name")

    # Build a list of date strings
    sdate = date(2020, 3, 13)
    edate = date(2020, 5, 12)
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    
    Gs =generate_graphs(dates,"ES",rand_weight)
    labels = labels.loc[list(Gs[0].nodes()),:]
    labels = labels.loc[:,dates]    #labels.sum(1).values>10
    labels = _backfill_late_starters(labels)
    labels = _interpolate_reporting_gaps(labels)
    labels = _smooth_batch_reporting(labels)

    final_labels.append(labels)

    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    final_graphs.append(gs_adj)

    features = generate_new_features(Gs ,labels ,dates ,window )

    final_features.append(features)

    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])

    final_targets.append(y)

    # England
    os.chdir("../England [COVID-19]")
    labels = pd.read_csv("england_labels.csv")
    labels = labels.set_index("name")

    # Build a list of date strings
    sdate = date(2020, 3, 13)
    edate = date(2020, 5, 12)
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]

    Gs =generate_graphs(dates,"EN",rand_weight)
    
    labels = labels.loc[list(Gs[0].nodes()),:]
    labels = labels.loc[:,dates]
    labels = _smooth_batch_reporting(labels)
    
    final_labels.append(labels)

    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]
    final_graphs.append(gs_adj)

    features = generate_new_features(Gs ,labels ,dates ,window)
    final_features.append(features)

    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])
    final_targets.append(y)

    # France
    os.chdir("../France [COVID-19]")
    labels = pd.read_csv("france_labels.csv")
    labels = labels.set_index("name")

    # Build a list of date strings
    sdate = date(2020, 3, 13)
    edate = date(2020, 5, 12)
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]

    Gs =generate_graphs(dates,"FR",rand_weight)
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    labels = labels.loc[list(Gs[0].nodes()),:]
    labels = labels.loc[:,dates]
    labels = _backfill_late_starters(labels)
    labels = _interpolate_reporting_gaps(labels)
    labels = _smooth_batch_reporting(labels)

    final_labels.append(labels)

    final_graphs.append(gs_adj)

    features = generate_new_features(Gs ,labels ,dates ,window)

    final_features.append(features)

    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])

    final_targets.append(y)
    
    # Return to the source code directory after reading all data files.
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__))))

    return final_labels, final_graphs, final_features, final_targets
    
    
def generate_graphs(dates, country, rand_weight=False):
    """
        Builds a directed graph for each date by reading the corresponding mobility CSV file.

        ARGS:
            dates       (list[str]): List of date strings (YYYY-MM-DD) to load graphs for.
            country     (str): Country code prefix used in filenames such as 'IT', 'FR', etc.
            rand_weight (bool): Whether every edge is added with weight 1 instead of real flows (FOR TESTING).

        RETURNS:
            list[nx.DiGraph]: One NetworkX directed graph per date, with edge weights as mobility flows.
    """
    
    Gs = []
    for date in dates:
        d = pd.read_csv("graphs/" + country + "_" + date + ".csv",header=None)
        G = nx.DiGraph()
        
        # Gather every *unique* region that appears as a source or destination.
        nodes = set(d[0].unique()).union(set(d[1].unique()))
        nodes = sorted(nodes)
        G.add_nodes_from(nodes)

        if rand_weight:
            # Ignores actual travel flows and connect every pair of regions with weight 1 [FOR ABLATION ONLY]
            for node_start in list(G.nodes):
                for node_end in list(G.nodes):
                    G.add_edge(node_start, node_end, weight=1)
        else:
            # Add each row as a directed edge from source to destination with the observed flow weight.
            for row in d.iterrows():
                G.add_edge(row[1][0], row[1][1], weight=row[1][2])

        Gs.append(G)
        
    return Gs


def generate_new_features(Gs, labels, dates, window=7, scaled=False):
    """
        Builds a feature matrix for each day, where each row is a region and columns are:
        - the past 'window' days of case counts, followed by
        - S, E, I, R proportions from a region-level fitted SEIR model at the current day.

        ARGS:
            Gs      (list[nx.DiGraph]): One graph per day.
            labels  (pd.DataFrame): Case counts with regions as rows and date strings as columns.
            dates   (list[str]): Ordered list of date strings matching the graphs.
            window  (int): How many past days of cases to include as features.
            scaled  (bool): Whether to standardise case counts by each region's historical mean and standard deviation.

        RETURNS:
            list[np.ndarray]: One feature matrix per day of size: [n_nodes, window + 5]. The next four columns are the fitted SEIR state [S, E, I, R], and the final column is the mean log-growth rate over the case-count window.
    """
    
    features = list()

    labs = labels.copy()

    # Pre-fit a SEIR model for every region using the full date range, then cache the trajectory so that each day's loop can simply index into it without re-fitting.
    seir_cache = {}
    for node in Gs[0].nodes():
        case_counts = labs.loc[node, dates].values.astype(float)
        seir_cache[node] = _fit_seir_to_region(case_counts)  # shape: [len(dates), 4]

    for idx,G in enumerate(Gs):
        # Feature matrix: 'window' case-count columns + 4 SEIR columns + 1 growth-rate column.
        H = np.zeros([G.number_of_nodes(), window + 5])

        # Per-region mean and std. deviation of all past cases (used only when scaled = True).
        # fillna(0) guards against NaN when idx is 0 or 1 (too few history points for std).
        me = labs.loc[:, dates[:(idx)]].mean(1).fillna(0)
        sd = labs.loc[:, dates[:(idx)]].std(1).fillna(0) + 1   # +1 avoids division by zero

        # Enumerates because the row index of 'H' and the node name in 'labs' don't match directly.
        for i,node in enumerate(G.nodes()):
            if(idx < window):   # not enough history yet, hence, pad the left with zeros
                if(scaled):
                    H[i,(window-idx):(window)] = (labs.loc[node, dates[0:(idx)]] - me[node])/ sd[node]
                else:
                    H[i,(window-idx):(window)] = np.log1p(labs.loc[node, dates[0:(idx)]].values)
                if idx > 1:
                    H[i, window + 4] = np.mean(np.diff(H[i, (window-idx):window]))

            elif idx >= window:   # full window available, hence, fill all columns
                if(scaled):
                    H[i,0:(window)] =  (labs.loc[node, dates[(idx-window):(idx)]] - me[node])/ sd[node]
                else:
                    H[i,0:(window)] = np.log1p(labs.loc[node, dates[(idx-window):(idx)]].values)
                H[i, window + 4] = np.mean(np.diff(H[i, 0:window]))

            # Append the fitted SEIR state for this region at the current day.
            H[i, window:window+4] = seir_cache[node][idx]

        features.append(H)
        
    return features


def generate_new_batches(Gs, features, y, idx, graph_window, shift, batch_size, device, test_sample, decay=0.5, augment_reverse=False):
    """
        Packages graph data into batches ready to feed into the model for training.

        ARGS:
            Gs              (list[sp.spmatrix]): List of sparse adjacency matrices, one per day.
            features        (list[np.ndarray]): Node feature matrices, one per day.
            y               (list): Target case counts, one list of values per day.
            idx             (list[int]): Indices of days to include in these batches.
            graph_window    (int): How many consecutive days to stack into a single sample.
            shift           (int): How many days ahead the model should predict.
            batch_size      (int): Number of samples per batch.
            device          (torch.device): CPU or GPU to send tensors to.
            test_sample     (int): Last training-day index; used to avoid peeking at future labels during testing.
            decay           (float): Exponential time decay rate applied to edge weights.
            augment_reverse (bool): If True, append a time-reversed copy of each sample to teach decline patterns.

        RETURNS:
            adj_lst      (list[torch.sparse_coo_tensor]): Block-diagonal adjacency tensors, one per batch.
            features_lst (list[torch.FloatTensor]): Stacked feature tensors, one per batch.
            y_lst        (list[torch.FloatTensor]): Target value tensors, one per batch.
    """

    # If augmenting, double the index list: original + reversed copies.
    if augment_reverse:
        aug_idx = list(idx) + list(idx)  # second half will be reversed in feature assembly
        aug_reversed = [False] * len(idx) + [True] * len(idx)
    else:
        aug_idx = list(idx)
        aug_reversed = [False] * len(idx)

    N = len(aug_idx)
    n_nodes = Gs[0].shape[0]

    adj_lst = list()
    features_lst = list()
    y_lst = list()

    for i in range(0, N, batch_size):
        n_nodes_batch = (min(i+batch_size, N)-i)*graph_window*n_nodes
        step = n_nodes*graph_window

        adj_tmp = list()
        features_tmp = np.zeros((n_nodes_batch, features[0].shape[1]))
        y_tmp = np.zeros((min(i+batch_size, N)-i)*n_nodes)

        # Fill features and adjacency for each sample in the batch.
        for e1,j in enumerate(range(i, min(i+batch_size, N) )):
            val = aug_idx[j]
            is_rev = aug_reversed[j]

            # Stack 'graph_window' consecutive days of graphs and features for this sample.
            day_indices = list(range(val-graph_window+1, val+1))
            if is_rev:
                day_indices = day_indices[::-1]  # reverse the temporal order

            for e2, k in enumerate(day_indices):
                # Compute how many steps back from the current day this snapshot is.
                lag = graph_window - 1 - e2

                # Scale edge weights exponentially where recent days keep full weight and older days are down-weighted accordingly.
                decay_weight = np.exp(-decay * lag)

                adj_tmp.append(Gs[k-1].T * decay_weight)  
                features_tmp[(e1*step+e2*n_nodes):(e1*step+(e2+1)*n_nodes),:] = features[k]
            
            # Ensures that during testing, no peeking occurs beyond the last known day. Hence, final label is repeated instead.
            if(test_sample>0):
                if(val+shift<test_sample):
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = np.log1p(y[val+shift])
                else:
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = np.log1p(y[val])
            else:
                y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = np.log1p(y[val+shift])
        
        # Merge all daily adjacency matrices into one huge block-diagonal sparse matrix.
        adj_tmp = sp.block_diag(adj_tmp)
        adj_lst.append(sparse_matrix_to_torch_sparse_tensor(adj_tmp).to(device))
        features_lst.append(torch.FloatTensor(features_tmp).to(device))
        y_lst.append(torch.FloatTensor(y_tmp).to(device))

    return adj_lst, features_lst, y_lst


def sparse_matrix_to_torch_sparse_tensor(sparse_mx):
    """
        Converts a SciPy sparse matrix into a PyTorch sparse tensor to be used on GPU.

        ARGS:
            sparse_mx (sp.spmatrix): Any scipy sparse matrix.

        RETURNS:
            torch.sparse.FloatTensor: Equivalent sparse tensor with the same values and shape.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

# === CLASS DEFINITION === 

class AverageMeter(object):
    """
        Running tracker that keeps a rolling average of a metric across batches.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Zeroes all counters and is called at the start of each epoch."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
            Records a new measurement and updates the running average.

            ARGS:
                val (float): The new metric value to record.
                n   (int): How many samples this value represents (default 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count