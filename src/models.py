"""
    Main contributing model definitions for epidemic forecasting.
    
    Includes class definitions for:
    - The graph message-passing encoder (MPNN_Encoder) which acts as a submodule embedded inside ATMGNN.
    - The full multi-resolution attention model (ATMGNN) that analyses a region's connections at multiple scopes over time.
"""

# === IMPORTS ===

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# === CLASS & FUNCTION DEFINITIONS ===

# Message Passing Neural Network Encoder
class MPNN_Encoder(nn.Module):
    """
        The graph encoder that summarizes each node by gathering information from its neighbours via message passing.
        Used as the first processing stage before multi-resolution coarsening.
        It produces no predictions, only embeddings.
    """

    def __init__(self, nfeat, nhidden, nout, dropout):
        """
            ARGS:
                nfeat    (int): Number of input features per node.
                nhidden  (int): Size of the hidden (intermediate) representation.
                nout     (int): Size of the final output embedding per node.
                dropout  (float): Fraction of neurons randomly turned off during training to prevent overfitting.
        """
        super(MPNN_Encoder, self).__init__()
        self.nhidden = nhidden
        self.nfeat = nfeat

        # Two graph convolution layers that spread information between neighbouring nodes via message passing.
        self.conv1 = GCNConv(nfeat, nhidden)
        self.conv2 = GCNConv(nhidden, nhidden)
        
        # Learns how important each connection (edge) between two nodes is based on their features (hence, 2*nfeat) and outputs 1 number.
        self.edge_attn = nn.Linear(2 * nfeat, 1)
        
        # Batch normalisation (per layer), such that mean = 0 and std = 1.
        self.bn1 = nn.BatchNorm1d(nhidden)
        self.bn2 = nn.BatchNorm1d(nhidden)

        # Two fully-connected dense layers that compress the concatenated features into the output size (nout).
        # Done in two consecutive  steps to achieve gradual, non-linear compression.
        self.fc1 = nn.Linear(nfeat+2*nhidden, nhidden ) # compress to size: nhidden
        self.fc2 = nn.Linear(nhidden, nout) # compress to size: nout

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()   # for non-linearity (zeroes out negative values).

    def forward(self, adj, x):
        """
            Encodes node features by passing messages along edges.

            ARGS:
                adj (torch.sparse_coo_tensor): Sparse adjacency matrix describing which nodes are connected.
                x   (torch.Tensor): Feature matrix of node with shape: [num_nodes, nfeat].

            RETURNS:
                torch.Tensor: Encoded node embeddings of shape: [num_nodes, nout].
        """
        # Unpack edge sources, destinations and weights from sparse adjacency tensor (adj)
        lst = list()   # for collecting feature snapshots at different stages and final concatenation 
        weight = adj.coalesce().values()    # merge duplicates and extract non-zero edge weights
        adj = adj.coalesce().indices()  # merge duplicates and extract non-zero connections
        src, dst = adj[0], adj[1]   # collects source node indices and destination node indices
        
        # Score each edge: "How much should this connection matter given the features at both ends?", given a score range [0, 1].
        attn = torch.sigmoid(self.edge_attn(torch.cat([x[src], x[dst]], dim=1))).squeeze(-1)
        
        # Re-scale original edge weights by the learned attention scores.
        weight = weight * attn
        lst.append(x)   # save original raw features as first element

        # First message-passing round where each node gathers weighted info from its neighbours.
        x = self.relu(self.conv1(x,adj,edge_weight=weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)

        # Second message-passing round where nodes get refined representations using the updated embeddings.
        x = self.relu(self.conv2(x, adj,edge_weight=weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)

        # Stack raw features and both hidden layers, then compress to output size.
        x = torch.cat(lst, dim=1)   # concatenate all three snapshots horizontally
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return x 


# Attention Temporal Multiresolution Graph Neural Network 
class ATMGNN(nn.Module):
    """
        Full TGNN model: encodes graphs at multiple coarseness levels, then uses self-attention
        across time window to produce node prediction.
    """

    def __init__(self, nfeat, nhidden, nout, n_nodes, window, dropout, nhead = 1, num_clusters = [10, 5], use_norm = False):
        """
            ARGS:
                nfeat        (int): Number of input features per node per timestep (equals 'window').
                nhidden      (int): Size of the internal embedding vectors produced by GCN layers.
                nout         (int): Size of the final prediction per node.
                n_nodes      (int): Total number of nodes (regions) in the graph (country).
                window       (int): Number of past days or timestampes fed as input features.
                dropout      (float): Fraction of neurons randomly switched off during training to prevent overfitting 
                nhead        (int): Number of parallel attention heads (singular perspective).
                num_clusters (list[int]): Resolution levels for graph coarsening.
                use_norm     (bool): Whether to L2-normalise embeddings at each resolution level before concatenation.
        """
        
        super(ATMGNN, self).__init__()
        
        self.window = window
        self.n_nodes = n_nodes
        self.nhidden = nhidden
        self.nfeat = nfeat
        self.nhead = nhead  # more heads = look at time form more perspectives
        self.use_norm = use_norm

        # Encodes the full-resolution graph (finest coarseness level w/ no cluster-grouping).
        self.bottom_encoder = MPNN_Encoder(nfeat, nhidden, nhidden, dropout)

        self.num_clusters = num_clusters

        # Defines clustering layer and one encoder per coarseness level.
        self.middle_linear = nn.ModuleList()
        self.middle_encoder = nn.ModuleList()

        for size in self.num_clusters:
            # Maps each node's embeddings to 'size'-dimensional cluster assignments, i.e., decides which group each node belongs to.
            self.middle_linear.append(nn.Linear(nhidden, size))
            
            # Learns a new embedding for each cluster after the graph is shrunk.
            self.middle_encoder.append(nn.Linear(nhidden, nhidden))

        # With 2 coarsening levels and the bottom (finest) level, we have 3 x 'nhidden' features per node.
        # Two layers that blend information from all coarseness levels into one representation.
        _mix_hidden = 4 * nhidden
        self.mix_1 = nn.Linear((len(self.num_clusters) + 1) * nhidden, _mix_hidden) # expand to identify cross-resolution interactions
        self.mix_2 = nn.Linear(_mix_hidden, (len(self.num_clusters) + 1) * nhidden) # compress back to discard noise

        # Lets each timestep decide how much to pay attention to every other timestep via voting.
        self.self_attention = nn.MultiheadAttention((len(self.num_clusters) + 1) * nhidden, self.nhead, dropout=dropout)
        
        # Turns window timesteps into a single summary vector per region.
        self.linear_reduction = nn.Linear(self.window, 1)
        
        # Final projection layer with combined temporal summary and raw input features, then prediction.
        self.fc1 = nn.Linear((len(self.num_clusters) + 1) * nhidden + window * nfeat, nhidden)
        self.fc2 = nn.Linear(nhidden, nout)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()   # for non-linearity
        
        
    def encode(self, adj, x):
        """
            Runs the multi-resolution graph encoding and temporal attention pipeline,
            returning the conditioning representation before the final fully-connected layers (for diffusion decoder).

            ARGS:
                adj (torch.sparse_coo_tensor): Shared adjacency matrix for the graph.
                x   (torch.Tensor): Flattened node features of shape: [window*n_nodes, nfeat].

            RETURNS:
                torch.Tensor: Conditioning representation of shape: [batch*n_nodes, cond_dim].
        """
        
        # Save raw input features before processing for end concatenation (skip connection).
        skip = x.view(-1, self.window, self.n_nodes, self.nfeat)    # reshape accordingly
        skip = torch.transpose(skip, 1, 2).reshape(-1, self.window, self.nfeat) # reorganizes such that each row = one city across time.

        x = x.view(-1, self.nfeat)  # flattens to MPNN_Encoder expected format

        # Collect embeddings from the finest level and every coarser level.
        all_latents = []

        # Encode the graph at its finest resolution (all individual nodes).
        bottom_latent = self.bottom_encoder(adj, x)
        all_latents.append(bottom_latent)

        # Tracks the cumulative mapping from original nodes to current coarse clusters.
        product = None

        # Multiresolution construction
        adj = adj.to_dense()
        latent = bottom_latent

        for level in range(len(self.num_clusters)):

            # Assign each node to exactly one cluster 
            assign = self.middle_linear[level](latent)
            assign = F.gumbel_softmax(assign, tau = 1, hard = True, dim = 1)

            # Builds the full mapping from original nodes to clusters at this level.
            if level == 0:
                product = assign
            else:
                product = torch.matmul(product, assign)

            # Averages node features within each cluster to get cluster-level features.
            x = torch.matmul(assign.transpose(0, 1), latent)
            x = F.normalize(x, dim = 1)

            # Shrinks the adjacency matrix to only describe connections between clusters.
            adj = torch.matmul(torch.matmul(assign.transpose(0, 1), adj), assign)
            # Row-wise normalisation: each cluster's outgoing weights sum to 1 as dividing by the global sum would collapse all values near zero for dense graphs.
            row_sums = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
            adj = adj / row_sums

            # Computes new cluster embeddings using a simple graph convolution on the coarse graph.
            latent = torch.tanh(self.middle_encoder[level](torch.matmul(adj, x)))

            # Projects cluster embeddings back to original nodes so all levels share the same shape.
            extended_latent = torch.matmul(product, latent)
            all_latents.append(extended_latent)

        # Normalization
        if self.use_norm == True:
            for idx in range(len(all_latents)):
                all_latents[idx] = all_latents[idx] / torch.norm(all_latents[idx], p = 2)

        # Concatenate all resolutions
        representation = torch.cat(all_latents, dim = 1)
        x = representation

        # Blend the multi-resolution features through two fully-connected layers.
        x = torch.relu(self.mix_1(x))
        x = torch.relu(self.mix_2(x))

        # Reshape so the time dimension is explicity structured as [window, batch*n_nodes, features].
        x = x.view(-1, self.window, self.n_nodes, x.size(1)) 
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3)) 

        # Each timestep attends to all other timesteps, learning which past days matter most.
        x, _ = self.self_attention(x, x, x)
        x = torch.transpose(x, 0, 2)
        
        # Compress the entire time window into one vector per node (region).
        x = self.linear_reduction(x)
        x = x.squeeze()
        x = torch.transpose(x, 0, 1)

        # Flatten and append the raw input features as a skip connection.
        skip = skip.reshape(skip.size(0),-1)
        x = torch.cat([x,skip], dim=1)

        return x

    def forward(self, adj, x):
        """
            Runs the complete ATMGNN pipeline: multi-resolution graph encoding -> time attention -> prediction.

            ARGS:
                adj (torch.sparse_coo_tensor): Shared adjacency matrix for the graph (remains same across timesteps).
                x   (torch.Tensor): Flattened node features for all timesteps of shape: [window*n_nodes, nfeat].

            RETURNS:
                torch.Tensor: Predicted values for every node (region) of shape: [n_nodes,] or [batch*n_nodes,].
        """
        x = self.encode(adj, x)

        # Final layers projection to output size, flattened to a one-dimensional prediction vector.
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)

        return x


# === DIFFUSION COMPONENTS ===

def _sinusoidal_embedding(timesteps, dim):
    """
        Computes sinusoidal positional embeddings for diffusion timesteps.
        Encodes each integer timestep as a fixed-length vector so the denoiser can distinguish which noise level it is operating at.

        ARGS:
            timesteps (torch.LongTensor): Integer timestep indices of shape: [N].
            dim       (int): Embedding dimensionality.

        RETURNS:
            torch.Tensor: Positional embeddings of shape: [N, dim].
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ConditionedDenoiser(nn.Module):
    """
        Lightweight MLP (multi-layer perceptron) that predicts the noise component in a noisy target,
        given the current timestep embedding and the conditioning vector produced by ATMGNN's encoder.
    """

    def __init__(self, target_dim, cond_dim, time_dim=32, hidden_dim=128):
        """
            ARGS:
                target_dim (int): Dimensionality of the denoised target (1 for scalar case counts).
                cond_dim   (int): Dimensionality of the conditioning vector from the encoder.
                time_dim   (int): Dimensionality of the sinusoidal timestep embedding.
                hidden_dim (int): Width of the hidden layers.
        """
        super(ConditionedDenoiser, self).__init__()
        self.time_dim = time_dim
        input_dim = target_dim + time_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x_noisy, t, cond):
        """
            Predicts the noise added to noisy targets at timestep t, conditioned on encoder output.

            ARGS:
                x_noisy (torch.Tensor): Noisy targets of shape: [N, target_dim].
                t       (torch.LongTensor): Timestep indices of shape: [N].
                cond    (torch.Tensor): Conditioning vectors of shape: [N, cond_dim].

            RETURNS:
                torch.Tensor: Predicted noise of shape: [N, target_dim].
        """
        t_emb = _sinusoidal_embedding(t, self.time_dim)
        inp = torch.cat([x_noisy, t_emb, cond], dim=-1)
        return self.net(inp)

# Denoising Diffusion Probabilistic Model (DDPM)
class DiffusionDecoder(nn.Module):
    """
        DDPM-based decoder that replaces the deterministic fc1 -> fc2 prediction head.

        During training it adds noise to the ground-truth target at a randomly sampled
        timestep and trains the denoiser to recover the noise.

        During inference it starts from pure Gaussian noise and iteratively denoises, conditioned on the ATMGNN encoder output, to produce a prediction.
        It draws multiple independent samples yields a distribution whose mean is the point forecast and whose spread quantifies uncertainty.
    """

    def __init__(self, target_dim, cond_dim, diffusion_steps=8, hidden_dim=128):
        """
            ARGS:
                target_dim      (int): Dimensionality of the prediction target per node.
                cond_dim        (int): Dimensionality of the encoder conditioning vector.
                diffusion_steps (int): Number of DDPM forward/reverse steps (T).
                hidden_dim      (int): Hidden width of the denoiser MLP.
        """
        super(DiffusionDecoder, self).__init__()
        self.T = diffusion_steps
        self.target_dim = target_dim

        self.denoiser = ConditionedDenoiser(
            target_dim, cond_dim, time_dim=32, hidden_dim=hidden_dim
        )

        # Linear beta schedule
        betas = torch.linspace(1e-4, 0.1, self.T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        # Register as buffers so they move with the model to the correct device, but are not treated as trainable parameters.
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1.0 - alpha_bar))

    def q_sample(self, x_0, t, noise=None):
        """
            Forward diffusion process: corrupts clean targets with Gaussian noise.

            ARGS:
                x_0   (torch.Tensor): Clean targets of shape: [N, target_dim].
                t     (torch.LongTensor): Timestep indices of shape: [N].
                noise (torch.Tensor | None): Pre-sampled noise; generated internally if None.

            RETURNS:
                x_t   (torch.Tensor): Noisy targets of shape: [N, target_dim].
                noise (torch.Tensor): The noise that was added.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return sqrt_ab * x_0 + sqrt_omab * noise, noise

    def compute_loss(self, x_0, cond):
        """
            Computes the DDPM training loss (epsilon-prediction MSE).

            For each node, a timestep is sampled uniformly from [0, T), the target is
            noised to that level, and the denoiser predicts the noise.

            ARGS:
                x_0  (torch.Tensor): Ground-truth targets of shape: [N, target_dim].
                cond (torch.Tensor): Conditioning vectors of shape: [N, cond_dim].

            RETURNS:
                torch.Tensor: Scalar MSE loss between predicted and true noise.
        """
        N = x_0.size(0)
        t = torch.randint(0, self.T, (N,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_noisy, _ = self.q_sample(x_0, t, noise)
        noise_pred = self.denoiser(x_noisy, t, cond)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, cond, n_samples=1):
        """
            Reverse diffusion: iteratively denoise from pure Gaussian noise.

            ARGS:
                cond      (torch.Tensor): Conditioning vectors of shape: [N, cond_dim].
                n_samples (int): How many independent denoising trajectories to run.

            RETURNS:
                torch.Tensor: If n_samples == 1 -> [N, target_dim]. If n_samples >  1 -> [n_samples, N, target_dim].
        """
        N = cond.size(0)
        device = cond.device

        all_samples = []
        for _ in range(n_samples):
            x = torch.randn(N, self.target_dim, device=device)

            for t_idx in reversed(range(self.T)):
                t = torch.full((N,), t_idx, device=device, dtype=torch.long)
                noise_pred = self.denoiser(x, t, cond)

                alpha_t = self.alphas[t_idx]
                alpha_bar_t = self.alpha_bar[t_idx]
                beta_t = self.betas[t_idx]

                # DDPM reverse-step mean
                coef1 = 1.0 / torch.sqrt(alpha_t)
                coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
                mu = coef1 * (x - coef2 * noise_pred)

                # Add stochastic noise for all steps except the final one.
                if t_idx > 0:
                    sigma = torch.sqrt(beta_t)
                    x = mu + sigma * torch.randn_like(x)
                else:
                    x = mu

            all_samples.append(x)

        if n_samples == 1:
            return all_samples[0]
        return torch.stack(all_samples, dim=0)


# Hybrid TGNN-Diffusion Model
class ATMGNN_Diff(ATMGNN):
    """
        Hybrid model that keeps the full ATMGNN multi-resolution encoder as an
        unchanged conditioning backbone and replaces the deterministic fc1 -> fc2
        prediction head with a DDPM-based diffusion decoder.

        At training time the decoder learns to denoise ground-truth targets
        conditioned on the encoder output.  At inference time the reverse
        diffusion process generates predictions from pure noise; drawing
        multiple samples yields a distribution whose mean is the point forecast
        and whose spread is the confidence interval.
    """

    def __init__(self, nfeat, nhidden, nout, n_nodes, window, dropout, nhead=1, num_clusters=[10, 5], use_norm=False, diffusion_steps=100, decoder_hidden=64):
        """
            ARGS:
                nfeat           (int): Number of input features per node per timestep.
                nhidden         (int): Size of internal GCN embedding vectors.
                nout            (int): Prediction dimensionality per node (typically 1).
                n_nodes         (int): Total number of nodes (regions) in the graph.
                window          (int): Number of past timesteps used as input.
                dropout         (float): Dropout rate for regularisation.
                nhead           (int): Number of attention heads.
                num_clusters    (list[int]): Resolution levels for graph coarsening.
                use_norm        (bool): Whether to L2-normalise embeddings before concatenation.
                diffusion_steps (int): Number of DDPM denoising steps (T).
                decoder_hidden  (int): Hidden width of the denoiser MLP.
        """
        super(ATMGNN_Diff, self).__init__(
            nfeat, nhidden, nout, n_nodes, window, dropout,
            nhead, num_clusters, use_norm
        )
        # Conditioning dimension = multi-resolution features + raw skip features.
        cond_dim = (len(num_clusters) + 1) * nhidden + window * nfeat
        self.diffusion = DiffusionDecoder(
            target_dim=nout, cond_dim=cond_dim,
            diffusion_steps=diffusion_steps, hidden_dim=decoder_hidden
        )

    def forward(self, adj, x, n_samples=1):
        """
            Inference path: encode the input then sample from the diffusion decoder.

            ARGS:
                adj       (torch.sparse_coo_tensor): Batch adjacency matrix.
                x         (torch.Tensor): Batch node features.
                n_samples (int): Number of independent samples for uncertainty estimation.

            RETURNS:
                torch.Tensor: If n_samples == 1 -> [batch*n_nodes] (matches ATMGNN output shape). If n_samples >  1 -> [n_samples, batch*n_nodes].
        """
        cond = self.encode(adj, x)
        if n_samples == 1:
            # Direct prediction via the inherited fc head: deterministic, stable point forecast.
            out = self.relu(self.fc1(cond))
            out = self.dropout(out)
            return self.relu(self.fc2(out)).squeeze().view(-1)
        # Diffusion sampling for uncertainty quantification (n_samples > 1).
        samples = self.diffusion.sample(cond, n_samples=n_samples)
        return samples.squeeze(-1)

    def compute_diffusion_loss(self, adj, x, y_target, node_weights=None):
        """
            Training path: encode the input then compute the DDPM denoising loss.

            ARGS:
                adj          (torch.sparse_coo_tensor): Batch adjacency matrix.
                x            (torch.Tensor): Batch node features.
                y_target     (torch.Tensor): Ground-truth target values of shape: [batch*n_nodes].
                node_weights (torch.Tensor | None): Per-node loss weights for scale balancing.

            RETURNS:
                torch.Tensor: Scalar denoising loss.
        """
        cond = self.encode(adj, x)
        if y_target.dim() == 1:
            y_target = y_target.unsqueeze(-1)
        diffusion_loss = self.diffusion.compute_loss(y_target, cond)
        direct = self.relu(self.fc1(cond))
        direct = self.dropout(direct)
        direct = self.relu(self.fc2(direct)).squeeze(-1).view(-1)
        y_flat = y_target.squeeze(-1).view(-1)
        if node_weights is not None:
            w = node_weights.repeat(direct.size(0) // node_weights.size(0))
            aux_loss = (w * (direct - y_flat) ** 2).mean()
        else:
            aux_loss = F.mse_loss(direct, y_flat)
        return diffusion_loss + 0.1 * aux_loss