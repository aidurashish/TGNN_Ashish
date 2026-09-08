"""
    This module contains the class definitions and respective hierarchy that comprises the DiffATMGNN model:
    
    1. MPNN_Encoder - The graph message-passing encoder class which acts as a submodule of the ATMGNN class
    2. ATMGNN - The multi-resolution self-attention model that analyses a region's connections at multiple scopes over time.
    3. DiffATMGNN - Delivers the final prediction with the help of Diffusion-based components: ConditionedDenoiser & DiffusionDecoder.
"""

# === IMPORTS ===

import math     # Built-in Math module
import torch        # Core deep-learning framework 
import torch.nn as nn       # Neural-network module with pre-built layers, activation functions, and loss functions
import torch.nn.functional as F     # NN functions such as ReLu, Softmax, Dropout that have no memory
from torch_geometric.nn import GCNConv      # GCN layer from PyTorch Geometric 

# === CLASS & FUNCTION DEFINITIONS ===

# Message Passing Neural Network Encoder
class MPNN_Encoder(nn.Module):
    """
        The graph message-passing encoder that summarizes each region (node) of the graph by gathering information from its neighbours via message passing.
        It is the first processing stage in the overall pipeline before multi-resolution coarsening.
        
        NOTE: It produces only embeddings, not predictions.
    """

    def __init__(self, nfeat, nhidden, nout, dropout):
        """
            ARGS:
                nfeat    (int): Number of input features per node.
                nhidden  (int): Size of the hidden (intermediate) representation or summary vector.
                nout     (int): Size of the final output embedding/vector per node.
                dropout  (float): Fraction of neurons randomly turned off during training to prevent overfitting.
        """
        super(MPNN_Encoder, self).__init__()        # Call parent class (nn.Module) to track defined layers using PyTorch
        self.nhidden = nhidden
        self.nfeat = nfeat

        # Two GCN layers that spread information between neighbouring nodes via message-passing.
        # Each region (node) ends up with a summary vector of length: [nhidden]
        self.conv1 = GCNConv(nfeat, nhidden)    # nfeat -> nhidden
        self.conv2 = GCNConv(nhidden, nhidden)  # nhidden -> nhidden
        
        # Learns how important each connection (edge) between two nodes is based on their features (2*nfeat) and outputs 1 number.
        # It utilizes a linear layer with adjustable learned weights to return a weighted sum.
        # Attention Formula used: attn(u, v) = sigmoid( w^T * [x_u || x_v] + b )
        # where, w^T = learned weight vector, b = learned bias scalar, and [x_u || x_v] = 2*nfeat of source and destination concatenated
        self.edge_attn = nn.Linear(2 * nfeat, 1)
        
        # Batch normalisation (for each message-passing layer), such that mean = 0, std = 1 and all values are similarly scaled.
        self.bn1 = nn.BatchNorm1d(nhidden)
        self.bn2 = nn.BatchNorm1d(nhidden)

        # Two fully-connected dense layers that compress the concatenated features into the output size (nout).
        # Done in two consecutive  steps to achieve gradual, non-linear compression, broken up by ReLU.
        self.fc1 = nn.Linear(nfeat+2*nhidden, nhidden ) # compress to size: nhidden
        self.fc2 = nn.Linear(nhidden, nout) # compress to size: nout (refinement when nhidden = nout)

        self.dropout = nn.Dropout(dropout)  # To prevent overfitting
        self.relu = nn.ReLU()   # Introduces non-linearity (zeroes out negative values).

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
        lst = list()   # To collect 3 snapshots: raw features, output of 'conv1' and output of 'conv2'
        weight = adj.coalesce().values()    # clean up duplicate entries (coalesce) and extract weights (values) from 'adj' to get a vector of shape [num_edges]
        adj = adj.coalesce().indices()      # clean up duplicate entries (coalesce) and extract region connections (indices) to get a 2-row matrix comprising sources and destinations ([2, num_edges])
        src, dst = adj[0], adj[1]   # unpack into source and destination regions
        
        # Score each edge: "How much should this connection matter given the features at both ends?", given a score range [0, 1].
        
        # First, concatenate for each edge, the features of both 'src' and 'dst' to get a vector of shape [num_edges, 2*nfeat].
        # Secondly, pass via 'edge_attn()' to obtain weighted sum per edge to get a vector of shape [num_edges, 1]
        # Then, project it into range (0, 1) with the sigmoid function.
        # Finally, remove any trailing dimension of size 1 with squeeze(-1) to get a flat list of attention scores. Eg: [4, 1] -> [4]
        attn = torch.sigmoid(self.edge_attn(torch.cat([x[src], x[dst]], dim=1))).squeeze(-1)    
        
        # Re-scale original edge weights by the learned attention scores.
        weight = weight * attn
        lst.append(x)   # Save original raw features as first element -> [N, nfeat]; where 'N' is number of regions in the graph

        # First message-passing round where each node gathers weighted info. from its neighbours.
        x = self.relu(self.conv1(x,adj,edge_weight=weight)) # ReLu eliminates negative values which are irrelevant or contradictory.
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)   # Save results of 'conv1' as second element -> [N, nhidden]

        # Second message-passing round where nodes get refined representations using the updated embeddings.
        x = self.relu(self.conv2(x, adj,edge_weight=weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)   # Save results of 'conv2' as third element -> [N, nhidden]

        # Stack raw features and both hidden layers, then compress to output size.
        x = torch.cat(lst, dim=1)   # Concatenate all three elements horizontally to get vector of shape [N, nfeat + 2*nhidden]
        x = self.relu(self.fc1(x))  # Compress to shape [N, nhidden]
        x = self.dropout(x)
        x = self.relu(self.fc2(x))  # Compress to shape [N, nout]
        # NOTE: If nhidden = nout, fc2 is performing a learned non-linear transformation
        return x    # Shape [N, nout]


# Attention Temporal Multiresolution Graph Neural Network 
class ATMGNN(nn.Module):
    """
        Encodes graphs at multiple coarseness levels and uses self-attention across a time window.
        
        NOTE: Can produce predictions (for model comparison) but primarily produces encodings for the full DiffATMGNN model implementation.
    """

    def __init__(self, nfeat, nhidden, nout, n_nodes, window, dropout, nhead = 1, num_clusters = [10, 5], use_norm = False):
        """
            ARGS:
                nfeat        (int): Number of input features per node.
                nhidden      (int): Size of the hidden (intermediate) representation or summary vector.
                nout         (int): Size of the final output embedding/vector per node.
                n_nodes      (int): Total number of nodes (regions) in the graph (country).
                window       (int): Number of past days or timestampes fed as input features.
                dropout      (float): Fraction of neurons randomly switched off during training to prevent overfitting 
                nhead        (int): Number of parallel attention heads
                num_clusters (list[int]): Resolution levels for graph coarsening.
                use_norm     (bool): Whether to L2-normalise embeddings at each resolution level before concatenation.
        """
        
        super(ATMGNN, self).__init__()      # Call parent class (nn.Module) to track defined layers using PyTorch
        
        self.window = window
        self.n_nodes = n_nodes
        self.nhidden = nhidden
        self.nfeat = nfeat
        self.nhead = nhead  # More heads = look at time from multiple perspectives
        self.use_norm = use_norm

        # Create MPNN_Encoder instantiation for the full-resolution graph, i.e., finest coarseness level w/ no cluster-grouping.
        self.bottom_encoder = MPNN_Encoder(nfeat, nhidden, nhidden, dropout)    # nout = nhidden to match the output encoding size (nhidden) at every coarseness step.

        self.num_clusters = num_clusters

        # Define two empty lists to hold one layer per coarsening level
        # NOTE: nn.ModuleList() is used over a simple Python list as PyTorch would not be able to see the layers inside it and train their weights. nn.ModuleList() registers every layer as a model parameter.
        self.middle_linear = nn.ModuleList()    # A learned linear layer that decides which cluster each region belongs to.
        self.middle_encoder = nn.ModuleList()   # A learned linear layers that encodes each cluster after the graph (adj) is shrunk.

        # Loop to create and store cluster layer objects to be used in encode()
        for size in self.num_clusters:      # 10 clusters and 5 clusters respectively
            self.middle_linear.append(nn.Linear(nhidden, size)) # Takes each region's 'nhidden' embedding and outputs an assignment matrix of length 'size' ([n_nodes, size])
            self.middle_encoder.append(nn.Linear(nhidden, nhidden)) # After regions have been assigned to their respective 'size' number of clusters, learn a new 'nhidden' summary for each of the clusters.

        # Concatenate the two specified coarsening levels and the bottom (finest) level to obtain 3*'nhidden' features per node.
        # Then, define two layers that blend information from all coarseness levels into one representation.
        _mix_hidden = 4 * nhidden
        self.mix_1 = nn.Linear((len(self.num_clusters) + 1) * nhidden, _mix_hidden) # Expand to 4*nhidden to identify cross-resolution interactions
        self.mix_2 = nn.Linear(_mix_hidden, (len(self.num_clusters) + 1) * nhidden) # Compress back to 3*nhidden to discard noise

        # Define a multi-head self-attention layer over the time window to let each timestep decide how much to pay attention to every other timestep via voting.
        self.self_attention = nn.MultiheadAttention((len(self.num_clusters) + 1) * nhidden, self.nhead, dropout=dropout)
        
        # Turns window timesteps into a single summary vector per region, i.e., [window, n_nodes, (len(self.num_clusters) + 1) * nhidden] -> [n_nodes, (len(self.num_clusters) + 1) * nhidden]
        self.linear_reduction = nn.Linear(self.window, 1)
        
        # Final projection layer with combined temporal summary and raw input features, then prediction.
        # NOTE: Not used in the main DiffATMGNN pipeline
        self.fc1 = nn.Linear((len(self.num_clusters) + 1) * nhidden + window * nfeat, nhidden)
        self.fc2 = nn.Linear(nhidden, nout)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()   # For non-linearity
        
        
    def encode(self, adj, x):
        """
            Runs the multi-resolution graph encoding and temporal attention pipeline, returning the conditioning representation before the final fully-connected layers (for the diffusion decoder).

            ARGS:
                adj (torch.sparse_coo_tensor): Sparse adjacency matrix describing which nodes are connected.
                x   (torch.Tensor): Flattened node features of shape: [window*n_nodes, nfeat].

            RETURNS:
                x   (torch.Tensor): Conditioning representation of shape: [batch*n_nodes, cond_dim].
        """
        # Reshape/unfold features matrix accordingly: [window × n_nodes, nfeat] -> [batch, window, n_nodes, nfeat] to form skip connection which preserves raw features
        # NOTE: '-1' implies that PyTorch calculates the (batch) size automatically. In practice, batch = 1 for ATMGNN since one country graph is processed at a time.
        skip = x.view(-1, self.window, self.n_nodes, self.nfeat)  
        
        # Reorganize such that each row = one region across time.
        # [batch, window, n_nodes, nfeat] → [n_nodes, window, nfeat] by swapping days and regions axis and then flattening.
        skip = torch.transpose(skip, 1, 2).reshape(-1, self.window, self.nfeat) 

        x = x.view(-1, self.nfeat)  # Flatten feature matrix to MPNN_Encoder expected format as a safety measure 

        # List to collect cluster embeddings from the finest level and every coarser level.
        all_latents = []

        # Encode the graph at its finest resolution (all individual nodes).
        bottom_latent = self.bottom_encoder(adj, x)
        all_latents.append(bottom_latent)

        product = None      # Variable to track the cumulative chain of cluster assignments, i.e., which original region maps to which cluster.

        adj = adj.to_dense()   # Converts the sparse 'adj' matrix to a dense 2-D matrix for PyTorch's 'matmul()' matrix multiplication operation
        latent = bottom_latent

        # Multi-resolution coarsening loop
        for level in range(len(self.num_clusters)):

            # Score each node to exactly one cluster to get a matrix of shape [n_nodes, n_clusters]
            assign = self.middle_linear[level](latent)
            
            # Softmax used to eliminate fuzzy assignment of nodes to clusters by picking highest-scoring cluster per node via a hard binary decision (hard=True).
            assign = F.gumbel_softmax(assign, tau = 1, hard = True, dim = 1)

            # Build the full mapping from original nodes to clusters at the current level.
            if level == 0:
                product = assign
            else:
                product = torch.matmul(product, assign) # Matrix multiplication: [n_nodes, 10]*[n_nodes, 5] -> [n_nodes, 5]

            # Average the node features within each cluster to get cluster-level features.
            x = torch.matmul(assign.transpose(0, 1), latent) # Calculate average sum of individual embeddings of regions per cluster 
            x = F.normalize(x, dim = 1) # Clusters with more regions will naturally have larger values. Normalization rescales each cluster's vector so its length equals maximum 1

            # Shrink and restructure the adjacency matrix to describe connections between clusters.
            # [n_nodes, n_nodes] -> [n_clusters, n_clusters]
            adj = torch.matmul(torch.matmul(assign.transpose(0, 1), adj), assign)
            
            # Row-wise normalisation: each cluster's outgoing weights sum up to 1 as dividing by the global sum would collapse all values near zero for dense graphs.
            # This prevents a cluster containing many regions from having large connection weights as a result of gaining more edges during the merge.
            row_sums = adj.sum(dim=1, keepdim=True).clamp(min=1e-8) # clamp() acts as a safeguard against division by zero, in case no edges exist
            adj = adj / row_sums

            # Compute new cluster embeddings on the coarse graph.
            # NOTE: No ReLU is used because negative embeddings are useful to discard irrelevant clusters, while still maintaining non-linearity
            latent = torch.tanh(self.middle_encoder[level](torch.matmul(adj, x)))

            # Project cluster embeddings back to original nodes so all levels share the same shape.
            extended_latent = torch.matmul(product, latent)
            all_latents.append(extended_latent)

        # L2-Normalization [OPTIONAL]
        if self.use_norm == True:
            for idx in range(len(all_latents)):
                all_latents[idx] = all_latents[idx] / torch.norm(all_latents[idx], p = 2)

        # Concatenate all resolution embeddings side-by-side such that each node now has 3 × nhidden features
        representation = torch.cat(all_latents, dim = 1)
        x = representation

        # Blend the multi-resolution features through two fully-connected layers.
        x = torch.relu(self.mix_1(x))   # expand to 4*nhidden
        x = torch.relu(self.mix_2(x))   # compress back to 3*nhidden

        # Reshape so that the time dimension is explicity structured as [window, batch*n_nodes, features], which is the format 'nn.MultiheadAttention' expects.
        x = x.view(-1, self.window, self.n_nodes, x.size(1)) 
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3)) 

        # Self-attention: each timestep attends to all other timesteps, learning which past days matter most.
        # NOTE: The _ discards the unnecessary attention weight matrix. Also, passing x three times means query = key = value = x.
        x, _ = self.self_attention(x, x, x)
        x = torch.transpose(x, 0, 2)    # [window, batch*n_nodes, nfeat] -> [nfeat, batch*n_nodes, window] for linear_reduction's expected format (where 'window' is last)
        
        # Compress the entire time window into one vector per node (region).
        x = self.linear_reduction(x)
        # Remove the size-1 day axis ([nfeat, batch*n_nodes, 1] -> [batch*n_nodes, nfeat])
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
                x   (torch.Tensor): Predicted values for every node (region) of shape: [n_nodes,] or [batch*n_nodes,].
        """
        x = self.encode(adj, x)

        # Final layers projection to output size, flattened to a one-dimensional prediction vector.
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)  # Flatten to a 1-D vector of predictions, one per region

        return x


# === DIFFUSION COMPONENTS ===

def _sinusoidal_embedding(timesteps, dim):
    """
        Helper function to compute sinusoidal positional embeddings for diffusion timesteps.
        It encodes each integer timestep as a fixed-length vector so the denoiser can distinguish which noise level it is operating at.

        ARGS:
            timesteps (torch.LongTensor): Integer timestep indices of shape: [N].
            dim       (int): Embedding dimensionality.

        RETURNS:
            torch.Tensor: Positional embeddings of shape: [N, dim].
    """
    half = dim // 2 # To generate dim//2 cosine and dim//2 sine values for the embedding
    
    # Create dim//2 different frequencies, ranging from fast-oscillating to slow-oscillating.
    # NOTE: Low-index frequencies change slowly across timesteps while high-index frequencies change quickly. Together they give every timestep a unique pattern.
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    ) 

    # Multiply each timestep by all 16 frequencies.
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0) # Shape [N, dim//2]
    
    # Apply cosine and sine to the dim//2 values and then concatenate
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ConditionedDenoiser(nn.Module):
    """
        Lightweight MLP (multi-layer perceptron) that predicts the noise in a noisy target, 
        given the current timestep embedding and the conditioning vector produced by ATMGNN's encoder.
    """

    def __init__(self, target_dim, cond_dim, time_dim=32, hidden_dim=128):
        """
            ARGS:
                target_dim (int): Dimensionality of the denoised target (1 for scalar case counts).
                cond_dim   (int): Dimensionality of the conditioning vector from the ATMGNN encoder.
                time_dim   (int): Dimensionality of the vector representing a particular timestep.
                hidden_dim (int): Hidden width of the MLP layers
        """
        super(ConditionedDenoiser, self).__init__() #  Call parent class (nn.Module) to track defined layers using 
        
        self.time_dim = time_dim
        input_dim = target_dim + time_dim + cond_dim    # Total input width
        
        # Construct a three-layer MLP: Input is all three pieces concatenated. Output is the predicted noise, i.e., one number per node.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim),
        )
        # NOTE: SiLU is an activation function similar to ReLU but smoother and appropriate for diffusion processes attempting to learn stable noise predictions

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
        t_emb = _sinusoidal_embedding(t, self.time_dim) # Convert the integer timestep into a 32-number vector the MLP can process.
        inp = torch.cat([x_noisy, t_emb, cond], dim=-1) # Concatenate all three inputs
        return self.net(inp)    # Pass via MLP and return predicted noise

# Denoising Diffusion Probabilistic Model (DDPM)
class DiffusionDecoder(nn.Module):
    """
        A DDPM decoder that adds Gaussian noise to the ground-truth target at a randomly sampled
        timestep and trains itself to recover the noise while conditioned on the ATMGNN encoder output.
        
        It draws multiple samples and yields a distribution whose mean is the point forecast and whose spread is used to calculate uncertainty.
    """

    def __init__(self, target_dim, cond_dim, diffusion_steps=8, hidden_dim=128):
        """
            ARGS:
                target_dim      (int): Dimensionality of the prediction target per node.
                cond_dim        (int): Dimensionality of the encoder conditioning vector.
                diffusion_steps (int): Number of DDPM forward/reverse steps (T).
                hidden_dim      (int): Hidden width of the denoiser MLP.
        """
        super(DiffusionDecoder, self).__init__()    # Call parent class (nn.Module) to track defined layers using 
        
        self.T = diffusion_steps   
        self.target_dim = target_dim   

        self.denoiser = ConditionedDenoiser(
            target_dim, 
            cond_dim, 
            time_dim=32, 
            hidden_dim=hidden_dim
        )

        # Create linear beta schedule: List of noise values added at each of the 'T' diffusion steps. It is 'linear' as values increase in linear increments.
        betas = torch.linspace(1e-4, 0.1, self.T) # Create 'T' equally spaced beta values from 0.0001 to 0.1, where each value represents the amount of new random noise associated with a diffusion step.
        # NOTE: Small beta value: little corruption. Large beta value: stronger corruption.
        
        # betas[t] = noise added at step t.
        # alphas[t] = signal kept at step t.
        # Therefore, more noise = less signal kept
        alphas = 1.0 - betas
        
        # alpha_bar[t] determines the total signal surviving from step 0 all the way through step t.
        # Therefore, alpha_bar is the list of cumulatively decreasing amounts of the signal remaining after adding noise at each diffusion step.
        alpha_bar = torch.cumprod(alphas, dim=0)

        # 'register_buffer' stores the pre-computed schedule tensors inside the model for later, but NOT as trainable parameters
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))   # For q_sample()
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1.0 - alpha_bar))   # For q_sample()

    # FORWARD
    def q_sample(self, x_0, t, noise=None):
        """
            Forward diffusion: corrupt clean targets with Gaussian noise.

            ARGS:
                x_0   (torch.Tensor): Clean targets of shape: [N, target_dim].
                t     (torch.LongTensor): Timestep indices of shape: [N].
                noise (torch.Tensor | None): Pre-sampled noise; generated internally if None.

            RETURNS:
                x_t   (torch.Tensor): Noisy targets of shape: [N, target_dim].
                noise (torch.Tensor): The noise that was added (to compare with what denoiser predicted).
        """
        
        if noise is None:   # If noise was passed externall, use that instead
            noise = torch.randn_like(x_0)   # Generate random Gaussian noise the same shape as 'x_0'
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)  # Value that determines how much of the original signal must be kept
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)  # Value to determine how much random Gaussian noise to add
        
        # At an early diffusion step: Keep most of the target clean and add little noise
        # At a later diffusion step: Barely any of the clean target left and almost complete noise is added
        return sqrt_ab * x_0 + sqrt_omab * noise, noise

    def compute_loss(self, x_0, cond):
        """
            Determines the loss between the actual noise and the denoiser's predicted noise

            ARGS:
                x_0  (torch.Tensor): Ground-truth targets of shape: [N, target_dim].
                cond (torch.Tensor): Conditioning vectors of shape: [N, cond_dim].

            RETURNS:
                torch.Tensor: MSE loss between predicted and true noise.
        """
        N = x_0.size(0) # Number of nodes
        t = torch.randint(0, self.T, (N,), device=x_0.device)   # Randomly assign each node a different timestep
        noise = torch.randn_like(x_0)   # Generate random Gaussian noise for each node
        x_noisy, _ = self.q_sample(x_0, t, noise)   # Corrupt each node's clean target using its assigned timestep and noise with q_sample()
        noise_pred = self.denoiser(x_noisy, t, cond)    # Predict added noise with ConditionedDenoiser
        return F.mse_loss(noise_pred, noise)    # Calculate MSE loss 

    # REVERSE
    @torch.no_grad()    # Do not track gradients, since sample() has no training or backwards passing.
    def sample(self, cond, n_samples=1):
        """
            Reverse diffusion: iteratively denoise from pure Gaussian nois to return prediction(s)

            ARGS:
                cond      (torch.Tensor): Conditioning vectors of shape: [N, cond_dim].
                n_samples (int): How many independent denoising trajectories to run.

            RETURNS:
                torch.Tensor: If n_samples == 1 -> [N, target_dim]. If n_samples >  1 -> [n_samples, N, target_dim].
        """
        N = cond.size(0)    # Number of nodes
        device = cond.device    # Location of 'cond': CPU or GPU

        all_samples = []    # List to collect all plausible forecasts (samples)
        for _ in range(n_samples):
            # Start from pure Gaussian noise 
            x = torch.randn(N, self.target_dim, device=device)  # Create random noise tensor [N, target_dim] (one random number per region) on the same device as 'cond' to avoid mismatch errors

            # Work backwards from T -> 0
            for t_idx in reversed(range(self.T)):
                # Ask denoiser: "Given this noisy value at step t and this region's ATMGNN embedding, what noise is present?"
                t = torch.full((N,), t_idx, device=device, dtype=torch.long)
                noise_pred = self.denoiser(x, t, cond)

                # Look up three schedule values for this step.
                alpha_t = self.alphas[t_idx]
                alpha_bar_t = self.alpha_bar[t_idx]
                beta_t = self.betas[t_idx]

                # Subtract the predicted noise to get a clean estimate 'mu'.
                coef1 = 1.0 / torch.sqrt(alpha_t)   # Since alpha_t is close to 1, coef1 is slightly above 1.
                coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)  # Controls how much of predicted noise to subtract: Large beta_t => Subtract more noise; Large alpha_bar_t => Less signal present. Hence, subtract less noise
                mu = coef1 * (x - coef2 * noise_pred)

                # For all steps except the last, a small fresh random jitter is added. This avoids redundant samples.
                # At the final step t=0, no jitter is added. The clean value 'mu' is accepted as the forecast.
                if t_idx > 0:
                    sigma = torch.sqrt(beta_t)
                    x = mu + sigma * torch.randn_like(x)
                else:
                    x = mu

            all_samples.append(x)
            
        # Return one forecast or a stack of forecasts.
        if n_samples == 1:
            return all_samples[0]
        return torch.stack(all_samples, dim=0)


# Hybrid TGNN-Diffusion Model
class ATMGNN_Diff(ATMGNN):
    """
        Implements the full DiffATMGNN pipeline and outputs a range of plausible forecasts inlcuding uncertainty estimations.
    """

    def __init__(self, nfeat, nhidden, nout, n_nodes, window, dropout, nhead=1, num_clusters=[10, 5], use_norm=False, diffusion_steps=100, decoder_hidden=64):
        """
            ARGS:
                nfeat           (int): Number of input features per node.
                nhidden         (int): Size of the hidden (intermediate) representation or summary vector.
                nout            (int): Size of the final output.
                n_nodes         (int): Total number of nodes (regions) in the graph.
                window          (int): Number of past timesteps used as input.
                dropout         (float): Fraction of neurons randomly turned off during training to prevent overfitting.Dropout rate for regularisation.
                nhead           (int): Number of parallel attention heads.
                num_clusters    (list[int]): Resolution levels for graph coarsening.
                use_norm        (bool): Whether to L2-normalise embeddings before concatenation.
                diffusion_steps (int): Number of DDPM denoising steps (T).
                decoder_hidden  (int): Hidden width of the denoiser MLP.
        """
        super(ATMGNN_Diff, self).__init__(
            nfeat, nhidden, nout, n_nodes, window, dropout,
            nhead, num_clusters, use_norm
        )       # Call parent class (nn.Module) to track defined layers using PyTorch
        
        cond_dim = (len(num_clusters) + 1) * nhidden + window * nfeat       # Conditioning dimension = multi-resolution features (3*nhidden) + raw skip features (window*nfeat).
        
        # Create and store DDPM
        self.diffusion = DiffusionDecoder(
            target_dim=nout, # Output size
            cond_dim=cond_dim, # Size of embedding/conditioniing vector per node (3*nhidden + window*nfeat)
            diffusion_steps=diffusion_steps, # Number of denoising steps
            hidden_dim=decoder_hidden   # Size/width of internal MLP layers
        )

        # Learnable SEIR transition rates
        # NOTE: 'nn.Parameter()' implies that beta and gamma are trainable parameters by PyTorch
        # Additionally, 'log_' prefix is used to indicate that the real rate stays positive after 'softplus'.
        self.log_beta  = nn.Parameter(torch.tensor(0.0))   # transmission rate
        self.log_gamma = nn.Parameter(torch.tensor(-1.0))  # recovery rate
        self._seir_sigma = 1.0 / 5.1                       # fixed (hardcoded) COVID-19 incubation rate (Lauer et. al)
        self._point_forecast_samples = 20                  # number of diffusion draws averaged into the point forecast

    # Inference
    def forward(self, adj, x, n_samples=1):
        """
            Encodes the input and then samples from the diffusion decoder to yield prediction(s)

            ARGS:
                adj       (torch.sparse_coo_tensor): Sparse adjacency matrix describing which nodes are connected.
                x         (torch.Tensor): Flattened node features for all timesteps of shape: [window*n_nodes, nfeat].
                n_samples (int): Number of independent samples for uncertainty estimation.

            RETURNS:
                torch.Tensor: If n_samples == 1 -> [batch*n_nodes] (matches ATMGNN output shape). If n_samples >  1 -> [n_samples, batch*n_nodes].
        """
        cond = self.encode(adj, x) # Inherited from ATMGNN
        
        # Calculate mean for point forecast
        if n_samples == 1:
            samples = self.diffusion.sample(cond, n_samples=self._point_forecast_samples)
            return samples.mean(dim=0).squeeze(-1)
        
        # Diffusion sampling for uncertainty estimation (n_samples > 1).
        samples = self.diffusion.sample(cond, n_samples=n_samples)
        return samples.squeeze(-1)

    # Loss / Traning
    def compute_diffusion_loss(self, adj, x, y_target, node_weights=None):
        """
            Encodes the input then compute three losses:
            
            1) Diffusion loss: Can the denoiser identify Gaussian noise added to the true target?
            
            2) Auxiliary loss: Does the deterministic ATMGNN head predict the observed target?
            
            3) SEIR loss: Is the deterministic forecast reasonably consistent with one SEIR update?
            
            ARGS:
                adj          (torch.sparse_coo_tensor): Sparse adjacency matrix describing which nodes are connected.
                x            (torch.Tensor): Flattened node features for all timesteps of shape: [window*n_nodes, nfeat].
                y_target     (torch.Tensor): Ground-truth target values of shape: [batch*n_nodes].
                node_weights (torch.Tensor | None): Per-node loss weights for scale balancing.

            RETURNS:
                torch.Tensor: Scalar denoising loss value
        """
        cond = self.encode(adj, x)  
        
        if y_target.dim() == 1:
            y_target = y_target.unsqueeze(-1)   # [N] -> [N, 1] because diffusion decoder expects a 2-D output
            
        # Add noise to the target,then predict that noise, and computes MSE between predicted and true noise
        diffusion_loss = self.diffusion.compute_loss(y_target, cond)
        
        # Produce a deterministic single-point forecast
        direct = self.relu(self.fc1(cond))
        direct = self.dropout(direct)
        direct = self.relu(self.fc2(direct)).squeeze(-1).view(-1)   
        
        y_flat = y_target.squeeze(-1).view(-1)  # [N, 1] -> [N] to match 'direct'
        
        # Calculate auxiliary MSE loss between target and predicted value
        # NOTE: If node weights exist, larger regions are penalized less.
        if node_weights is not None:
            w = node_weights.repeat(direct.size(0) // node_weights.size(0))
            aux_loss = (w * (direct - y_flat) ** 2).mean()
        else:
            aux_loss = F.mse_loss(direct, y_flat)

        # Extract fitted SEIR compartments from the most recent timestep's features.
        feat_w = self.nfeat - 5                                       # width of case-count window in features
        x_4d   = x.view(-1, self.window, self.n_nodes, self.nfeat)    # [batch, window, n_nodes, nfeat]
        x_last = x_4d[:, -1, :, :]                                    # [batch, n_nodes, nfeat]

        S_t      = x_last[:, :, feat_w].detach()                      # susceptible proportion
        E_t      = x_last[:, :, feat_w + 1].detach()                  # exposed proportion
        I_t      = x_last[:, :, feat_w + 2].detach()                  # infected proportion
        cur_log  = x_last[:, :, feat_w - 1].detach()                  # log1p(cases) at current day

        beta  = F.softplus(self.log_beta)
        gamma = F.softplus(self.log_gamma)

        # SEIR one-step forward (update)
        dI     = self._seir_sigma * E_t - gamma * I_t + beta * S_t * I_t
        I_next = (I_t + dI).clamp(min=1e-8)
        ratio  = I_next / I_t.clamp(min=1e-8)

        # SEIR (physics-based) prediction in log1p-case space.
        physics_pred = (cur_log + torch.log(ratio.clamp(min=1e-8))).reshape(-1)
        seir_loss    = F.mse_loss(direct, physics_pred)

        return diffusion_loss + 0.1 * aux_loss + 0.05 * seir_loss