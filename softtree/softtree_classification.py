import torch
import torch.nn as nn
import torch.nn.functional as F


# Class defined to calculate the log of probability for each class
class LeafLogMixtureHead(nn.Module):
    """
    Convert path probabilities mu (N, L) into per-class log-probs (N, C).
    L = number of leaves, C = number of classes
    N = number of samples

    Learnable params:
        - leaf_scores: unconstrained (L, C). Class log-probabilites for each class:
        - logQ = log_softmax(leaf_scores, dim=1) for any (-∞, 0], per leaf sums to 1 in prob space.

    Forward:
        log_mu = log(mu)
        logQ   = log_softmax(leaf_scores, dim=1)
        y_log_pro[n, k] = logsumexp_l( log_mu[n, l] + logQ[l, k] )
    """
    def __init__(self, n_leaves: int, n_classes: int, eps: float = 1e-12):
        super().__init__()
        self.eps = eps
        self.leaf_scores = nn.Parameter(torch.randn(n_leaves, n_classes))

    @property
    def logQ(self) -> torch.Tensor:
        """(L, C) per-leaf log-probabilities."""
        # While mathematically equivalent to log(softmax(x)),
        # doing these two operations separately is slower and numerically unstable.
        return F.log_softmax(self.leaf_scores, dim=1)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        """
        mu: (N, L) path probabilities (sum to 1 across leaves per sample)
        returns: (N, C) log-probabilities per class 
        """
        
        # Clamping to a tiny eps: avoids -inf and NaNs
        log_mu = (mu.clamp_min(self.eps)).log()                  # (N, L)
        logQ   = self.logQ                                       
        # log P(Y=k|x) = logsumexp over leaves of log_mu + logQ          
        y_log_probs = torch.logsumexp(log_mu.unsqueeze(-1) + logQ.unsqueeze(0), dim=1)  # (N, C)
        return y_log_probs


class SoftTreeClassifier(nn.Module):
    """Fast implementation of soft decision tree in PyTorch. 
    Parameters
    ----------
    input_dim : int
        The number of input dimensions.
    output_dim : int
        The number of output dimensions. For example, for a multi-class
        classification problem with `K` classes, it is set to `K`.
    depth : int, default=5
        The depth of the soft decision tree. Since the soft decision tree is
        a full binary tree, setting `depth` to a large value will drastically
        increases the training and evaluating cost.
    beta : float, default = 1
        beta = 1/Tempeture, p(x) = sigmoid(beta * (w_T@x + b)).

    Attributes
    ----------
    internal_node_num_ : int
        The number of internal nodes in the tree. Given the tree depth `d`, it
        equals to :math:`2^d - 1`.
    leaf_node_num_ : int
        The number of leaf nodes in the tree. Given the tree depth `d`, it equals
        to :math:`2^d`.
    inner_nodes : torch.nn.Linear
        Linear layer that outputs logits for all internal nodes; sigmoid is applied in forward for probabilistic routing.
    leaf_nodes : torch.nn.Linear
        A `nn.Linear` module that simulates all leaf nodes in the tree.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        depth=5,
        beta=1.0,                
        apply_batchNorm=False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.beta = beta         
        self.apply_batchNorm = bool(apply_batchNorm)
      
        self._validate_parameters()

        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        self.inner_nodes = nn.Linear(self.input_dim, self.internal_node_num_, bias=True)

        # nn.BatchNorm1d implements batch‐normalization for 1D feature vectors. At training time it:
        #     1. Computes the per‐feature mean and variance over our mini-batch.
        #     2. Normalizes each feature to zero mean, unit variance.
        #     3. Applies a learnable affine transform (gamma and beta): gamma*z + beta
        #     4. Updates running estimates of mean/variance for use at inference by track_running_stats.(running mean and running var will use for prediction in model.eval)      
        # See https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html for more details
        if self.apply_batchNorm:
            self.inner_bn = nn.BatchNorm1d(self.internal_node_num_, affine=True, track_running_stats=True)
        else:
            self.inner_bn = None 

        self.leaf_nodes = LeafLogMixtureHead(self.leaf_node_num_, self.output_dim) 

    def forward(self, X):
        X = torch.atleast_2d(X)
        branch_probs = self.get_branch_log_prob(X)          
        y_log_probs = self.leaf_nodes(branch_probs)             # (N, C)   log-probabilities
        y_log_probs = y_log_probs.squeeze(0)
        return y_log_probs

    def get_branch_log_prob(self, X):
        """Derive branch log-prob based on node output"""

        batch_size = X.size()[0]

        linear_outputs = self.inner_nodes(X)         # shape [batch_size, internal_node_num_]
        
        linear_outputs = self.beta * linear_outputs        # apply temperature effect          

        if self.apply_batchNorm:
            linear_outputs = self.inner_bn(linear_outputs)       
      
        # TODO: use logsigmoid(x) and logsimgoid(-x) to get logprobs directly
        sig_outputs = torch.sigmoid(linear_outputs)    # sigmoid - shape [batch_size, internal_node_num_]

        # TODO: why not use sig_outputs.unsqueeze(-1)?
        path_prob = torch.unsqueeze(sig_outputs, dim=2)
        # TODO: if using logsigmoid, combine logsigmoid(x) and logsigmoid(-x)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)

        # TODO: why 3d tensor? 
        # branch probs without any depth (1.0)
        branch_probs = X.new_ones((batch_size, 1, 1))        
        begin_idx = 0
        end_idx = 1

        # identify branches by updating begin and end indices
        for layer_idx in range(0, self.depth):
            current_layer_probs = path_prob[:, begin_idx:end_idx, :]

            # broadcast to get ready for multiplication with latest layer probs
            # TODO: do I need ".view" here?
            branch_probs_expanded = branch_probs.view(batch_size, -1, 1).repeat(1, 1, 2)

            # update branch probabilities
            # TODO: if use logsigmoid, replace "mul" with "add"
            branch_probs = branch_probs_expanded * current_layer_probs

            # update indices to extract next layer probs
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        branch_probs = branch_probs.view(batch_size, self.leaf_node_num_)

        return branch_probs

    def _validate_parameters(self):

        if not self.depth > 0:
            msg = ("The tree depth should be strictly positive, but got {}"
                   "instead.")
            raise ValueError(msg.format(self.depth))
        if not self.input_dim > 0:
            raise ValueError(f"input_dim must be > 0, got {self.input_dim}")

        if not self.output_dim > 0:
            raise ValueError(f"output_dim must be > 0, got {self.output_dim}")

        if not self.beta > 0:
            raise ValueError(f"beta must be > 0, got {self.beta}")