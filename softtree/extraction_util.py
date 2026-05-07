# %%

import torch
import gc
import numpy as np

from jaxtyping import Float
from .softtree_classification import SoftTreeClassifier


def prune_STC_nodes(
    full_STC_model,
    X_train_tensor: Float[torch.Tensor, "_ _"],     # must be float32 flattened features
    pruning_threshold=1e-12,
):

    # extract hyperparameters
    input_dim = full_STC_model.input_dim
    output_dim = full_STC_model.output_dim
    tree_depth = full_STC_model.depth
    beta = full_STC_model.beta

    # extract trainable variables
    flat_weights = full_STC_model.inner_nodes.weight.detach().numpy()
    flat_biases = full_STC_model.inner_nodes.bias.detach().numpy()
    flat_leaf_logits = full_STC_model.leaf_nodes.leaf_scores.detach().numpy()

    # set to zero based on pruning threshold
    flat_weights[np.abs(flat_weights) < pruning_threshold] = 0

    # assert the root node is not pruned
    if np.all(flat_weights[0, :] == 0):
        raise RuntimeError("The root node is pruned!")

    prune_mask = np.array([None] * flat_biases.shape[0])

    idx_to_skip = []
    for i in range(flat_biases.shape[0]):
        w = flat_weights[i, :]

        current_depth = int(np.log2(i+1))
        if np.all(w == 0) and i not in idx_to_skip:
            # child node indices of the current node
            idx_child = get_subtree_index(i, tree_depth)
            idx_leaf = get_leaf_index(i, tree_depth)

            idx_to_skip += list(idx_child)

            # weights and biases and leaves of the subtree from the current node
            subtree_weights = flat_weights[idx_child]
            subtree_biases = flat_biases[idx_child]
            subtree_leaves = flat_leaf_logits[idx_leaf]

            # create a softtree to get label
            stc = SoftTreeClassifier(
                input_dim, output_dim, tree_depth-current_depth, beta,
            )
            # load weights, biases, and leaf logits
            stc.inner_nodes.weight.data = torch.from_numpy(subtree_weights)
            stc.inner_nodes.bias.data = torch.from_numpy(subtree_biases)
            stc.leaf_nodes.leaf_scores.data = torch.from_numpy(subtree_leaves)

            # predicted label
            stc.eval()
            with torch.no_grad():
                y_logp = stc(X_train_tensor)
                _, y_pred = torch.max(y_logp, dim=1)
                y_pred_np = y_pred.detach().numpy()
            del stc
            _flush_memory()

            # get majority label
            counts = np.bincount(y_pred_np)
            majority_label = counts.argmax().item()
            prune_mask[i] = majority_label

    return prune_mask


def get_subtree_index(current_idx, max_depth):

    node_1heap = current_idx + 1

    # Calculate max index for boundary checking
    limit = 2**max_depth - 1

    # Start with the current layer (relative depth 0)
    # The 'width' of the subtree doubles at every layer
    current_start = node_1heap
    current_width = 1

    subtree_indices = []
    while current_start <= limit:
        # Calculate the end of the current row
        current_end = current_start + current_width - 1

        # Clip the end if it exceeds the tree limit
        real_end = min(current_end, limit)

        # Add this range of indices
        # We use range(start, end + 1) to get the values
        subtree_indices.extend(range(current_start, real_end + 1))

        # Move to next layer
        # Start index shifts: new_start = old_start * 2
        current_start *= 2
        # Width doubles: 1 -> 2 -> 4 -> 8
        current_width *= 2
    
    # Convert to 0-based indexing
    subtree_indices = np.array(subtree_indices) - 1

    return subtree_indices


def get_leaf_index(current_idx, max_depth):

    node_1heap = current_idx + 1

    # 1. Calculate current depth
    current_depth = int(np.log2(node_1heap))

    # 2. Check if node is below the tree
    if current_depth > max_depth:
        return []

    # 3. Calculate height difference (how far to the leaves)
    height_diff = max_depth - current_depth
    
    # 4. Calculate the range (Inclusive)
    # The multiplier is 2^height_diff
    multiplier = 2 ** height_diff
    
    start_leaf = node_1heap * multiplier
    end_leaf = (node_1heap + 1) * multiplier - 1
    
    # convert to 0-based indexing starting from 0
    leaf_indices = np.arange(start_leaf, end_leaf + 1)
    leaf_indices = leaf_indices - 2**max_depth

    return leaf_indices


def _flush_memory():
    # run Garbage Collector
    # This forces Python to clean up circular references and actually destroy the object
    gc.collect()

    # clear PyTorch Cache (GPU Only)
    # This releases the memory from PyTorch's internal cache back to the GPU
    torch.cuda.empty_cache()


# %%
if __name__ == '__main__':
    # load data
    imported_data = np.load('data/make_gaussian_1000_seed42.npz')
    X_train, y_train = imported_data['X_train'], imported_data['y_train']
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    
    # load model
    model_hypers = np.load('models/STC_make_gaussian_1000_seed42.npz')
    max_depth = model_hypers['tree_depth'].item()
    input_size = model_hypers['input_size'].item()
    num_classes = model_hypers['num_classes'].item()
    tree_depth = model_hypers['tree_depth'].item()
    beta = model_hypers['beta'].item()

    # load trainable parameters
    loaded_STC_model = SoftTreeClassifier(
        input_dim=input_size,
        output_dim=num_classes,
        depth=tree_depth,
        beta=beta,
        apply_batchNorm=False,
    )

    loaded_STC_model.load_state_dict(torch.load('models/STC_make_gaussian_1000_seed42.pt'))
    loaded_STC_model.eval()

    # # remove weights in node 2 and 3
    # with torch.no_grad():
    #     loaded_STC_model.inner_nodes.weight[1,:] = 0.0
    #     loaded_STC_model.inner_nodes.weight[2,:] = 0.0

    # # load Amir's results
    # amir_results = np.load('models/amir_soft_tree_params_lambda_1.00e-02_d=3_T=1.0.npz')
    # loaded_STC_model.inner_nodes.weight.data = torch.from_numpy(amir_results['weights'])
    # loaded_STC_model.inner_nodes.bias.data = torch.from_numpy(amir_results['bias'])
    # loaded_STC_model.leaf_nodes.leaf_scores.data = torch.from_numpy(amir_results['leaf_log_probs'])
    
    # get prune mask for oblique tree
    prune_mask = prune_STC_nodes(loaded_STC_model, X_train_tensor, pruning_threshold=1e-2)
    np.save('models/STC_make_gaussian_1000_seed42_prune_mask.npy', prune_mask)
    print(prune_mask)

# %%