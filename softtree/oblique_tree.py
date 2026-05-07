#%%
import numpy as np
from graphviz import Digraph
from sklearn.base import BaseEstimator, ClassifierMixin


class ObliqueNode:
    def __init__(
        self, id, weights=None, bias=0.0, value=None,
        left=None, right=None,
    ):
        self.id = id
        self.weights = weights  # Vector w
        self.bias = bias        # Scalar b
        self.left = left        # Left Child Node (False path)
        self.right = right      # Right Child Node (True path)
        self.value = value      # Leaf value (Class label)

    @property
    def is_leaf(self):
        return self.value is not None


class CustomObliqueTree(BaseEstimator, ClassifierMixin):
    def __init__(self, root=None):
        self.root = root

    def fit(self, X, y=None):
        # Placeholder to satisfy Sklearn API. 
        # In this application, we build the tree manually from soft tree output.
        return self

    def predict(self, X):
        X = np.array(X)
        predictions = [self._predict_one(x, self.root) for x in X]
        return np.array(predictions)

    def _predict_one(self, x, node):
        # 1. Base Case: If Leaf, return value
        if node.is_leaf:
            return node.value
            
        # 2. Oblique Split Logic: w*x + b > 0
        # This is the "hyperplane" equation
        score = np.dot(x, node.weights) + node.bias
        
        if score > 0:
            return self._predict_one(x, node.right)
        else:
            return self._predict_one(x, node.left)


class ParameterizedObliqueTree(BaseEstimator, ClassifierMixin):
    def __init__(
        self, max_depth,
        flat_weights, flat_biases, flat_leaves,
        prune_mask,
    ):
        """_summary_

        Args:
            max_depth (_type_): _description_
            flat_weights (_type_): n_node by n_features
            flat_biases (_type_): n_node
            flat_leaves (_type_): n_node+1 class labels in the non-trimmed leaf nodes
            prune_mask (_type_): same size as flattened_bias, use -99 as a placeholder for non-trimed nodes
        """

        # store data
        self.max_depth = max_depth
        self.flat_weights = flat_weights
        self.flat_biases = flat_biases
        self.flat_leaves = flat_leaves
        self.prune_mask = prune_mask

        # Pointers to track consumption of the flat arrays during recursion
        self._internal_idx = 0
        self._leaf_idx = 0

        # build tree
        self.root = self._build_recursive(0, "root")

    @classmethod
    def node_id_to_idx(cls, node_id):
        # use 1-based heap indexing
        parts = node_id.split('_')
        
        if parts[0] != 'root':
            raise ValueError("node_id must start with 'root_'")
        
        # 1-based indexing
        node_idx = 1
        
        # iterate though paths
        for part in parts[1:]:
            # shift left by 1 bit
            node_idx = node_idx << 1
            
            if part == "R":
                # Add 1 (equivalent to setting the last bit)
                node_idx = node_idx | 1
            elif part == "L":
                # Add 0 (do nothing, just keep the shift)
                pass
            else:
                raise ValueError(f"Invalid path component: {part}")
            
        node_idx = node_idx - 1

        return node_idx

    def _build_recursive(self, current_depth, node_id):
        current_idx = self.node_id_to_idx(node_id)

        # reached full-tree leave nodes
        if current_depth == self.max_depth:
            leaf_idx = current_idx - self.flat_biases.shape[0]
            val = self.flat_leaves[leaf_idx]
            return ObliqueNode(node_id, value=val)
        
        prune_val = self.prune_mask[current_idx]
        
        # CASE A: PRUNED (Became a leaf)
        if prune_val is not None:
            return ObliqueNode(node_id, value=prune_val)
        
        # CASE B: UNPRUNED
        w = self.flat_weights[current_idx]
        b = self.flat_biases[current_idx]
        node = ObliqueNode(node_id, weights=w, bias=b)

        node.left = self._build_recursive(current_depth+1, f"{node_id}_L")
        node.right = self._build_recursive(current_depth+1, f"{node_id}_R")
        
        return node

    def _predict_one(self, x, node):
        # 1. Base Case: If Leaf, return value
        if node.is_leaf:
            return node.value
            
        # 2. Oblique Split Logic: w*x + b > 0
        # This is the "hyperplane" equation
        score = np.dot(x, node.weights) + node.bias
        
        if score < 0:
            return self._predict_one(x, node.right)
        else:
            return self._predict_one(x, node.left)

    def fit(self, X, y=None):
        # Placeholder to satisfy Sklearn API. 
        # In this application, we build the tree manually from soft tree output.
        return self
    
    def predict(self, X):
        X = np.array(X)
        predictions = [self._predict_one(x, self.root) for x in X]
        return np.array(predictions)
    
    def visualize(self, mode='text', save_path=None):

        if mode == 'graphviz':
            dot = self._visualize_graphviz()
            if save_path is not None:
                dot.save(save_path)
            return dot
        elif mode == 'text':
            return self._visualize_text()
        else:
            raise ValueError(f"Invalid visualization mode: {mode}")     
        
    def _visualize_graphviz(self):
        """
            Generates a Graphviz Digraph for the custom Oblique Tree.
        """
        dot = Digraph()
        dot.attr(rankdir='TB')  # Top to Bottom direction

        def _plot_tree_recursive(node):
            # 1. Unique ID for the graph node
            # We use the node.id string directly as the identifier
            uid = str(node.node_id_to_idx(node.id))

            # 2. Label & Shape
            if node.is_leaf:
                # LEAF NODE
                label = f"Class: {node.value}"
                dot.node(uid, label, shape='box', style='filled', fillcolor='#e2f0cb')
            else:
                # DECISION NODE
                terms = [f"{w:.2f}x_{i}" for i, w in enumerate(node.weights)]
                equation = " + ".join(terms) + f" + {node.bias:.2f} > 0"

                # Formatting the label with a line break
                label = f"Node: {uid}\n{equation}"
                dot.node(uid, label, shape='ellipse', style='filled', fillcolor='#ffd8b1')

                # 3. Recursion for Edges
                # Left Child (False / <= 0)
                if node.left:
                    dot.edge(uid, str(node.left.id), label="False", color="red")
                    _plot_tree_recursive(node.left)

                # Right Child (True / > 0)
                if node.right:
                    dot.edge(uid, str(node.right.id), label="True", color="blue")
                    _plot_tree_recursive(node.right)

        # Start recursion
        _plot_tree_recursive(self.root)
        return dot

    def _visualize_text(self):
        def _print_tree_recursive(node, indent="", branch="Root"):
            """
            Recursively prints the tree structure in the console.
            """

            if not node:
                return

            # Visual marker for the branch
            symbol = "└── " if branch == "Right" else "├── "
            if branch == "Root": symbol = ""

            # Print current node info
            if node.is_leaf:
                print(f"{indent}{symbol}Leaf ({node.id}): Class {node.value}")
            else:
                # Create equation string
                terms = [f"{w:.2f}x_{i}" for i, w in enumerate(node.weights)]
                equation = " + ".join(terms) + f" + {node.bias:.2f} > 0"
                print(f"{indent}{symbol}Node ({node.id}): [{equation}]")

                # Recurse
                # Add indentation for children
                next_indent = indent + ("    " if branch == "Right" else "│   ")
                if branch == "Root": next_indent = ""

                _print_tree_recursive(node.left, next_indent, "Left")
                _print_tree_recursive(node.right, next_indent, "Right")

        # Start recursion
        _print_tree_recursive(self.root)


# %%
if __name__ == "__main__":
    import torch
    import numpy as np
    
    from softtree_classification import SoftTreeClassifier

    # load data
    imported_data = np.load('data/make_gaussian_1000_seed42.npz')
    X_train, y_train = imported_data['X_train'], imported_data['y_train']
    X_val, y_val = imported_data['X_val'], imported_data['y_val']
    X_test, y_test = imported_data['X_test'], imported_data['y_test']
    
    # load model
    model_hypers = np.load('models/STC_make_gaussian_1000_seed42.npz')
    max_depth = model_hypers['tree_depth'].item()
    input_size = model_hypers['input_size'].item()
    num_classes = model_hypers['num_classes'].item()
    tree_depth = model_hypers['tree_depth'].item()
    beta = model_hypers['beta'].item()

    loaded_STC_model = SoftTreeClassifier(
        input_dim=input_size,
        output_dim=num_classes,
        depth=tree_depth,
        beta=beta,
        apply_batchNorm=False,
    )

    loaded_STC_model.load_state_dict(torch.load('models/STC_make_gaussian_1000_seed42.pt'))
    loaded_STC_model.eval()

    weights = loaded_STC_model.inner_nodes.weight.detach().numpy()
    biases = loaded_STC_model.inner_nodes.bias.detach().numpy()
    leaf_logits = loaded_STC_model.leaf_nodes.leaf_scores.detach().numpy()
    leaf_values = np.argmax(leaf_logits, axis=1)

    # # use Amir's data
    # amir_results = np.load('models/amir_soft_tree_params_lambda_1.00e-02_d=3_T=1.0.npz')
    # weights = amir_results['weights']
    # biases = amir_results['bias']
    # leaf_logits = amir_results['leaf_log_probs']
    # leaf_values = np.argmax(leaf_logits, axis=1)

    # prune_mask = np.array(len(biases) * [None])
    prune_mask = np.load('models/STC_make_gaussian_1000_seed42_prune_mask.npy', allow_pickle=True)

    # create oblique tree
    odt_model = ParameterizedObliqueTree(
        max_depth,
        weights, biases, leaf_values,
        prune_mask,
    )
    odt_model.visualize()
    
    # y_pred = odt_model.predict(X_test)
    train_accuracy = odt_model.score(X_train, y_train) 
    val_accuracy = odt_model.score(X_val, y_val) 
    test_accuracy = odt_model.score(X_test, y_test) 
    print(f'Train Accuracy (converted oblique model): {train_accuracy:.4f}')
    print(f'Val Accuracy (converted oblique model): {val_accuracy:.4f}')
    print(f'Test Accuracy (converted oblique model): {test_accuracy:.4f}')

# %%
