#%%
import numpy as np
from graphviz import Digraph
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.optimize import linprog


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
        
        if score < 0:
            return self._predict_one(x, node.right)
        else:
            return self._predict_one(x, node.left)


class ParameterizedObliqueTree(BaseEstimator, ClassifierMixin):
    def __init__(
        self, max_depth, flat_weights, flat_biases, flat_leaves,
    ):
        """_summary_

        Args:
            max_depth (_type_): _description_
            flat_weights (_type_): n_node by n_features
            flat_biases (_type_): n_node
            flat_leaves (_type_): n_node+1 class labels in the non-trimmed leaf nodes
        """

        # store data
        self.max_depth = max_depth
        self.flat_weights = flat_weights
        self.flat_biases = flat_biases
        self.flat_leaves = flat_leaves

        # track numbers of internal and leaf nodes
        self.internal_num = 0
        self.leaf_num = 0

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
            self.leaf_num += 1
            return ObliqueNode(node_id, value=val)
        
        # build internal node
        w = self.flat_weights[current_idx]
        b = self.flat_biases[current_idx]
        node = ObliqueNode(node_id, weights=w, bias=b)
        self.internal_num += 1

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
                terms = [f"{w:.2e}x_{i}" for i, w in enumerate(node.weights)]
                equation = " + ".join(terms) + f" + {node.bias:.2e} > 0"
                print(f"{indent}{symbol}Node ({node.id}): [{equation}]")

                # Recurse
                # Add indentation for children
                next_indent = indent + ("    " if branch == "Right" else "│   ")
                if branch == "Root": next_indent = ""

                _print_tree_recursive(node.left, next_indent, "Left")
                _print_tree_recursive(node.right, next_indent, "Right")

        # Start recursion
        _print_tree_recursive(self.root)

    def prune_zero_weight_branches(self):
        """Public method to trigger zero-weight pruning."""
        self.root = self._prune_zero_weights_recursive(self.root)
        self._update_node_num()

    def _prune_zero_weights_recursive(self, node):
        # 1. Base case: Reached a leaf
        if node is None or node.is_leaf:
            return node

        # 2. Recurse bottom-up (clean children before checking parent)
        node.left = self._prune_zero_weights_recursive(node.left)
        node.right = self._prune_zero_weights_recursive(node.right)

        # 3. Check if all weights are effectively zero
        # Using an epsilon tolerance to account for floating point inaccuracies
        if np.all(node.weights == 0.0):
            # Your predict logic: score = dot(x, w) + b. 
            # If score < 0 -> right, else -> left.
            # Since w is 0, score is just b.
            if node.bias < 0:
                # The score is always < 0. Left branch is dead.
                return node.right 
            else:
                # The score is always >= 0. Right branch is dead.
                return node.left 

        return node







    ####################################################################
    def prune_saturated_threshold_branches(self, hi_min=0.0, hi_max=1.0):
        """Prune nodes whose split is logically constant because the routed
        feature  w . x  (a health index, HI) is bounded to [hi_min, hi_max].

        Each oblique node tests  score = w . x + b  with this tree's convention
            score >= 0  -> LEFT  branch   (i.e.  HI > threshold  is TRUE)
            score <  0  -> RIGHT branch   (i.e.  HI > threshold  is FALSE)
        where  HI = w . x  and the threshold is  threshold = -b.

        Because  HI in [hi_min, hi_max]  for every feasible input:
          * threshold > hi_max  (b < -hi_max):  HI can never exceed it, so the
            test is ALWAYS FALSE -> the node always routes RIGHT and is replaced
            by its RIGHT child (the LEFT / TRUE subtree is unreachable).
          * threshold <= hi_min (b >= -hi_min): HI always meets it, so the test
            is ALWAYS TRUE  -> the node always routes LEFT and is replaced by its
            LEFT child (the RIGHT / FALSE subtree is unreachable).

        WARNING: valid ONLY when  w . x  is guaranteed to lie in
        [hi_min, hi_max] for all feasible x. This holds for the BHI/GHI oblique
        nodes (each routes on a normalized health index in [min K, max K]) but
        NOT for a generic oblique tree whose  w . x  is unbounded.
        """
        self.root = self._prune_saturated_threshold_recursive(
            self.root, hi_min, hi_max,
        )
        self._update_node_num()

    def _prune_saturated_threshold_recursive(self, node, hi_min, hi_max):
        # 1. Base case: leaf or empty branch
        if node is None or node.is_leaf:
            return node

        # 2. Recurse bottom-up so children are simplified before the parent
        node.left = self._prune_saturated_threshold_recursive(
            node.left, hi_min, hi_max,
        )
        node.right = self._prune_saturated_threshold_recursive(
            node.right, hi_min, hi_max,
        )

        # 3. Constant-split test.  score = w.x + b = HI + b,  HI in [hi_min, hi_max].
        b = node.bias

        # max score = hi_max + b < 0  ->  always RIGHT  (HI > threshold always FALSE)
        if hi_max + b < 0.0:
            return node.right

        # min score = hi_min + b >= 0  ->  always LEFT   (HI > threshold always TRUE)
        if hi_min + b >= 0.0:
            return node.left

        return node
    ####################################################################


    def prune_infeasible_paths(
        self, epsilon=1e-6,
        A_ub=[], b_ub=[], bounds=(None, None),
        lp_kwargs={"method": "highs"}
    ):
        """Public method to trigger LP-based feasibility pruning."""
        # A_ub * x <= b_ub
        self.root = self._prune_infeasible_recursive(
            self.root,
            epsilon=epsilon,
            A_ub=A_ub, b_ub=b_ub, bounds=bounds,
            lp_kwargs=lp_kwargs
        )
        self._update_node_num()

    def _prune_infeasible_recursive(
        self, node, epsilon,
        A_ub, b_ub, bounds,
        lp_kwargs
    ):
        # Check if the current path is geometrically possible
        if len(A_ub) > 0:
            # We use a dummy objective (minimize 0) just to check feasibility
            c = np.zeros(A_ub[0].shape[0]) 

            # Disable bounds if your inputs can be negative. 
            # If your data is strictly positive (e.g., images), keep bounds=(0, None).
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, **lp_kwargs)

            if not res.success:
                # The polytope is empty. This node and its children are unreachable.
                # We return None, effectively deleting this branch.
                return None

        if node is None or node.is_leaf:
            return node

        # the predict logic:
        # Left child requires: w*x + b >= 0  =>  -w*x <= b
        # Right child requires: w*x + b < 0  =>   w*x <= -b - epsilon

        # 1. Process Left Child (True for w*x + b >= 0)
        A_ub_left = A_ub + [-node.weights]
        b_ub_left = b_ub + [node.bias]
        node.left = self._prune_infeasible_recursive(
            node.left, epsilon,
            A_ub_left, b_ub_left, bounds,
            lp_kwargs
        )

        # 2. Process Right Child (True for w*x + b < 0)
        # We use a small epsilon since linprog only handles <=, not strict <
        A_ub_right = A_ub + [node.weights]
        b_ub_right = b_ub + [-node.bias - epsilon]
        node.right = self._prune_infeasible_recursive(
            node.right,
            epsilon, 
            A_ub_right, b_ub_right, bounds,
            lp_kwargs
        )

        # 3. Post-processing: If a child became None, replace parent with the surviving child
        if node.left is None and node.right is not None:
            return node.right
        elif node.right is None and node.left is not None:
            return node.left

        return node

    def prune_identical_leaves(self):
        """Public method to trigger identical leaf collapse."""
        self.root = self._prune_identical_leaves_recursive(self.root)
        self._update_node_num()

    def _prune_identical_leaves_recursive(self, node):
        # Base case: Reached the bottom or an already established leaf
        if node is None or node.is_leaf:
            return node

        # 1. Recurse bottom-up: process the children first
        node.left = self._prune_identical_leaves_recursive(node.left)
        node.right = self._prune_identical_leaves_recursive(node.right)

        # 2. Check if both surviving children are leaves
        if node.left and node.left.is_leaf and node.right and node.right.is_leaf:
            # 3. Check if they predict the exact same class
            if node.left.value == node.right.value:
                # 4. Collapse this internal node into a leaf
                node.value = node.left.value
                node.weights = None
                node.bias = 0.0
                node.left = None
                node.right = None

        return node

    def _update_node_num(self):
        self.internal_num = self._count_internal_recursive(self.root)
        self.leaf_num = self._count_leaf_recursive(self.root)
    
    def _count_internal_recursive(self, node):
        if node is None:
            return 0
        if node.is_leaf:
            return 0
        return 1 + self._count_internal_recursive(node.left) + self._count_internal_recursive(node.right)
    
    def _count_leaf_recursive(self, node):
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return self._count_leaf_recursive(node.left) + self._count_leaf_recursive(node.right)


# %%