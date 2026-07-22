import os
import warnings
import numpy as np
from tqdm import tqdm
from collections import defaultdict


# actor and critic networks to rl modules
import torch
import torch.nn as nn
from torchrl.modules import ProbabilisticActor, ValueOperator
from torch.distributions import Categorical as CategoricalDist
from tensordict.nn import TensorDictModule

# collector and replay buffer
# TODO: use MultiCollector for more complex envs
from torchrl.collectors import Collector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data import TensorSpec

# training
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives import ValueEstimators
from torchrl.envs.utils import set_exploration_type
from torchrl.envs.utils import ExplorationType

# I/O actors, checkpoints, and convert to oblique tree
from .rl_util import ActorNetLogit
from .rl_util import RunningRewardNormalizer
from softtree.oblique_tree import ParameterizedObliqueTree



# Reason: _set_actor_core() needs to reconstruct either normal SoftTreeClassifier or the custom SoftTreeBHI. we have both.
from softtree.bhi_softtree import SoftTreeBHI
from softtree.softtree_classification import SoftTreeClassifier

class PPOTrainer:
    def __init__(
        self,
        env,
        actor_net,
        critic_net,
        config: dict,
    ):
        self.env = env
        self.config = config

        # extract device and fallback to cpu if no device key
        device_str = self.config.get('device', 'cpu')
        self.device = torch.device(device_str)

        # prepare actor and critic
        self.actor = self._setup_actor(self.env.action_spec, actor_net, self.device)
        self.critic = self._setup_critic(critic_net, self.device)

        # prepare collector and replay buffer
        self.collector = self._setup_collector()
        self.replay_buffer = self._setup_replay_buffer()

        # estbalish PPO loss
        self.loss_module = self._setup_loss()

        # prepare Adam optimizer and learning schedule
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.config['learning_rate']
        )
        self.scheduler = self._setup_scheduler()

        # prepare reward scaler
        self.reward_scaler = self._setup_reward_scaler()

    def train(self):
        total_frames = self.config['total_frames']
        frames_per_batch = self.config['frames_per_batch']
        epochs_per_batch = self.config['epochs_per_batch']
        frames_per_minibatch = self.config['frames_per_minibatch']

        eval_freq = self.config.get('eval_freq', 0)
        eval_episodes = self.config.get('eval_episodes', 10)
        eval_deterministic = self.config.get('eval_deterministic', True)

        train_log = defaultdict(list)
        eval_log = defaultdict(list)

        with tqdm(total=total_frames) as pbar:
            for i, td_data in enumerate(self.collector):
                # scale reward
                if self.reward_scaler is not None:
                    td_data = self.reward_scaler(td_data)
                
                # estimate GAE
                with torch.no_grad():
                    self.loss_module.value_estimator(td_data)

                # load to replay buffer
                self.replay_buffer.extend(td_data)












                # # Original code
                # # training weights
                # for _ in range(epochs_per_batch):
                #     minibatch = self.replay_buffer.sample(frames_per_minibatch)

                #     # forward loss
                #     loss_vals = self.loss_module(minibatch.to(self.device))
                #     loss_total = loss_vals["loss_objective"] \
                #         + loss_vals["loss_entropy"] \
                #         + loss_vals["loss_critic"]
                    
                #     loss_total += self._add_regularization_loss()

                #     # backprop
                #     self.optimizer.zero_grad()
                #     loss_total.backward()
                    
                #     if self.config.get('max_grad_norm') is not None:
                #         torch.nn.utils.clip_grad_norm_(
                #             self.loss_module.parameters(), self.config['max_grad_norm']
                #         )
                #     self.optimizer.step()


                if frames_per_batch % frames_per_minibatch != 0:
                    raise ValueError("frames_per_batch must be divisible by frames_per_minibatch.")

                num_minibatches = frames_per_batch // frames_per_minibatch

                for _ in range(epochs_per_batch):
                    for _ in range(num_minibatches):
                        minibatch = self.replay_buffer.sample(frames_per_minibatch)

                        loss_vals = self.loss_module(minibatch.to(self.device))
                        loss_total = (
                            loss_vals["loss_objective"]
                            + loss_vals["loss_entropy"]
                            + loss_vals["loss_critic"]
                            + self._add_regularization_loss()
                        )

                        self.optimizer.zero_grad()
                        loss_total.backward()

                        if self.config.get("max_grad_norm") is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.loss_module.parameters(),
                                self.config["max_grad_norm"],
                            )

                        self.optimizer.step()























                # evaluate policy
                if eval_freq != 0 and (i % eval_freq == 0 or i == total_frames//frames_per_batch - 1):
                    eval_log_i = self.evaluate(
                        self.actor, self.env,
                        num_episodes=eval_episodes,
                        max_steps=self.env._env.max_steps,
                        deterministic=eval_deterministic
                    )
                    eval_log['batch'].append(i)
                    eval_log['eval_reward'].append(np.mean(eval_log_i['eval_reward']))

                # update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()

                # update other state parameters (nothing in NN; beta in softtree with annealing)
                self._update_state_params(i)

                # update progress bar
                pbar.update(frames_per_batch)

                train_reward_per_step = td_data['next', 'reward'].mean().item()
                train_reward = train_reward_per_step * self.env._env.max_steps

                if eval_log.get('eval_reward') is not None:
                    eval_reward = eval_log['eval_reward'][-1]
                else:
                    eval_reward = np.nan
            
                pbar_str = "| ".join([
                    f"reward: {train_reward:.4e}",
                    f"reward (eval): {eval_reward:.4e}",
                ])
                pbar.set_description(pbar_str)

                # # update log
                # train_log['batch'].append(i)
                # train_log['reward'].append(train_reward)
        
                # update log
                train_log['batch'].append(i)
                train_log['reward'].append(train_reward)

                # optional architecture-specific diagnostics
                # For SoftTreeBHI, this logs how training rollout samples are distributed
                # over tree leaves. This is different from validation leaf counting.
                self._append_custom_train_diagnostics(td_data, train_log)






        return train_log, eval_log
    
    @classmethod
    @torch.no_grad()
    def evaluate(cls, actor, eval_env, num_episodes=1, max_steps=1, deterministic=True):
        """
        Evaluates a policy
        """
        actor.eval()
        exploration_type = ExplorationType.MODE if deterministic else ExplorationType.RANDOM
        
        eval_log = defaultdict(list)
        with set_exploration_type(exploration_type):
            for i in range(num_episodes):
                eval_data = eval_env.rollout(
                    max_steps=max_steps,
                    policy=actor,
                    auto_cast_to_device=True,
                    break_when_any_done=True
                )

                eval_log["eval_trial"].append(i)
                eval_log["init_state"].append(eval_data["observation"][0].detach().cpu().numpy())
                eval_log["eval_reward"].append(eval_data["next", "reward"].sum().item())


                # Save all observations visited during this episode(I need this for visiting table in validation)
                eval_log["observations"].append(eval_data["observation"].detach().cpu().numpy())

                # Save reward corresponding to each visited observation/action(I need this for visiting table in validation)
                eval_log["step_rewards"].append(eval_data["next", "reward"].detach().cpu().numpy().reshape(-1))





        actor.train()

        return eval_log

    def save_checkpoint(self, filepath):
        """Save checkpint to pickup training.
           
           Do not use it to save policy. Use `save_actor` instead.
        """
        # create file if not exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # basic checkpoint
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        # include the scheduler if it exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # save user defined state
        user_state = self._get_state_params()
        checkpoint.update(user_state)
            
        torch.save(checkpoint, filepath)

        print(f"[*] Checkpoint saved successfully to {filepath}")

    def load_checkpoint(self, filepath):
        # check filepath must exist
        if not os.path.exists(filepath):
            raise ValueError(f"Path {filepath} does not exist")

        # load the dictionary to the correct device
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # sestore scheduler if applicable
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # restore user defined state
        self._set_state_params(checkpoint)
            
        print(f"[*] Checkpoint loaded successfully from {filepath}")

    def save_actor(self, filepath):
        # create file if not exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        actor_core_state = self.actor.module[0].module.state_dict()
        actor_core_hyperparams = self._get_actor_core_hyperparams()

        save_dict = {
            'actor_core_state': actor_core_state,
            'actor_core_hyperparams': actor_core_hyperparams,
        }

        torch.save(save_dict, filepath)

        print(f"[*] Actor saved successfully to {filepath}")
    
    @classmethod
    @torch.no_grad()
    def load_actor(cls, filepath, action_spec, device=torch.device("cpu")):
        # check filepath must exist
        if not os.path.exists(filepath):
            raise ValueError(f"Path {filepath} does not exist")

        load_dict = torch.load(filepath, map_location=device)

        actor_net = cls._set_actor_core(
            load_dict['actor_core_state'],
            load_dict['actor_core_hyperparams']
        )
        actor = cls._setup_actor(action_spec, actor_net, device=device)

        print(f"[*] Actor loaded successfully from {filepath}")

        return actor

    @classmethod
    def _setup_actor(cls, action_spec, actor_net, device=torch.device("cpu")):
        actor_module = TensorDictModule(actor_net, in_keys=['observation'], out_keys=['logits'])

        actor = ProbabilisticActor(
            module=actor_module,
            spec=action_spec,
            distribution_class=CategoricalDist,
            in_keys=['logits'],  # Key in the input tensor containing the observation
            out_keys=['action'],  # Key where the sampled action will be written
            return_log_prob=True,
        ).to(device)

        return actor
    
    @classmethod
    def _setup_critic(cls, critic_net, device=torch.device("cpu")):
        return ValueOperator(module=critic_net, in_keys=['observation']).to(device)
    
    def _setup_collector(self):
        # make sure if total_frames is not -1, it mus be divisible by frames_per_batch
        if self.config['total_frames'] != -1:
            assert self.config['total_frames'] % self.config['frames_per_batch'] == 0

        collector = Collector(
            create_env_fn=self.env,
            policy=self.actor,
            frames_per_batch=self.config['frames_per_batch'],
            total_frames=self.config['total_frames'],
            split_trajs=self.config.get('split_trajs', False),
            device=self.device
        )

        return collector
    
    def _setup_replay_buffer(self):
        # max_size must be frames_per_batch due to on-policy learning
        # use it for mini-batch shuffler and data loader
        return ReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.config['frames_per_batch'],
                device=self.device
            ),
            sampler=SamplerWithoutReplacement(),
        )
    
    def _setup_loss(self):
        """
        Create PPO loss module
        
        loss_module uses defaults from torchrl v0.11

        GAE parameters
        - gamma normally is set to 1 to avoid double counting gamma in the env
        - lmbda should be between 0 and 1
          lmbda=0 is equivalent to using TD0
          lmbda=1 is equivalent to using TD1
          default is set to 1.0 (full episode estimate of value)
        - average_GAE defaults to False based on torchrl v0.11 (True may be more robust)
        """
        loss_module = ClipPPOLoss(
            actor_network=self.actor,
            critic_network=self.critic,
            clip_epsilon=self.config.get('clip_epsilon', 0.2),
            entropy_bonus=bool(self.config.get('entropy_eps', 0.0)),
            entropy_coeff=self.config.get('entropy_eps', 0.0),
            critic_coeff=self.config.get('critic_coef', 1.0),
            loss_critic_type=self.config.get('loss_critic_type', "smooth_l1"),
        ).to(self.device)

        loss_module.make_value_estimator(
            ValueEstimators.GAE,
            gamma=self.config.get('GAE_gamma', 1.0),
            lmbda=self.config.get('GAE_lmbda', 1.0),
            average_gae=self.config.get('average_GAE', False),
        )

        return loss_module

    def _setup_scheduler(self):
        scheduler_type = self.config.get('scheduler_type', None)
        total_iterations = self.config['total_frames'] // self.config['frames_per_batch']
        
        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=total_iterations,
                eta_min=self.config.get('lr_min', 0.0)
            )
        elif scheduler_type == 'step':
            step_size = self.config.get('step_size')
            if step_size is None:
                raise ValueError("step_size must be specified for step scheduler")

            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=step_size,
                gamma=self.config.get('step_gamma', 0.1)
            )
        else:
            warnings.warn(f"Unknown scheduler type: {scheduler_type}; using None scheduler")
            return None # No scheduler
    
    def _setup_reward_scaler(self):
        if self.config.get('reward_decay') is None:
            return None
        else:
            return RunningRewardNormalizer(
                decay=self.config['reward_decay'],
                device=self.device
            )


    def _get_actor_core_hyperparams(self):
        params_dict = {
            "input_dim": self.actor.module[0].module.layers[0].in_features,
            "output_dim": self.actor.module[0].module.layers[-1].out_features,
            "actor_cells": self.actor.module[0].module.layers[0].out_features,
            # "actor_layers": (len(self.actor.module[0].module.layers)-1)//2-1,
            "actor_layers": (len(self.actor.module[0].module.layers)-1)//2,

        }

        return params_dict

    @classmethod
    @torch.no_grad()
    def _set_actor_core(cls, state_dict, params_dict):
        actor_net = ActorNetLogit(
            input_dim=params_dict["input_dim"],
            output_dim=params_dict["output_dim"],
            actor_cells=params_dict["actor_cells"],
            actor_layers=params_dict["actor_layers"],
        )
        actor_net.load_state_dict(state_dict)
        return actor_net






    @torch.no_grad()
    def _append_custom_train_diagnostics(self, td_data, train_log):
        """
        Log leaf usage of the soft-tree actor on the actual PPO rollout batch.

        This answers:

            During training, do rollout observations reach many leaves,
            or does the tree mostly use one leaf?

        We log two diagnostics:

        1. hard_leaf_*:
        Force each observation through the current soft tree using hard
        left/right routing based on node score >= 0.

        This is close to the later oblique-tree routing logic.

        2. soft_leaf_*:
        Use the soft path probabilities from the soft tree.

        This is closer to how the soft tree actually trains.
        """
        actor_core = self.actor.module[0].module

        # Only soft-tree actors have these attributes.
        if not hasattr(actor_core, "get_branch_log_prob"):
            return

        if not hasattr(actor_core, "leaf_node_num_"):
            return

        obs = td_data["observation"].detach().to(self.device)

        # Flatten possible trajectory/batch dimensions into one sample dimension.
        obs = obs.reshape(-1, obs.shape[-1])

        if obs.numel() == 0:
            return

        was_training = actor_core.training
        actor_core.eval()

        # ------------------------------------------------------------
        # 1. Soft expected leaf occupancy
        # ------------------------------------------------------------
        # get_branch_log_prob is badly named in your code; it returns path
        # probabilities, not log-probabilities.
        leaf_probs = actor_core.get_branch_log_prob(obs)  # shape: (N, L)

        mean_leaf_prob = leaf_probs.mean(dim=0)           # shape: (L,)
        soft_top_leaf_prob = mean_leaf_prob.max().item()
        soft_top_leaf = int(mean_leaf_prob.argmax().item())

        # Effective number of leaves:
        #   1 / sum(p_l^2)
        # If all mass is on one leaf -> 1.
        # If mass is uniform over 64 leaves -> 64.
        soft_effective_leaves = (
            1.0 / mean_leaf_prob.pow(2).sum().clamp_min(1e-12)
        ).item()

        # ------------------------------------------------------------
        # 2. Hard-routed leaf occupancy
        # ------------------------------------------------------------
        node_scores = actor_core.inner_nodes(obs)  # shape: (N, internal_nodes)

        n_samples = obs.shape[0]
        depth = actor_core.depth
        n_leaves = actor_core.leaf_node_num_

        sample_ids = torch.arange(n_samples, device=obs.device)
        node_idx = torch.zeros(n_samples, dtype=torch.long, device=obs.device)
        leaf_idx = torch.zeros(n_samples, dtype=torch.long, device=obs.device)

        for _ in range(depth):
            score = node_scores[sample_ids, node_idx]

            # Same convention as oblique extraction:
            # score >= 0 -> left branch
            # score <  0 -> right branch
            go_right = score < 0

            leaf_idx = 2 * leaf_idx + go_right.long()
            node_idx = 2 * node_idx + 1 + go_right.long()

        leaf_counts = torch.bincount(
            leaf_idx,
            minlength=n_leaves,
        ).float()

        total = leaf_counts.sum().clamp_min(1.0)
        hard_top_leaf = int(leaf_counts.argmax().item())
        hard_top_leaf_pct = float(100.0 * leaf_counts.max().item() / total.item())
        hard_unique_leaves = int((leaf_counts > 0).sum().item())

        # ------------------------------------------------------------
        # 3. Optional action usage from actual PPO rollout
        # ------------------------------------------------------------
        if "action" in td_data.keys():
            action = td_data["action"].detach().reshape(-1).to(torch.long)
            action_counts = torch.bincount(
                action.cpu(),
                minlength=actor_core.output_dim,
            ).float()

            action_total = action_counts.sum().clamp_min(1.0)
            top_action = int(action_counts.argmax().item())
            top_action_pct = float(100.0 * action_counts.max().item() / action_total.item())
        else:
            top_action = -1
            top_action_pct = np.nan

        train_log["hard_unique_leaves"].append(hard_unique_leaves)
        train_log["hard_top_leaf"].append(hard_top_leaf)
        train_log["hard_top_leaf_pct"].append(hard_top_leaf_pct)

        train_log["soft_effective_leaves"].append(soft_effective_leaves)
        train_log["soft_top_leaf"].append(soft_top_leaf)
        train_log["soft_top_leaf_prob"].append(soft_top_leaf_prob)

        train_log["rollout_top_action"].append(top_action)
        train_log["rollout_top_action_pct"].append(top_action_pct)

        if was_training:
            actor_core.train()














    def _add_regularization_loss(self):
        l1_coef = self.config.get('actor_l1_coef', 0.0)
        l2_coef = self.config.get('actor_l2_coef', 0.0)

        loss = 0.0
        
        if l1_coef > 0.0:
            l1_loss = sum(p.abs().sum() for p in self.actor.parameters())
            loss += l1_coef * l1_loss

        if l2_coef > 0.0:
            l2_loss = sum(p.pow(2).sum() for p in self.actor.parameters())
            loss += l2_coef * l2_loss

        return loss

    def _update_state_params(self, batch_index):
        pass

    def _get_state_params(self):
        return {}
    
    def _set_state_params(self, state_dict):
        pass


class SofttreePPOTrainer(PPOTrainer):
    def __init__(self, env, actor_tree, critic_net, config: dict):
        super().__init__(env, actor_tree, critic_net, config)
    
    @classmethod
    @torch.no_grad()
    def convert_to_obtree_actor(
        cls, actor,
        pruning_threshold,
        lp_threshold=1e-6,
        # A_ub=[], b_ub=[], bounds=(None, None),
        A_ub=[], b_ub=[], A_eq=[], b_eq=[], bounds=(None, None),
        enforce_simplex=True,        
        lp_kwargs={"method": "highs"}
    ):
        STC_core = actor.module[0].module
        max_depth = STC_core.depth















        ######################################################################
        # First change out of five changes in this file
        # original
        # weights = STC_core.inner_nodes.weight.detach().numpy()
        # biases = STC_core.inner_nodes.bias.detach().numpy()
        # leaf_logits = STC_core.leaf_nodes.leaf_scores.detach().numpy()
        # leaf_values = np.argmax(leaf_logits, axis=1)
 
 
        # But the oblique tree class does not know what BHI is.In the follwoing we have both version(after else is for normal soft tree)
        biases = STC_core.inner_nodes.bias.detach().cpu().numpy()
        leaf_logits = STC_core.leaf_nodes.leaf_scores.detach().cpu().numpy()
        leaf_values = np.argmax(leaf_logits, axis=1)
 
        # ================================================================
        # Per-node GHI weight extraction (Solution 1).
        #
        # In the soft actor each node routes on phi_n = sum_k p_n(k)*HI_k.
        # To turn that into a single oblique hyperplane w_n . x + b_n, we take
        # the node's MOST-LIKELY HI (argmax of its selection logits) and write
        # the linear feature-weight vector that reproduces that one HI as a
        # function of the raw observation x (the flattened CS probabilities).
        #
        # For a group-level HI_k:
        #     HI_k(x) = sum_{i in group k} ( w_i / sum_{j in group k} w_j ) * (CS_i . K)
        # so the coefficient on observation entry (element i, condition state s)
        # is  (w_i * K_s) / sum_{j in group k} w_j   for i in group k, else 0.
        #
        # For the aggregate BHI (k = 4) the group is ALL elements, so the
        # normalizer is the full weight sum.
        # ================================================================
        # HI-structured tree? Only the per-node GHI/BHI selector guarantees that
        # each node's  w . x  is a normalized health index bounded to [min K, max K];
        # the saturated-threshold prune below relies on that bound.
        is_hi_tree = hasattr(STC_core.inner_nodes, "selection_logits")
        hi_min, hi_max = 0.0, 1.0

        if hasattr(STC_core.inner_nodes, "selection_logits"):
            inner = STC_core.inner_nodes

            learned_w = torch.nn.functional.softplus(
                inner.raw_element_weights
            ).detach().cpu().numpy()                                   # (E,)
            K = inner.health_coefficients.detach().cpu().numpy()       # (NCS,)
            hi_min, hi_max = float(K.min()), float(K.max())            # HI range = [min K, max K]
            group_idx = inner.element_to_group_idx.detach().cpu().numpy()  # (E,)
            selected_k = inner.selection_logits.argmax(dim=1).detach().cpu().numpy()  # (nodes,) # This is for extracted oblique tree
 
            num_elements = inner.num_elements
            ncs = inner.ncs
            n_obs = num_elements * ncs + int(inner.include_step_count)
            w_sum_all = learned_w.sum()
 
            weights = np.zeros((inner.num_nodes, n_obs), dtype=float)
 
            for n in range(inner.num_nodes):
                k = int(selected_k[n])
                if k < 4:
                    member = np.where(group_idx == k)[0]              # elements in group k
                    norm = learned_w[member].sum()
                else:
                    member = np.arange(num_elements)                 # aggregate BHI: all
                    norm = w_sum_all
                norm = norm if norm > 1e-8 else 1e-8
                for i in member:
                    for s in range(ncs):
                        weights[n, i * ncs + s] = (learned_w[i] * K[s]) / norm # n is the node index, i is the element index, s is the condition state index
                # step-count column (if any) stays 0: it never enters any HI
 
 
        else:
            # normal soft tree (not a BHI/GHI tree)
            weights = STC_core.inner_nodes.weight.detach().cpu().numpy()
        ######################################################################



    













        ratios = np.divide(
            weights,
            biases[:, np.newaxis],
            where=biases[:, np.newaxis] != 0
        )
        
        prune_mask = np.all(np.abs(ratios) <= pruning_threshold, axis=1)
        weights[np.abs(ratios) <= pruning_threshold] = 0.0

        odt_model = ParameterizedObliqueTree(
            max_depth, weights, biases, leaf_values,
        )
        odt_model.prune_zero_weight_branches()


        ##############################################
        # New rule: drop branches that are logically unreachable because each
        # HI-node tests  HI > threshold  with  HI in [hi_min, hi_max].
        #   threshold > hi_max  -> always FALSE -> keep only the FALSE (right) child
        #   threshold <= hi_min -> always TRUE  -> keep only the TRUE  (left)  child
        if is_hi_tree: # we meed to check if this is normal soft tree(like in single element project) we don't need to prune saturated threshold branches
            odt_model.prune_saturated_threshold_branches(
                hi_min=hi_min, hi_max=hi_max
            )
        ##############################################



        ##############################################
        # Per-element simplex equality constraints for the LP feasibility prune.
        # Every real observation obeys  sum_s x[i, s] = 1  for each element i (the
        # condition-state probabilities of one element must sum to 1). Adding these
        # equalities makes prune_infeasible_paths delete branches that are only
        # feasible inside the raw [0,1] box but impossible for an actual bridge
        # state -- e.g. paths that would need two health indices to be simultaneously
        # out of their jointly-reachable range. The step-count column (if any) is
        # left unconstrained by these equalities.
        A_eq = list(A_eq)
        b_eq = list(b_eq)
        if enforce_simplex and is_hi_tree:
            n_obs = weights.shape[1]
            ncs_local = STC_core.inner_nodes.ncs
            num_elem_local = STC_core.inner_nodes.num_elements
            for i in range(num_elem_local):
                row = np.zeros(n_obs, dtype=float)
                row[i * ncs_local:(i + 1) * ncs_local] = 1.0
                A_eq.append(row)
                b_eq.append(1.0)
        ##############################################






        odt_model.prune_infeasible_paths(
            epsilon=lp_threshold,
            # A_ub=A_ub, b_ub=b_ub, bounds=bounds, 
            A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
            lp_kwargs=lp_kwargs
        )
        odt_model.prune_identical_leaves()

        odt_module = ObliqueTreePolicy(odt_model)

        # wrap to torchrl (_setup_actor not applicable sine odt doesn't give logits)
        odt_actor = TensorDictModule(
            odt_module, in_keys=['observation'], out_keys=['action']
        )

        return odt_actor, prune_mask


    #######################################################
    # Second change out of five changes in this file
    # original code 
    # def _get_actor_core_hyperparams(self):
    #     params_dict = {
    #         "input_dim": self.actor.module[0].module.input_dim,
    #         "output_dim": self.actor.module[0].module.output_dim,
    #         "depth": self.actor.module[0].module.depth,
    #         "beta": self.actor.module[0].module.beta
    #     }
    #     return params_dict
 
 
    # Reason: now the saved actor file knows whether it is a normal soft tree(for single element) or BHI-soft tree.
    def _get_actor_core_hyperparams(self):
        # Normal soft tree
        actor_core = self.actor.module[0].module
 
        params_dict = {
            "input_dim": actor_core.input_dim,
            "output_dim": actor_core.output_dim,
            "depth": actor_core.depth,
            "beta": actor_core.beta,
            "actor_type": "SoftTreeClassifier",
        }

        #  BHI-soft tree with per-node GHI selector (Solution 1) or legacy shared-BHI selector
        if hasattr(actor_core.inner_nodes, "selection_logits"):
            # This is the per-node GHI actor (Solution 1).
            inner = actor_core.inner_nodes
            params_dict["actor_type"] = "SoftTreeBHI"   # same class name on load
            params_dict["num_elements"] = inner.num_elements
            params_dict["ncs"] = inner.ncs
            params_dict["include_step_count"] = inner.include_step_count
            params_dict["health_coefficients"] = (
                inner.health_coefficients.detach().cpu().tolist()
            )
            params_dict["initial_element_weights"] = (
                torch.nn.functional.softplus(
                    inner.raw_element_weights
                ).detach().cpu().tolist()
            )
            # The following are the new parameters for per-node GHI selector (Solution 1)
            params_dict["element_to_group_idx"] = (
                inner.element_to_group_idx.detach().cpu().tolist()
            )
            params_dict["tau"] = float(inner.tau)
            # LEARNABLE_SIGNIFICANCE_FACTOR toggle: whether the element weights
            # were trained (True) or kept fixed at ELEMENT_WEIGHTS (False).
            # Saved so load_actor() rebuilds the actor in the same mode.
            params_dict["learnable_element_weights"] = getattr(
                inner, "learnable_element_weights", True
            )
 
        # BHI-soft tree with legacy shared-BHI selector (kept for backward compatibility)
        elif hasattr(actor_core.inner_nodes, "raw_element_weights"):
            # legacy shared-BHI actor (kept for backward compatibility)
            params_dict["actor_type"] = "SoftTreeBHI_legacy_shared"
            params_dict["num_elements"] = actor_core.inner_nodes.num_elements
            params_dict["ncs"] = actor_core.inner_nodes.ncs
            params_dict["include_step_count"] = actor_core.inner_nodes.include_step_count
            params_dict["health_coefficients"] = (
                actor_core.inner_nodes.health_coefficients.detach().cpu().tolist()
            )
            params_dict["initial_element_weights"] = (
                torch.nn.functional.softplus(
                    actor_core.inner_nodes.raw_element_weights
                ).detach().cpu().tolist()
            )
 
        return params_dict
    ########################################################################     


    ##################################################################
    # Third change out of five changes in this file
    # Original
    # @classmethod
    # @torch.no_grad()
    # def _set_actor_core(cls, state_dict, params_dict):
    #     actor_tree = SoftTreeClassifier(
    #         input_dim=params_dict['input_dim'],
    #         output_dim=params_dict['output_dim'],
    #         depth=params_dict['depth'],
    #         beta=params_dict['beta']
    #     )
    #     actor_tree.load_state_dict(state_dict)
    #     return actor_tree
    
 
 
    # Reason: now load_actor() will correctly rebuild both SoftTreeClassifier and SoftTreeBHI.
    # and we don't need to rebuild the whole actor in validation files. We already defined
    # actor_type in the saved state dictionary in _get_actor_core_hyperparams(). 
    @classmethod
    @torch.no_grad()
    def _set_actor_core(cls, state_dict, params_dict):
        actor_type = params_dict.get("actor_type", "SoftTreeClassifier")
 
        if actor_type == "SoftTreeBHI":
            # Per-node GHI actor (Solution 1).
            actor_tree = SoftTreeBHI(
                input_dim=params_dict["input_dim"],
                output_dim=params_dict["output_dim"],
                depth=params_dict["depth"],
                beta=params_dict["beta"],
                num_elements=params_dict["num_elements"],
                ncs=params_dict["ncs"],
                health_coefficients=params_dict["health_coefficients"],
                initial_element_weights=params_dict["initial_element_weights"],
                element_to_group_idx=params_dict["element_to_group_idx"],
                include_step_count=params_dict.get("include_step_count", False),
                tau_init=params_dict.get("tau", 1.0),
                apply_batchNorm=False,
                # Rebuild in the same LEARNABLE_SIGNIFICANCE_FACTOR mode the
                # actor was trained with (defaults to True for old saves).
                learnable_element_weights=params_dict.get(
                    "learnable_element_weights", True
                ),
            )
 
        elif actor_type == "SoftTreeClassifier":
            actor_tree = SoftTreeClassifier(
                input_dim=params_dict["input_dim"],
                output_dim=params_dict["output_dim"],
                depth=params_dict["depth"],
                beta=params_dict["beta"],
            )
 
        else:
            raise ValueError(f"Unknown actor_type: {actor_type}")
 
        actor_tree.load_state_dict(state_dict)
        actor_tree.eval()
 
        return actor_tree
    ################################################################## 


    def _add_regularization_loss(self):
        l1_coef = self.config.get('actor_l1_coef', 0.0)
        l2_coef = self.config.get('actor_l2_coef', 0.0)
        gl1_coef = self.config.get('actor_gl1_coef', 0.0)

        loss = 0.0
        weights = self.actor.module[0].module.inner_nodes.weight
        # weights = torch.cat([
        #     self.actor.module[0].module.inner_nodes.weight.view(-1),
        #     self.actor.module[0].module.inner_nodes.bias.view(-1)
        # ])

        if l1_coef > 0.0:
            l1_loss = weights.abs().sum()
            loss += l1_coef * l1_loss

        if l2_coef > 0.0:
            l2_loss = weights.pow(2).sum()
            loss += l2_coef * l2_loss
        
        if gl1_coef > 0.0:
            gl1_loss = weights.pow(2).sum(dim=1).sqrt().sum()
            loss += gl1_coef * gl1_loss

        return loss
    






    def _append_custom_train_diagnostics(self, td_data, train_log):
        """
        Optional hook for architecture-specific diagnostics.

        Base PPOTrainer does nothing. SofttreePPOTrainer overrides this to log
        soft-tree leaf usage during training.
        """
        return













    ##################################################################
    # forth change out of five changes in this file
    # Reason: we need to anneal both beta and tau in the BHI-soft tree actor.
    # we had beta annealing before, but now we also need to anneal tau (per-node GHI selection temperature).
    def _update_state_params(self, batch_index):
        actor_core = self.actor.module[0].module
 
        # ----- beta annealing (routing sharpness) -- unchanged from before ---
        beta_anneal = self.config.get('beta_anneal', 1.0)
        beta_update_freq = self.config.get('beta_update_freq', 1)
        if beta_anneal > 1.0 and (batch_index + 1) % beta_update_freq == 0:
            actor_core.beta *= beta_anneal
 
        # ----- tau annealing (per-node HI selection temperature) ------
        # Only runs if the actor exposes tau (i.e. the per-node GHI actor).
        # tau is DIVIDED by tau_anneal each step so the softmax sharpens toward
        # one-hot. tau_anneal > beta_anneal is recommended so selection commits
        # before routing fully hardens. Floored at tau_min to
        # keep the backward softmax numerically well-behaved.
        if hasattr(actor_core, 'tau'):
            tau_anneal = self.config.get('tau_anneal', 1.0)
            tau_update_freq = self.config.get('tau_update_freq', 1)
            tau_min = self.config.get('tau_min', 0.01)
            if tau_anneal > 1.0 and (batch_index + 1) % tau_update_freq == 0:
                actor_core.tau = max(actor_core.tau / tau_anneal, tau_min)
 
        return super()._update_state_params(batch_index)
    ##################################################################














    ###########################################################################
    # fifth change out of five changes in this file to save and load the state of tau in addition to beta.
    # def _get_state_params(self):
    #     state_dict = {
    #         'current_beta': self.actor.module[0].module.beta,
    #         'beta_anneal': self.config.get('beta_anneal', 1.0),
    #         'beta_update_freq': self.config.get('beta_update_freq', 1),
    #     }
    #     return state_dict
    
    # def _set_state_params(self, state_dict):
    #     self.config.update(state_dict)
    #     self.actor.module[0].module.beta = state_dict['current_beta']

    
    def _get_state_params(self):
        actor_core = self.actor.module[0].module
        state_dict = {
            'current_beta': actor_core.beta,
            'beta_anneal': self.config.get('beta_anneal', 1.0),
            'beta_update_freq': self.config.get('beta_update_freq', 1),
        }
        # persist tau too, if present (per-node GHI actor)
        if hasattr(actor_core, 'tau'):
            state_dict['current_tau'] = actor_core.tau
            state_dict['tau_anneal'] = self.config.get('tau_anneal', 1.0)
            state_dict['tau_update_freq'] = self.config.get('tau_update_freq', 1)
            state_dict['tau_min'] = self.config.get('tau_min', 0.01)
        return state_dict
 
    def _set_state_params(self, state_dict):
        self.config.update(state_dict)
        actor_core = self.actor.module[0].module
        actor_core.beta = state_dict['current_beta']
        if hasattr(actor_core, 'tau') and 'current_tau' in state_dict:
            actor_core.tau = state_dict['current_tau']
     ###########################################################################
















class ObliqueTreePolicy(nn.Module):
    def __init__(self, tree_model):
        super().__init__()
        # Store the trained ParameterizedObliqueTree instance
        self.tree = tree_model

    @torch.no_grad()  # Strictly enforce no gradients!
    def forward(self, observation: torch.Tensor):
        # 1. Safely convert PyTorch tensor to NumPy
        obs_np = observation.detach().cpu().numpy()
        
        # Ensure it's 2D for the tree's predict method
        if obs_np.ndim == 1:
            obs_np = np.atleast_2d(obs_np)
            
        # 2. Get discrete class predictions from your tree
        actions_np = self.tree.predict(obs_np)
        
        # 3. Convert back to a PyTorch tensor
        # Match the device of the input observation and cast to long (integer) for discrete actions
        action_tensor = torch.tensor(
            actions_np, 
            device=observation.device, 
            dtype=torch.long
        )
        
        return action_tensor