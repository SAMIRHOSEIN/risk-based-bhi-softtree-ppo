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
# from torchrl.collectors import SyncDataCollector # In Amir's version(when we use torchrl: 0.9.2 and tensordict: 0.9.1)
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
from softtree.softtree_classification import SoftTreeClassifier
from softtree.extraction_util import prune_STC_nodes
from softtree.oblique_tree import ParameterizedObliqueTree


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
                # estimate GAE
                with torch.no_grad():
                    self.loss_module.value_estimator(td_data)

                # load to replay buffer
                self.replay_buffer.extend(td_data)

                # training weights
                for _ in range(epochs_per_batch):
                    minibatch = self.replay_buffer.sample(frames_per_minibatch)

                    # forward loss
                    loss_vals = self.loss_module(minibatch.to(self.device))
                    loss_total = loss_vals["loss_objective"] \
                        + loss_vals["loss_entropy"] \
                        + loss_vals["loss_critic"]
                    
                    loss_total += self._add_regularization_loss()

                    # backprop
                    self.optimizer.zero_grad()
                    loss_total.backward()
                    
                    if self.config.get('max_grad_norm') is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.loss_module.parameters(), self.config['max_grad_norm']
                        )
                    self.optimizer.step()

                # evaluate policy
                if eval_freq != 0 and i % eval_freq == 0:
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
                    eval_reward = np.mean(eval_log['eval_reward'])
                else:
                    eval_reward = np.nan
            
                pbar_str = "| ".join([
                    f"reward: {train_reward:.4e}",
                    f"reward (eval): {eval_reward:.4e}",
                ])
                pbar.set_description(pbar_str)

                # update log
                train_log['batch'].append(i)
                train_log['reward'].append(train_reward)
        
        return train_log, eval_log
    
    @classmethod
    @torch.no_grad()
    def evaluate(cls, actor, eval_env, num_episodes=1, max_steps=1, deterministic=True):
        """
        Evaluates a policy
        """
        actor.eval()
        # exploration_type = ExplorationType.RANDOM if deterministic else ExplorationType.GREEDY
        exploration_type = ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM

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
        
        # set_exploration_type(ExplorationType.RANDOM) # reset to default after evaluation
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
        # collector = SyncDataCollector( # Amir's version
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

    def _get_actor_core_hyperparams(self):
        params_dict = {
            "input_dim": self.actor.module[0].module.layers[0].in_features,
            "output_dim": self.actor.module[0].module.layers[-1].out_features,
            "actor_cells": self.actor.module[0].module.layers[0].out_features,
            "actor_layers": (len(self.actor.module[0].module.layers)-1)//2-1,
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
        observations_t: torch.Tensor,
        pruning_threshold,
    ):
        STC_core = actor.module[0].module
        max_depth = STC_core.depth

        weights = STC_core.inner_nodes.weight.detach().numpy()
        biases = STC_core.inner_nodes.bias.detach().numpy()
        leaf_logits = STC_core.leaf_nodes.leaf_scores.detach().numpy()
        leaf_values = np.argmax(leaf_logits, axis=1)

        prune_mask = prune_STC_nodes(STC_core, observations_t, pruning_threshold=pruning_threshold)
        odt_model = ParameterizedObliqueTree(
            max_depth, weights, biases, leaf_values, prune_mask,
        )

        odt_module = ObliqueTreePolicy(odt_model)

        # wrap to torchrl (_setup_actor not applicable sine odt doesn't give logits)
        odt_actor = TensorDictModule(
            odt_module, in_keys=['observation'], out_keys=['action']
        )

        return odt_actor

    def _get_actor_core_hyperparams(self):
        params_dict = {
            "input_dim": self.actor.module[0].module.input_dim,
            "output_dim": self.actor.module[0].module.output_dim,
            "depth": self.actor.module[0].module.depth,
            "beta": self.actor.module[0].module.beta
        }
        return params_dict
    
    @classmethod
    @torch.no_grad()
    def _set_actor_core(cls, state_dict, params_dict):
        actor_tree = SoftTreeClassifier(
            input_dim=params_dict['input_dim'],
            output_dim=params_dict['output_dim'],
            depth=params_dict['depth'],
            beta=params_dict['beta']
        )
        actor_tree.load_state_dict(state_dict)
        return actor_tree
    
    def _add_regularization_loss(self):
        l1_coef = self.config.get('actor_l1_coef', 0.0)
        l2_coef = self.config.get('actor_l2_coef', 0.0)
        gl1_coef = self.config.get('actor_gl1_coef', 0.0)

        loss = 0.0
        weights = self.actor.module[0].module.inner_nodes.weight
        
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
    
    def _update_state_params(self, batch_index):
        beta_anneal = self.config.get('beta_anneal', 1.0)
        beta_update_freq = self.config.get('beta_update_freq', 1)
        
        # update beta
        if beta_anneal > 1.0 and (batch_index+1) % beta_update_freq == 0:
            self.actor.module[0].module.beta *= beta_anneal
        
        return super()._update_state_params(batch_index)
    
    def _get_state_params(self):
        state_dict = {
            'current_beta': self.actor.module[0].module.beta,
            'beta_anneal': self.config.get('beta_anneal', 1.0),
            'beta_update_freq': self.config.get('beta_update_freq', 1),
        }
        return state_dict
    
    def _set_state_params(self, state_dict):
        self.config.update(state_dict)
        self.actor.module[0].module.beta = state_dict['current_beta']


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