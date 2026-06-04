
import numpy as np
import matplotlib.pyplot as plt

# gynasium imports
import gymnasium as gym
from gymnasium import spaces

###############################
from .settings import (
    NCS,
    NA,
    ELEMENT_NUMBERS,
    ELEMENT_TO_GROUP,
    ELEMENT_WEIGHTS,
    ELEMENT_QUANTITIES,
    ACTION_REPLACEMENT_MASK,
    DO_NOTHING_TRANSITIONS,
    REPLACEMENT_TRANSITION,
    HEALTH_COEFFICIENTS,
)
################################


class BridgeBHIEnv(gym.Env):
    # standard: should not change
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

        ###############################
    def __init__(
        self,
        max_steps,
        discount,
        element_numbers=ELEMENT_NUMBERS,
        action_size=NA,
        include_step_count=False,
        reset_prob=None,
        reward_normalizer: float | None = None,
        render_mode=None,
        render_kwargs: dict | None = None,
        seed: int | None = None,
    ):

        super().__init__()
        # store environment parameters
        self.element_numbers = np.asarray(element_numbers, dtype=int)
        self.num_elements = len(self.element_numbers)

        # State contains the condition-state probability vector for each element.
        # Shape internally: (num_elements, NCS)
        # Shape returned to PPO: (num_elements * NCS,)
        self.state_size = self.num_elements * NCS

        self.action_size = action_size
        self.max_steps = max_steps
        self.include_step_count = include_step_count
        self.discount = discount


        # Principal bridge value:
        # C0 = sum_i W_i * Q_i
        self.C0 = self._compute_principal_cost()


        self.reward_normalizer = reward_normalizer if reward_normalizer is not None else self.C0


        # reset parameters
        # If reset_prob is not provided, all elements start from [1, 0, 0, 0]
        if reset_prob is None:
            self.reset_prob = None
        else:
            reset_prob = np.asarray(reset_prob, dtype=np.float32)
            assert reset_prob.shape == (self.num_elements, NCS), (
                f"reset_prob must have shape ({self.num_elements}, {NCS})."
            )
            self.reset_prob = reset_prob
            ###############################
        # define observation and action spaces
        obs_size = self.state_size+1 if include_step_count else self.state_size
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.action_size)


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._seed = seed
        self._first_rest = True

        # plotting parameters
        self.fig, self.ax, self.colors = None, None, None
        self.render_kwargs = {} if render_kwargs is None else render_kwargs
        # We'll use a list of colors for the different state components
    


        ###################################################
        # This initializes every element as: [1, 0, 0, 0]
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            super().reset(seed=seed)
        elif self._first_rest:
            super().reset(seed=self._seed)
            self._first_rest = False

        self._time = 0

        if self.reset_prob is not None:
            self._state = self.reset_prob.astype(np.float32)
        else:
            self._state = np.zeros((self.num_elements, NCS), dtype=np.float32)
            self._state[:, 0] = 1.0

        observation = self._get_observation()

        info = {
            "cs": self._state,
            "time": self._time,
            "bhi": self._compute_bhi(self._state),
            "C0": self.C0,
        }

        return observation, info        
        #########################################################







        #####################################
    def step(self, action):
        action = int(action)

        # Apply bridge-level action to all elements.
        next_state = self._apply_action_transition(action, self._state)

        # Compute BHI after the action.
        # I need to enter next_state to compute BHI because 
        # Reward function is "After I apply this action, what is the bridge worth, minus what I paid?"
        # So I need to calculte the BHI after the action is applied, and then calcualte the remaining value of bridge
        # and then subtract the action cost to get the reward. 
        bhi = self._compute_bhi(next_state)
        # Compute BHI before the action.
        # bhi = self._compute_bhi(self._state)


        # Compute action cost:
        # C(a) = sum_j W_j * Q_j
        # where j belongs to the replaced element groups under action a.
        action_cost = self._compute_action_cost(action)

        # Reward:
        # R(s,a) = [BHI(s) * C0 - C(a)] / reward_normalizer
        reward = (bhi * self.C0 - action_cost) / self.reward_normalizer

        discount_factor = self.discount ** self._time
        reward = np.float32(discount_factor * reward)

        self._state = next_state
        self._time += 1

        observation = self._get_observation()

        terminated = self._time >= self.max_steps
        truncated = False


        info = {
            "cs": self._state,
            "time": self._time,
            "bhi": bhi,
            "C0": self.C0,
            "action_cost": action_cost,
            "reward_normalizer": self.reward_normalizer,
            "reward": reward,
            "discount": discount_factor,
        }


        return observation, reward, terminated, truncated, info
        #####################################





        ##################### helper functions
    def _get_observation(self):
        flat_state = self._state.reshape(-1)

        if self.include_step_count:
            return np.append(flat_state, self._time / self.max_steps).astype(np.float32)

        return flat_state.astype(np.float32)


    def _apply_action_transition(self, action, state):
        next_state = np.zeros_like(state, dtype=np.float32)
        replaced_groups = ACTION_REPLACEMENT_MASK[action]

        for idx, element_no in enumerate(self.element_numbers):
            element_no = int(element_no)
            group = ELEMENT_TO_GROUP[element_no]

            if group in replaced_groups:
                transition_matrix = REPLACEMENT_TRANSITION
            else:
                transition_matrix = DO_NOTHING_TRANSITIONS[element_no]

            element_state = transition_matrix.T @ state[idx, :]
            element_state = element_state / element_state.sum()
            next_state[idx, :] = element_state.astype(np.float32)

        return next_state


    def _compute_element_health(self, state):
        # H_i = CS_i dot K
        # K = [1.00, 0.66, 0.33, 0.00]
        return state @ HEALTH_COEFFICIENTS


    def _compute_bhi(self, state):
        element_health = self._compute_element_health(state)

        numerator = 0.0
        denominator = 0.0

        for idx, element_no in enumerate(self.element_numbers):
            element_no = int(element_no)
            weight = ELEMENT_WEIGHTS[element_no]

            numerator += weight * element_health[idx]
            denominator += weight

        return float(numerator / denominator)


    def _compute_principal_cost(self):
        principal_cost = 0.0

        for element_no in self.element_numbers:
            element_no = int(element_no)
            weight = ELEMENT_WEIGHTS[element_no]
            quantity = ELEMENT_QUANTITIES[element_no]
            principal_cost += weight * quantity

        return float(principal_cost)


    def _compute_action_cost(self, action):
        replaced_groups = ACTION_REPLACEMENT_MASK[action]
        action_cost = 0.0

        for element_no in self.element_numbers:
            element_no = int(element_no)
            group = ELEMENT_TO_GROUP[element_no]

            if group in replaced_groups:
                weight = ELEMENT_WEIGHTS[element_no]
                quantity = ELEMENT_QUANTITIES[element_no]
                action_cost += weight * quantity

        return float(action_cost)
            #####################


 




    ###########################################################
    #update rendering because state is now 2D
    def render(self):
        if self.render_mode == "human":
            self._render_gui()

        elif self.render_mode == "ansi":
            print(f"Step {self._time}: CS = {self._state}")

    def close(self):
        pass

    def _render_gui(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 1, tight_layout=True, **self.render_kwargs)
            self.colors = plt.cm.viridis(np.linspace(0, 1, NCS))
            self.ax.set_xlim(-0.8, self.max_steps+0.8)
            self.ax.set_ylim(0, 1.05)
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("CS distribution")
            # setup colorbar
            norm = plt.Normalize(vmin=0, vmax=NCS - 1)
            mappable = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
            cbar = self.fig.colorbar(
                mappable, 
                ax=self.ax, 
                ticks=range(NCS),
                boundaries=np.arange(-0.5, NCS, 1) # Centers ticks in color blocks
            )
            cbar.set_ticklabels([f"CS{i+1}" for i in range(NCS)])
            plt.show(block=False)

        # Draw the stacked bar for the current time step
        avg_state = self._state.mean(axis=0)

        bottom = 0
        for i in range(NCS):
            val = avg_state[i]
            self.ax.bar(
                self._time,
                val,
                bottom=bottom,
                color=self.colors[i],
                width=0.8,
            )
            bottom += val


        # Update the display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()