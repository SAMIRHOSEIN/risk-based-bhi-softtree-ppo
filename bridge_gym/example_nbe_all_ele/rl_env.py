import numpy as np
import matplotlib.pyplot as plt
from jaxtyping import Float32

# gynasium imports
import gymnasium as gym
from gymnasium import spaces

# import element costs
from .settings import NCS, NA
from .settings import CS_PFS, FAILURE_COST
from .settings import ACTION_MODEL, UNIT_COSTS
from .cost_util import normalized_cost as cost_util


class SingleElement(gym.Env):
    # standard: should not change
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self, max_steps, discount,
        state_size=NCS, action_size=NA,
        include_step_count=False,
        reset_prob=None,
        dirichlet_alpha: Float32[np.ndarray, f"{NCS}"] | None = None,
        action_model=ACTION_MODEL,
        unit_costs=UNIT_COSTS,
        pf_array=CS_PFS,
        failure_cost=FAILURE_COST,
        render_mode=None,
        render_kwargs: dict | None = None,
        cost_kwargs: dict | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        # store env parameters
        self.state_size = state_size
        self.action_size = action_size
        self.max_steps = max_steps
        self.include_step_count = include_step_count

        self.discount = discount
        self.pf_array = pf_array
        self.failure_cost = failure_cost
        self.action_model, self.unit_costs = action_model, unit_costs
        if cost_kwargs is None:
            self.cost_kwargs = {}
        else:
            self.cost_kwargs = cost_kwargs

        # reset parameters
        if reset_prob is None:
            assert dirichlet_alpha is not None, "Must provide dirichlet_alpha if no reset_prob"
            assert dirichlet_alpha.shape == (self.state_size,), "dirichlet_alpha must be of shape (state_size,)"
            self.reset_prob = None
            self.dirichlet_alpha = dirichlet_alpha
        else:
            assert dirichlet_alpha is None, "Cannot provide dirichlet_alpha if reset_prob is provided"
            self.reset_prob = reset_prob
            self.dirichlet_alpha = None

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
    
    def reset(self, seed: int | None = None, options: dict | None = None):
        # set seed
        if seed is not None:
            super().reset(seed=seed)
        elif self._first_rest:
            # only initialize if needed
            super().reset(seed=self._seed)
            self._first_rest = False

        self._time = 0
        if self.reset_prob is not None:
            self._state = self.reset_prob
        else:
            state = self.np_random.dirichlet(
                self.dirichlet_alpha,
            )
            self._state = state.astype(np.float32)

        if self.include_step_count:
            observation = np.append(self._state, self._time/self.max_steps)
        else:
            observation = self._state
        info = {"cs": self._state, "time": self._time}

        return observation, info

    def step(self, action):
        # state after the action
        state = self.action_model[int(action)].T @ self._state
        # normalize state to fix numerical errors
        state = state / state.sum()

        # reward due to action
        dir_cost = self.unit_costs[int(action)] @ self._state
        fail_risk = (self.pf_array @ self._state) * self.failure_cost
        cost = dir_cost + fail_risk
        reward = cost_util(cost, **self.cost_kwargs)
        discount_factor = self.discount**self._time
        reward = (discount_factor*reward).astype(np.float32)

        # update hidden state
        self._state = state
        self._time += 1

        # update observation
        if self.include_step_count:
            observation = np.append(state, self._time/self.max_steps)
        else:
            observation = state

        # check if done
        if self._time > self.max_steps:
            done = True
            _, info = self.reset()
        else:
            done = False
            info = {
                "cs": state, "dir_cost": dir_cost, "fail_risk": fail_risk,
                "cost": cost, "reward": reward, "discount": discount_factor
            }

        terminated = done

        return observation, reward, terminated, done, info

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
            self.colors = plt.cm.viridis(np.linspace(0, 1, self.state_size))
            self.ax.set_xlim(-0.8, self.max_steps+0.8)
            self.ax.set_ylim(0, 1.05)
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("CS distribution")
            # setup colorbar
            norm = plt.Normalize(vmin=0, vmax=self.state_size - 1)
            mappable = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
            cbar = self.fig.colorbar(
                mappable, 
                ax=self.ax, 
                ticks=range(self.state_size),
                boundaries=np.arange(-0.5, self.state_size, 1) # Centers ticks in color blocks
            )
            cbar.set_ticklabels([f'CS{i+1}' for i in range(self.state_size)])
            plt.show(block=False)

        # Draw the stacked bar for the current time step
        bottom = 0
        for i in range(self.state_size):
            val = self._state[i]
            self.ax.bar(
                self._time, val, bottom=bottom,
                color=self.colors[i], width=0.8,
            )
            bottom += val

        # Update the display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()