
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
    STATE_TRANSITION_MODE,
    ELEMENT_UNIT_COSTS,
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
        transition_mode: str = STATE_TRANSITION_MODE,
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







        # validate transition_mode(It also checks that each stochastic unit count is a positive integer.)
        valid_transition_modes = {"deterministic", "stochastic"}
        if transition_mode not in valid_transition_modes:
            raise ValueError(
                f"transition_mode must be one of {valid_transition_modes}, got {transition_mode!r}."
            )

        self.transition_mode = transition_mode







        self.stochastic_unit_quantities = {}
        for element_no in self.element_numbers:
            element_no = int(element_no)
            q = int(ELEMENT_QUANTITIES[element_no])

            if q <= 0:
                raise ValueError(
                    f"stochastic_unit_quantities[{element_no}] must be a positive integer."
                )

            self.stochastic_unit_quantities[element_no] = q












        # Principal bridge value:
        # C0 = sum_i Q_i * UC_i        
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









        # if self.reset_prob is not None:
        #     self._state = self.reset_prob.astype(np.float32)
        # else:
        #     self._state = np.zeros((self.num_elements, NCS), dtype=np.float32)
        #     self._state[:, 0] = 1.0

        # observation = self._get_observation()

        if self.reset_prob is not None:
            self._state = self.reset_prob.astype(np.float32)
        else:
            self._state = np.zeros((self.num_elements, NCS), dtype=np.float32)
            self._state[:, 0] = 1.0







        # ================================================================
        # Initialize hidden stochastic counts
        # ================================================================
        # In deterministic mode:
        #     The environment only needs probability vectors.
        #     Therefore, self._counts is not used.
        #
        # In stochastic mode:
        #     The environment internally tracks integer counts for each condition state.
        #     The PPO actor still observes probabilities, not counts.
        #
        # Example:
        #     If element quantity = 100 and state = [0.4, 0.3, 0.2, 0.1],
        #     then counts = [40, 30, 20, 10].
        #
        # Then:
        #     state = counts / total_quantity
        #
        # This keeps the observation format unchanged for PPO.
        # ================================================================
        if self.transition_mode == "stochastic":
            self._counts = self._state_to_counts(self._state)
            self._state = self._counts_to_state(self._counts)
        else:
            self._counts = None

        observation = self._get_observation()









        # info = {
        #     "cs": self._state,
        #     "time": self._time,
        #     "bhi": self._compute_bhi(self._state),
        #     "C0": self.C0,
        # }

        info = {
            "cs": self._state,
            "cs_counts": None if self._counts is None else self._counts.copy(), # Integer condition-state counts used only in stochastic mode.
            "time": self._time,
            "bhi": self._compute_bhi(self._state),
            "C0": self.C0,
            "transition_mode": self.transition_mode,
        }






        return observation, info        
        #########################################################







        #####################################
    def step(self, action):
        action = int(action)

        # Apply bridge-level action to all elements.
        # next_state = self._apply_action_transition(action, self._state)
        # Apply bridge-level action to all elements.
        if self.transition_mode == "deterministic":
            next_state = self._apply_action_transition_deterministic(action, self._state)
            next_counts = None
        else:
            next_state, next_counts = self._apply_action_transition_stochastic(action)        








        # Compute BHI after the action.
        # I need to enter next_state to compute BHI because 
        # Reward function is "After I apply this action, what is the bridge worth, minus what I paid?"
        # So I need to calculte the BHI after the action is applied, and then calcualte the remaining value of bridge
        # and then subtract the action cost to get the reward. 
        bhi = self._compute_bhi(next_state)
        # Compute BHI before the action.
        # bhi = self._compute_bhi(self._state)


        # Compute action cost:
        # C(a) = sum_j Q_j * UC_j
        # where j belongs to the replaced element groups under action a.
        action_cost = self._compute_action_cost(action)


        # Reward:
        # R(s,a) = [BHI(s) * C0 - C(a)] / reward_normalizer
        reward = (bhi * self.C0 - action_cost) / self.reward_normalizer

        discount_factor = self.discount ** self._time
        reward = np.float32(discount_factor * reward)




        # self._state = next_state
        # self._time += 1

        self._state = next_state

        # Store the new integer counts only in stochastic mode.
        # These counts are needed for the next stochastic transition.
        if self.transition_mode == "stochastic":
            self._counts = next_counts

        self._time += 1












        observation = self._get_observation()

        terminated = self._time >= self.max_steps
        truncated = False


        # info = {
        #     "cs": self._state,
        #     "time": self._time,
        #     "bhi": bhi,
        #     "C0": self.C0,
        #     "action_cost": action_cost,
        #     "reward_normalizer": self.reward_normalizer,
        #     "reward": reward,
        #     "discount": discount_factor,
        # }

        info = {
            "cs": self._state,
            "cs_counts": None if self._counts is None else self._counts.copy(),
            "time": self._time,
            "bhi": bhi,
            "C0": self.C0,
            "action_cost": action_cost,
            "reward_normalizer": self.reward_normalizer,
            "reward": reward,
            "discount": discount_factor,
            "transition_mode": self.transition_mode,
        }












        return observation, reward, terminated, truncated, info
        #####################################





        ##################### helper functions
    def _get_observation(self):
        flat_state = self._state.reshape(-1)

        if self.include_step_count:
            return np.append(flat_state, self._time / self.max_steps).astype(np.float32)

        return flat_state.astype(np.float32)













    # def _apply_action_transition(self, action, state):
    #     next_state = np.zeros_like(state, dtype=np.float32)
    #     replaced_groups = ACTION_REPLACEMENT_MASK[action]

    #     for idx, element_no in enumerate(self.element_numbers):
    #         element_no = int(element_no)
    #         group = ELEMENT_TO_GROUP[element_no]

    #         if group in replaced_groups:
    #             transition_matrix = REPLACEMENT_TRANSITION
    #         else:
    #             transition_matrix = DO_NOTHING_TRANSITIONS[element_no]

    #         element_state = transition_matrix.T @ state[idx, :]
    #         element_state = element_state / element_state.sum()
    #         next_state[idx, :] = element_state.astype(np.float32)

    #     return next_state



    # ================================================================
    #normalize probability vectors
    # ================================================================
    # This function makes sure a vector is a valid probability distribution.
    #
    # Why needed?
    #     Due to floating-point numerical issues, a vector may contain tiny
    #     negative values or may not sum exactly to 1.
    # This function clips negative values to zero and normalizes the vector.
    # ================================================================
    @staticmethod
    def _normalize_probabilities(probabilities):
        probabilities = np.asarray(probabilities, dtype=float)

        # Remove tiny negative numerical errors.
        # Example: -1e-16 becomes 0.
        probabilities = np.clip(probabilities, 0.0, None)

        total = probabilities.sum()

        # A probability vector with zero total mass is invalid.
        if total <= 0:
            raise ValueError("Probability vector has zero or negative total mass.")

        return probabilities / total


    # ================================================================
    # Convert probability state to integer counts
    # ================================================================
    # Convert probabilities to non-integer expected counts.
    #
    # Example:
    #     probs = [0.33, 0.33, 0.34, 0.0]
    #     q = 10
    #     raw_counts = [3.3, 3.3, 3.4, 0.0]
    #
    # First, we take the floor:
    #     base_counts = [3, 3, 3, 0]
    #
    # This sums to 9, but q = 10, so one unit is still missing.
    # The missing unit is assigned to the condition state with the largest
    # fractional part.
    #
    # fractional_parts = [0.3, 0.3, 0.4, 0.0]
    #
    # Therefore, the final count becomes:
    #     final_counts = [3, 3, 4, 0]
    # ================================================================

    def _state_to_counts(self, state):
        counts = np.zeros((self.num_elements, NCS), dtype=np.int64)

        for idx, element_no in enumerate(self.element_numbers):
            element_no = int(element_no)

            q = self.stochastic_unit_quantities[element_no]

            probs = self._normalize_probabilities(state[idx, :])

            # Convert probabilities to non-integer expected counts.
            raw_counts = probs * q

            # Take the floor as the base integer count.
            # Example:floor([3.3, 3.3, 3.4, 0.0]) = [3, 3, 3, 0]
            base_counts = np.floor(raw_counts).astype(np.int64)

            # Count how many units still need to be assigned.
            # Example:
            #     q = 10
            #     sum(base_counts) = 9
            #     remainder = 1
            remainder = q - int(base_counts.sum())

            # Assign the leftover units to the condition states with the largest
            # fractional parts.
            # Example:
            #     fractional_parts = [0.3, 0.3, 0.4, 0.0]
            #     The extra unit goes to CS3.
            if remainder > 0:
                fractional_parts = raw_counts - base_counts
                add_order = np.argsort(fractional_parts)[::-1]
                base_counts[add_order[:remainder]] += 1

            counts[idx, :] = base_counts

        return counts


    # ================================================================
    # Convert integer counts back to probability state
    # ================================================================
    def _counts_to_state(self, counts):
        state = np.zeros((self.num_elements, NCS), dtype=np.float32)

        for idx, element_no in enumerate(self.element_numbers):
            element_no = int(element_no)

            # Total number of stochastic units for this element.
            q = self.stochastic_unit_quantities[element_no]

            state[idx, :] = counts[idx, :] / q

        return state


    # Deterministic transition function
    # This is our original next-state method.
    def _apply_action_transition_deterministic(self, action, state):
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

            element_state = self._normalize_probabilities(element_state)

            next_state[idx, :] = element_state.astype(np.float32)

        return next_state


    # ================================================================
    # Stochastic transition function
    # ================================================================
    # Instead of directly calculating the expected next condition distribution,
    # this method samples the next condition-state counts.
    #
    # For each element:
    #
    #     1. The element has integer counts in each current condition state.
    #
    #     2. For each current condition state, sample how many units move to
    #        each next condition state using a multinomial distribution.
    #
    #     3. Add sampled results from all current condition states.
    #
    #     4. Convert final counts back to probabilities.
    # ================================================================
    def _apply_action_transition_stochastic(self, action):

        
        if self._counts is None:
            self._counts = self._state_to_counts(self._state)

        next_counts = np.zeros_like(self._counts, dtype=np.int64)

        replaced_groups = ACTION_REPLACEMENT_MASK[action]

        for idx, element_no in enumerate(self.element_numbers):
            element_no = int(element_no)
            group = ELEMENT_TO_GROUP[element_no]

            q = self.stochastic_unit_quantities[element_no]

            # ------------------------------------------------------------
            # Case 1: the action replaces this element's group
            # ------------------------------------------------------------
            # Replacement is treated as deterministic restoration.
            #
            # If an element is replaced, all its units return to CS1:
            #
            #     next_counts = [q, 0, 0, 0]
            #
            # This is consistent with the current REPLACEMENT_TRANSITION matrix.
            # We do not sample replacement because replacement means full reset.
            # ------------------------------------------------------------
            if group in replaced_groups:
                next_counts[idx, 0] = q
                continue

            # ------------------------------------------------------------
            # Case 2: the element is not replaced
            # ------------------------------------------------------------
            # Use the do-nothing deterioration transition matrix for this element.
            # ------------------------------------------------------------
            transition_matrix = DO_NOTHING_TRANSITIONS[element_no]

            # For each current condition state, sample where the units go.
            # Example:If current_cs = CS1 and n_current = 40,
            # then sample how many of those 40 units go to CS1, CS2, CS3, CS4.
            for current_cs in range(NCS):
                n_current = int(self._counts[idx, current_cs])


                if n_current == 0:
                    continue


                transition_probabilities = self._normalize_probabilities(
                    transition_matrix[current_cs, :]
                )

                # Multinomial sampling:
                sampled_counts = self.np_random.multinomial(
                    n_current,
                    transition_probabilities,
                )


                next_counts[idx, :] += sampled_counts

        # Convert sampled integer counts back to probability vectors.
        # PPO and BHI calculations continue using probabilities, not raw counts.
        next_state = self._counts_to_state(next_counts)

        return next_state, next_counts

































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
            unit_cost = ELEMENT_UNIT_COSTS[element_no]
            quantity = ELEMENT_QUANTITIES[element_no]
            principal_cost += unit_cost * quantity

        return float(principal_cost)


    def _compute_action_cost(self, action):
        replaced_groups = ACTION_REPLACEMENT_MASK[action]
        action_cost = 0.0

        for element_no in self.element_numbers:
            element_no = int(element_no)
            group = ELEMENT_TO_GROUP[element_no]

            if group in replaced_groups:
                unit_cost = ELEMENT_UNIT_COSTS[element_no]
                quantity = ELEMENT_QUANTITIES[element_no]
                action_cost += unit_cost * quantity

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