import numpy as np
import casadi as cd


class environment():
    """
    A class representing an environment for reinforcement learning.

    Attributes:
        x (casadi.SX.sym): The state of the system.
        u (casadi.SX.sym): The action taken in the current state.
        system_equations (casadi.Function): The system equations.
        x_num (numpy.ndarray): The current state of the system.
        x0 (numpy.ndarray): The initial state of the system.
        x_lower (numpy.ndarray): The lower bounds for the state.
        x_upper (numpy.ndarray): The upper bounds for the state.
        relaxation_weights (numpy.ndarray): The penalty for constraint violation.
        Q_mat (numpy.ndarray): The quadratic cost matrix for the state.
        R_mat (numpy.ndarray): The quadratic cost matrix for the action.
        seed (int): The seed for the random number generator.
        rng (numpy.random.Generator): The random number generator.

    Methods:
        __init__(self, x0: np.ndarray = np.zeros((2,1)), seed: int = 99, max_error_boundary: float = 1e1): Initializes an instance of the Environment class.
        make_step(self, action): Executes a single step in the environment.
        _compute_stage_cost(self, state, action): Computes the stage cost for a given state and action, taking into account any constraint violations.
        set_initial_state(self, x0): Set the initial state of the environment.
        reset(self, random: bool = False): Resets the environment to its initial state.
    """

    def __init__(self, x0: np.ndarray = np.zeros((2,1)), seed: int = 99, max_error_boundary: float = 1e1):
        """
        Initializes an instance of the Environment class.

        Args:
            x0 (np.ndarray): The initial state of the system. Defaults to a 2x1 zero array.
            seed (int): The seed for the random number generator. Defaults to 99.
            max_error_boundary (float): The maximum error boundary for the system. Defaults to 10.

        Returns:
            None
        """

        # Define the system matrices
        A = np.array([[0.9,  0.35], [0, 1.1]])
        B = np.array([[0.0813], [0.2]])

        # Define the state and action
        self.x = cd.SX.sym("x", 2)
        self.u = cd.SX.sym("u", 1)

        # Define the system equations
        self.system_equations = A @ self.x + B @ self.u
        self._sys_func = cd.Function("system_fun", [self.x, self.u], [self.system_equations], ["x_k", "u_k"], ["x_k+1"])

        # Introduce an internal state
        self.x_num = self.x0 = x0

        # Define the state constraints
        self.x_lower = np.array([0, -1]).reshape(-1,1)
        self.x_upper = np.array([1, +1]).reshape(-1,1)

        # Define the penalty for the constraint violation
        self.relaxation_weights = np.ones((2,1)) * 1e2

        # Define the quadratic cost matrices
        self.Q_mat = np.eye(self.x.shape[0]) * 1
        self.R_mat = np.eye(self.u.shape[0]) * 0.5

        # Define the RNGs to obtain reproducable results
        self.seed = seed
        self.rng = np.random.default_rng(seed = seed)


    def make_step(self, action: np.ndarray):
        """
        Executes a single step in the environment.

        Args:
            action: The action to take in the environment.

        Returns:
            A tuple containing the next state, the reward received from the environment, and a flag indicating whether
            the episode is done.
        """

        old_state = self.x_num 
        next_state = self.x_num = self._sys_func(old_state, action).full()

        reward, done_flag = self._compute_stage_cost(old_state, action)

        return next_state, reward, done_flag
    

    def _compute_stage_cost(self, state: np.ndarray, action: np.ndarray):
        """
        Computes the stage cost for a given state and action, taking into account any constraint violations.

        Args:
            state (numpy.ndarray): The current state of the system.
            action (numpy.ndarray): The action to take in the current state.

        Returns:
            Tuple[float, bool]: A tuple containing the computed stage cost and a boolean indicating whether the episode is done.
        """

        # Basic stage cost without constraint violation
        stage_cost = float(state.T @ self.Q_mat @ state + action.T @ self.R_mat @ action)

        # Compute the penalty for constraint violation
        reference = np.zeros(self.x_upper.shape)
        h_upper = state - self.x_upper
        h_lower = self.x_lower - state

        max_error = np.concatenate((reference, h_upper, h_lower), axis=-1)
        max_error = np.amax(max_error, keepdims=True, axis=-1)

        max_error = float(self.relaxation_weights.T @ max_error)

        stage_cost += max_error

        done = False # This is always set to False to always run a full episode

        return stage_cost, done
    

    def set_initial_state(self, x0: np.ndarray):
        """
        Set the initial state of the environment.

        Args:
            x0 (numpy.ndarray): The initial state of the environment.

        Returns:
            None
        """
        self.x0 = x0
        self.x_num = x0


    def reset(self, random: bool = False):
            """
            Resets the environment to its initial state.

            Args:
                random (bool): If True, the state is randomly initialized within the bounds of the environment. If False, the RNG is reset to the seed and the initial state is set to the initial state of the environment.

            Returns:
                numpy.ndarray: The initial state of the environment.
            """
            if random:
                self.x_num = self.x_lower.reshape(2,1) + (self.x_upper - self.x_lower).reshape(2,1) * self.rng.uniform(low = 0, high = 1, size = 2).reshape(2,1)
                return self.x_num
            else:
                self.rng = np.random.default_rng(seed = self.seed)
                self.x_num = self.x0.copy()
                return self.x0
        
