import os
import pickle
import casadi as cd
import numpy as np

from do_mpc.model import Model
from scipy.linalg import solve_discrete_are
from multiprocessing.pool import Pool
from itertools import cycle

from environments import environment as Environment

from RL_Tools.tools.RL_AC_MPC import RL_AC_MPC as MPC
from RL_Tools.agents.AC_MPC_agent_NN import AC_MPC_agent_NN_QuasiNewton_TR as Agent

def build_perfect_MPC():
    # Define perfect model
    exact_model = Model("discrete")
    
    x_exact = exact_model.set_variable(var_type = "_x", var_name = "x", shape = (2,1))
    u_exact = exact_model.set_variable(var_type = "_u", var_name = "u", shape = (1,1))
    
    A_exact = cd.DM([[0.9,  0.35], [0, 1.1]])
    B_exact = cd.DM([[0.0813], [0.2]])
    
    x_next_exact = A_exact @ x_exact + B_exact @ u_exact
    exact_model.set_rhs("x", x_next_exact)
    
    exact_model.setup()

    benchmark_MPC = MPC(model = exact_model)
    
    q_exact = cd.DM([1, 1])
    lterm = x_exact.T @ cd.diag(q_exact) @ x_exact + 0.5 * u_exact.T @ u_exact
    
    S_exact = solve_discrete_are(A_exact, B_exact, cd.diag(q_exact), cd.DM(0.5))
    mterm = x_exact.T @ S_exact @ x_exact
    benchmark_MPC.set_objective(lterm = lterm, mterm = mterm)
    
    benchmark_MPC.set_rterm(u = 0)
    
    benchmark_MPC.bounds["lower", "_u", "u"] = -1
    benchmark_MPC.bounds["upper", "_u", "u"] = +1
    
    x_lb_exact = cd.DM([0, -1])
    x_ub_exact = cd.DM([1, +1])
    
    benchmark_MPC.bounds["lower", "_x", "x"] = x_lb_exact
    benchmark_MPC.bounds["upper", "_x", "x"] = x_ub_exact
    
    
    benchmark_MPC.set_param(n_horizon = 50, t_step= 1, nlpsol_opts = {"ipopt": {"print_level": 0}, "print_time": False}, gamma = 1)
    
    benchmark_MPC.setup()
    return benchmark_MPC

def run_benchmark_MPC(x0, seed, n_steps, path):
    if len(x0.shape) == 1:
        x0 = x0.reshape(-1, 1)
    
    env = Environment(seed = seed)
    env.set_initial_state(x0.copy())

    MPC = build_perfect_MPC()
    MPC._x0.master = cd.DM(x0.copy())
    MPC.set_initial_guess()

    clc_list = []

    x_data = np.empty(shape = (n_steps + 1, x0.shape[0]))
    u_data = np.empty(shape = (n_steps, MPC.u0.shape[0]))
    x_data[0, :] = x0.flatten()

    for idx in range(n_steps):
        u = MPC.make_step(x0)
        x_next, stage_cost, done_flag = env.make_step(u)
        x0 = x_next.copy()

        x_data[idx + 1, :] = x0.flatten().copy()
        u_data[idx, :] = u.flatten().copy()
        clc_list.append(stage_cost)

    if not path.endswith("\\"):
        path += "\\"
    path += "rntm\\"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + f"trajectory_x_{seed}.pkl", "wb") as f:
        pickle.dump(x_data, f)
    
    with open(path + f"trajectory_u_{seed}.pkl", "wb") as f:
        pickle.dump(u_data, f)
    
    with open(path + f"trajectory_clc_{seed}.pkl", "wb") as f:
        pickle.dump(clc_list, f)

    with open(path + f"clc_{seed}.pkl", "wb") as f:
        pickle.dump(np.sum(clc_list), f)
    
    return

def run_RL_MPC(x0: np.ndarray, seed: int, n_steps: int, agent_path: str, path: str):
    if len(x0.shape) == 1:
        x0 = x0.reshape(-1, 1)
    
    env = Environment(seed = seed)
    env.set_initial_state(x0.copy())

    MPC = Agent.load(path = agent_path)
    MPC.actor._x0.master = cd.DM(x0.copy())
    MPC.actor.set_initial_guess()

    clc_list = []

    x_data = np.empty(shape = (n_steps + 1, x0.shape[0]))
    u_data = np.empty(shape = (n_steps, MPC.actor.u0.shape[0]))
    x_data[0, :] = x0.flatten()

    for idx in range(n_steps):
        u = MPC.act(x0)
        x_next, stage_cost, done_flag = env.make_step(u)
        x0 = x_next.copy()

        x_data[idx + 1, :] = x0.flatten().copy()
        u_data[idx, :] = u.flatten().copy()
        clc_list.append(stage_cost)

    if not path.endswith("\\"):
        path += "\\"
    path += "rntm\\"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + f"trajectory_x_{seed}.pkl", "wb") as f:
        pickle.dump(x_data, f)
    
    with open(path + f"trajectory_u_{seed}.pkl", "wb") as f:
        pickle.dump(u_data, f)
    
    with open(path + f"trajectory_clc_{seed}.pkl", "wb") as f:
        pickle.dump(clc_list, f)

    with open(path + f"clc_{seed}.pkl", "wb") as f:
        pickle.dump(np.sum(clc_list), f)
    
    return

def combine_data(path: str, seeds: list):

    # Preprocessing of path
    if not path.endswith("\\"):
        path += "\\"
    path += "rntm\\"

    # Assign empty lists to the loaded data
    x_data = []
    u_data = []
    clc_data = []
    clc_sum = []

    # Iterate over all seeds and load and delete the data
    for seed in seeds:
        with open(path + f"trajectory_x_{seed}.pkl", "rb") as f:
            x_data.append(pickle.load(f))
        os.remove(path + f"trajectory_x_{seed}.pkl")

        with open(path + f"trajectory_u_{seed}.pkl", "rb") as f:
            u_data.append(pickle.load(f))
        os.remove(path + f"trajectory_u_{seed}.pkl")

        with open(path + f"trajectory_clc_{seed}.pkl", "rb") as f:
            clc_data.append(pickle.load(f))
        os.remove(path + f"trajectory_clc_{seed}.pkl")

        with open(path + f"clc_{seed}.pkl", "rb") as f:
            clc_sum.append(pickle.load(f))
        os.remove(path + f"clc_{seed}.pkl")
    
    # Stack the data to one struct
    x_data = np.stack(x_data, axis = 0)
    u_data = np.stack(u_data, axis = 0)
    clc_data = np.stack(clc_data, axis = 0)
    clc_sum = np.stack(clc_sum, axis = 0)

    # Save the data
    path = path[:-len("rntm\\")]
    with open(path + f"trajectory_x.pkl", "wb") as f:
        pickle.dump(x_data, f)
    with open(path + f"trajectory_u.pkl", "wb") as f:
        pickle.dump(u_data, f)
    with open(path + f"trajectory_clc.pkl", "wb") as f:
        pickle.dump(clc_data, f)
    with open(path + f"clc.pkl", "wb") as f:
        pickle.dump(clc_sum, f)

    return

if __name__ == "__main__":

    n_free_cpu = 2
  
    # Define Testing Hyperparameters
    rng_seed = 10 # Note: The agents were trained on 99
    rng = np.random.default_rng(seed = rng_seed)
    
    n_initial_conditions = 2500
    n_steps = 50
    
    n_x = 2
    x_lower = np.array([0, -1]).reshape(-1, 1)
    x_upper = np.array([1, 1]).reshape(-1, 1)
    
    X0 = x_lower.T + (x_upper - x_lower).T * rng.random((n_initial_conditions, n_x))
    seed_list = range(n_initial_conditions)
    
    gamma = 1

    # Data paths
    benchmark_MPC_data_path = "data\\test_data\\Benchmark MPC\\"
    
    # Benchmark MPC
    with Pool(os.cpu_count() - n_free_cpu) as p:
        p.starmap(run_benchmark_MPC, zip(X0, seed_list, cycle([n_steps]), cycle([benchmark_MPC_data_path])))
    combine_data(benchmark_MPC_data_path, seed_list)

    # Untrained MPC
    untrained_MPC_data_path = "data\\test_data\\Untrained MPC\\"
    untrained_MPC_agent_path ="data\\agent\\GD_MPC\\untrained"
    with Pool(os.cpu_count() - n_free_cpu) as p:
        p.starmap(run_RL_MPC, zip(X0, seed_list, cycle([n_steps]), cycle([untrained_MPC_agent_path]), cycle([untrained_MPC_data_path])))
    combine_data(untrained_MPC_data_path, seed_list)

    # First order
    GD_MPC_data_path = "data\\test_data\\GD_MPC\\"
    GD_MPC_agent_path ="data\\agent\\GD_MPC"
    with Pool(os.cpu_count() - n_free_cpu) as p:
        p.starmap(run_RL_MPC, zip(X0, seed_list, cycle([n_steps]), cycle([GD_MPC_agent_path]), cycle([GD_MPC_data_path])))
    combine_data(GD_MPC_data_path, seed_list)

    # First order with TR
    TRGD_MPC_data_path = "data\\test_data\\TRGD_MPC\\"
    TRGD_MPC_agent_path ="data\\agent\\TRGD_MPC"
    with Pool(os.cpu_count() - n_free_cpu) as p:
        p.starmap(run_RL_MPC, zip(X0, seed_list, cycle([n_steps]), cycle([TRGD_MPC_agent_path]), cycle([TRGD_MPC_data_path])))
    combine_data(TRGD_MPC_data_path, seed_list)

    # Second order
    QN_MPC_data_path = "data\\test_data\\QN_MPC\\"
    QN_MPC_agent_path ="data\\agent\\QN_MPC"
    with Pool(os.cpu_count() - n_free_cpu) as p:
        p.starmap(run_RL_MPC, zip(X0, seed_list, cycle([n_steps]), cycle([QN_MPC_agent_path]), cycle([QN_MPC_data_path])))
    combine_data(QN_MPC_data_path, seed_list)

    # Second order with TR
    TRQN_MPC_data_path = "data\\test_data\\TRQN_MPC\\"
    TRQN_MPC_agent_path ="data\\agent\\TRQN_MPC"
    with Pool(os.cpu_count() - n_free_cpu) as p:
        p.starmap(run_RL_MPC, zip(X0, seed_list, cycle([n_steps]), cycle([TRQN_MPC_agent_path]), cycle([TRQN_MPC_data_path])))
    combine_data(TRQN_MPC_data_path, seed_list)