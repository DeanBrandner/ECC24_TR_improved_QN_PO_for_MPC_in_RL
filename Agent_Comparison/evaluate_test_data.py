import pickle
import numpy as np
import pandas as pd

def load_data(path: str):
    if not path.endswith("\\"):
        path += "\\"

    with open(path + "trajectory_x.pkl", "rb") as f:
        x_data = pickle.load(f)
    with open(path + "trajectory_u.pkl", "rb") as f:
        u_data = pickle.load(f)
    with open(path + "trajectory_clc.pkl", "rb") as f:
        clc_data = pickle.load(f)
    with open(path + "clc.pkl", "rb") as f:
        clc_sum = pickle.load(f)

    return x_data, u_data, clc_data, clc_sum

def extract_feasible_trajectories_from_benchmark(x_data: np.ndarray, u_data: np.ndarray, clc_data: np.ndarray, clc_sum: np.ndarray):
    feasible_indices = np.where(
        (x_data[:, :, 0] > 0) * (x_data[:, :, 0] < 1) * (x_data[:, :, 1] > -1) * (x_data[:, :, 1] < 1),
        np.ones(x_data[:, :, 0].shape, dtype=np.int8),
        np.zeros(x_data[:,:,0].shape, dtype=np.int8)
        )
    feasible_trajectories = np.where(feasible_indices.sum(axis = 1) == feasible_indices.shape[1])[0]

    x_data_feasible, x_data_infeasible = separate(x_data, feasible_trajectories)
    u_data_feasible, u_data_infeasible = separate(u_data, feasible_trajectories)
    clc_data_feasible, clc_data_infeasible = separate(clc_data, feasible_trajectories)
    clc_sum_feasible, clc_sum_infeasible = separate(clc_sum, feasible_trajectories)

    return x_data_feasible, x_data_infeasible, u_data_feasible, u_data_infeasible, clc_data_feasible, clc_data_infeasible, clc_sum_feasible, clc_sum_infeasible, feasible_trajectories

def separate(data: np.ndarray, feasible_trajectories: np.ndarray):
    data_feasible = data[feasible_trajectories]
    data_infeasible = np.delete(data, feasible_trajectories, axis = 0)
    return data_feasible, data_infeasible

def extract_trajectories(x_data: np.ndarray, u_data: np.ndarray, clc_data: np.ndarray, clc_sum: np.ndarray, feasible_trajectories: np.ndarray):
    x_data_feasible, x_data_infeasible = separate(x_data, feasible_trajectories)
    u_data_feasible, u_data_infeasible = separate(u_data, feasible_trajectories)
    clc_data_feasible, clc_data_infeasible = separate(clc_data, feasible_trajectories)
    clc_sum_feasible, clc_sum_infeasible = separate(clc_sum, feasible_trajectories)
    return x_data_feasible, x_data_infeasible, u_data_feasible, u_data_infeasible, clc_data_feasible, clc_data_infeasible, clc_sum_feasible, clc_sum_infeasible

def evaluate_constraint_violations(x_data: np.ndarray):
    x_data = x_data.reshape(-1, x_data.shape[-1])
    x_ub = np.array([1, +1]).reshape(1, -1)
    x_lb = np.array([0, -1]).reshape(1, -1)

    upper_constraints = (x_data - x_ub).clip(min = 0)
    lower_constraints = (x_lb - x_data).clip(min = 0)

    max_violation = np.max([upper_constraints.max(), lower_constraints.max()])

    violations = np.max([upper_constraints.flatten(), lower_constraints.flatten()], axis = 0)
    violation_loc = np.where((violations > 0.))[0]
    n_violations = violation_loc.shape[0]

    violations = violations[violation_loc]
    mean_violation = violations.mean()
    std_violation = violations.std()
    return max_violation, n_violations, mean_violation, std_violation

def evaluate(x_data: np.ndarray, u_data: np.ndarray, clc_data: np.ndarray, clc_sum: np.ndarray):

    x_data_feasible, x_data_infeasible, u_data_feasible, u_data_infeasible, clc_data_feasible, clc_data_infeasible, clc_sum_feasible, clc_sum_infeasible, feasible_trajectories = extract_feasible_trajectories_from_benchmark(x_data, u_data, clc_data, clc_sum)

    n_feasible = x_data_feasible.shape[0]
    n_infeasible = x_data_infeasible.shape[0]
    p_feasible = n_feasible / (n_feasible + n_infeasible)
    p_infeasible = 1 - p_feasible
    clc_mean_all = clc_sum.mean()
    clc_mean = clc_sum_feasible.mean()
    clc_std = clc_sum_feasible.std()
    max_constraint_violation = 0
    n_violations = 0
    mean_constraint_violation = 0
    if n_infeasible > 0:
        max_constraint_violation, n_violations, mean_constraint_violation, std_constraint_violation = evaluate_constraint_violations(x_data_infeasible)

    return n_feasible, n_infeasible, p_feasible, p_infeasible, clc_mean_all, clc_mean, clc_std, max_constraint_violation, n_violations, mean_constraint_violation

if __name__ == "__main__":
    benchmark_data_path = "data\\test_data\\Benchmark MPC\\"
    untrained_data_path = "data\\test_data\\Untrained MPC\\"
    first_order_data_path = "data\\test_data\\GD_MPC\\"
    first_order_TR_data_path = "data\\test_data\\TRGD_MPC\\"
    second_order_data_path = "data\\test_data\\QN_MPC\\"
    second_order_TR_data_path = "data\\test_data\\TRQN_MPC\\"

    save_path = "data\\test_data\\clc_evaluation.xlsx"

    # Load data
    x_data_benchmark, u_data_benchmark, clc_data_benchmark, clc_sum_benchmark = load_data(benchmark_data_path)
    x_data_untrained, u_data_untrained, clc_data_untrained, clc_sum_untrained = load_data(untrained_data_path)
    x_data_first_order, u_data_first_order, clc_data_first_order, clc_sum_first_order = load_data(first_order_data_path)
    x_data_first_order_TR, u_data_first_order_TR, clc_data_first_order_TR, clc_sum_first_order_TR = load_data(first_order_TR_data_path)
    x_data_second_order, u_data_second_order, clc_data_second_order, clc_sum_second_order = load_data(second_order_data_path)
    x_data_second_order_TR, u_data_second_order_TR, clc_data_second_order_TR, clc_sum_second_order_TR = load_data(second_order_TR_data_path)

    # Extract feasible benchmark trajectories
    x_data_benchmark, _, u_data_benchmark, _, clc_data_benchmark, _, clc_sum_benchmark, _, feasible_trajectories = extract_feasible_trajectories_from_benchmark(x_data_benchmark, u_data_benchmark, clc_data_benchmark, clc_sum_benchmark)
    x_data_untrained, _, u_data_untrained, _, clc_data_untrained, _, clc_sum_untrained, _ = extract_trajectories(x_data_untrained, u_data_untrained, clc_data_untrained, clc_sum_untrained, feasible_trajectories)
    x_data_first_order, _, u_data_first_order, _, clc_data_first_order, _, clc_sum_first_order, _ = extract_trajectories(x_data_first_order, u_data_first_order, clc_data_first_order, clc_sum_first_order, feasible_trajectories)
    x_data_first_order_TR, _, u_data_first_order_TR, _, clc_data_first_order_TR, _, clc_sum_first_order_TR, _ = extract_trajectories(x_data_first_order_TR, u_data_first_order_TR, clc_data_first_order_TR, clc_sum_first_order_TR, feasible_trajectories)
    x_data_second_order, _, u_data_second_order, _, clc_data_second_order, _, clc_sum_second_order, _ = extract_trajectories(x_data_second_order, u_data_second_order, clc_data_second_order, clc_sum_second_order, feasible_trajectories)
    x_data_second_order_TR, _, u_data_second_order_TR, _, clc_data_second_order_TR, _, clc_sum_second_order_TR, _ = extract_trajectories(x_data_second_order_TR, u_data_second_order_TR, clc_data_second_order_TR, clc_sum_second_order_TR, feasible_trajectories)
    n_feasible_benchmark = x_data_benchmark.shape[0]
    n_infeasible_benchmark = 0
    clc_mean_benchmark = clc_sum_benchmark.mean()
    clc_std_benchmark = clc_sum_benchmark.std()
    p_feasible = 1
    p_infeasible = 0
    max_constraint_violation = 0
    mean_constraint_violation = 0
    n_violations_benchmark = 0
    n_total = np.prod(x_data_benchmark.shape[:-1])

    # Evaluate data
    n_feasible_untrained, n_infeasible_untrained, p_feasible_untrained, p_infeasible_untrained, clc_mean_all_untrained, clc_mean_untrained, clc_std_untrained, max_constraint_violation_untrained, n_violations_untrained, mean_constraint_violation_untrained = evaluate(x_data_untrained, u_data_untrained, clc_data_untrained, clc_sum_untrained)
    n_feasible_first_order, n_infeasible_first_order, p_feasible_first_order, p_infeasible_first_order, clc_mean_all_first_order, clc_mean_first_order, clc_std_first_order, max_constraint_violation_first_order, n_violations_first_order, mean_constraint_violation_first_order = evaluate(x_data_first_order, u_data_first_order, clc_data_first_order, clc_sum_first_order)
    n_feasible_first_order_TR, n_infeasible_first_order_TR, p_feasible_first_order_TR, p_infeasible_first_order_TR, clc_mean_all_first_order_TR, clc_mean_first_order_TR, clc_std_first_order_TR, max_constraint_violation_first_order_TR, n_violations_first_order_TR, mean_constraint_violation_first_order_TR = evaluate(x_data_first_order_TR, u_data_first_order_TR, clc_data_first_order_TR, clc_sum_first_order_TR)
    n_feasible_second_order, n_infeasible_second_order, p_feasible_second_order, p_infeasible_second_order, clc_mean_all_second_order, clc_mean_second_order, clc_std_second_order, max_constraint_violation_second_order, n_violations_second_order, mean_constraint_violation_second_order = evaluate(x_data_second_order, u_data_second_order, clc_data_second_order, clc_sum_second_order)
    n_feasible_second_order_TR, n_infeasible_second_order_TR, p_feasible_second_order_TR, p_infeasible_second_order_TR, clc_mean_all_second_order_TR, clc_mean_second_order_TR, clc_std_second_order_TR, max_constraint_violation_second_order_TR, n_violations_second_order_TR, mean_constraint_violation_second_order_TR = evaluate(x_data_second_order_TR, u_data_second_order_TR, clc_data_second_order_TR, clc_sum_second_order_TR)

    # Save data
    index = ["Benchmark", "Untrained", "First Order", "First Order TR", "Second Order", "Second Order TR"]
    columns = ["Feasible Trajectories", "Infeasible Trajectories", "Feasible Percentage", "Infeasible Percentage", "CLC Mean All", "CLC Mean", "CLC Std", "Max Constraint Violation", "All Points", "Violation Points", "Mean Constraint Violation"]
    clc_data = pd.DataFrame(
        data = [
            [n_feasible_benchmark, n_infeasible_benchmark, p_feasible, p_infeasible, clc_mean_benchmark, clc_mean_benchmark, clc_std_benchmark, max_constraint_violation, n_total, n_violations_benchmark, mean_constraint_violation],
            [n_feasible_untrained, n_infeasible_untrained, p_feasible_untrained, p_infeasible_untrained, clc_mean_all_untrained, clc_mean_untrained, clc_std_untrained, max_constraint_violation_untrained, n_total, n_violations_untrained, mean_constraint_violation_untrained],
            [n_feasible_first_order, n_infeasible_first_order, p_feasible_first_order, p_infeasible_first_order, clc_mean_all_first_order, clc_mean_first_order, clc_std_first_order, max_constraint_violation_first_order, n_total, n_violations_first_order, mean_constraint_violation_first_order],
            [n_feasible_first_order_TR, n_infeasible_first_order_TR, p_feasible_first_order_TR, p_infeasible_first_order_TR, clc_mean_all_first_order_TR, clc_mean_first_order_TR, clc_std_first_order_TR, max_constraint_violation_first_order_TR, n_total, n_violations_first_order_TR, mean_constraint_violation_first_order_TR],
            [n_feasible_second_order, n_infeasible_second_order, p_feasible_second_order, p_infeasible_second_order, clc_mean_all_second_order, clc_mean_second_order, clc_std_second_order, max_constraint_violation_second_order, n_total, n_violations_second_order, mean_constraint_violation_second_order],
            [n_feasible_second_order_TR, n_infeasible_second_order_TR, p_feasible_second_order_TR, p_infeasible_second_order_TR, clc_mean_all_second_order_TR, clc_mean_second_order_TR, clc_std_second_order_TR, max_constraint_violation_second_order_TR, n_total, n_violations_second_order_TR, mean_constraint_violation_second_order_TR]
            ],
        index = index,
        columns = columns
        )
    
    clc_data.to_excel(save_path)
    pass