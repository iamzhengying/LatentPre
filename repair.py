# Import required libraries
import pandas as pd
import numpy as np
import time
import itertools
from multiprocessing import Pool
from pyitlib.discrete_random_variable import information_mutual, information_mutual_conditional
from scipy.stats import chi2_contingency
from causallearn.utils.cit import CIT
from utils_tools import * 


def repair_nav(method, train_data, constraint, L_size=4, max_iter=int(5e3), thr=1e-3):
    """
    Main navigation function for data repair methods.
    
    Args:
        method: Repair method to use ('Latent', 'Lazy', 'Cap-MS', 'Cap-MF')
        train_data: Training data to repair
        constraint: Tuple of (S,I,A,W,Y) constraint variables
        type: Type parameter (unused)
        alpha: Alpha parameter (unused) 
        L_size: Size of latent variable space
        max_iter: Maximum iterations for EM
        thr: Convergence threshold
        num_samples: Number of samples to generate
        
    Returns:
        data_fair: Repaired fair dataset
        runtime: Total runtime in seconds
    """
    start_time = time.time()

    if method == 'Latent':
        data_fair = repair_latent(train_data, constraint, L_size=L_size, max_iter=max_iter, thr=thr)
        print(f'Complete "repair_latent" at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}...', flush=True)
    
    elif method == 'Lazy':
        pass
        
    elif method == 'Cap-MS':
        pass

    elif method == 'Cap-MF':
        pass

    elif method == 'OTClean' or method == 'OTClean_RT':
        pass  
    
    else:
        print(f"ERROR: no repair since cannot successfully detect the input repair method [{method}].", flush=True)
        data_fair = None
    
    end_time = time.time() 
    runtime = end_time - start_time
    
    return data_fair, runtime



def repair_latent(train_data, constraint, L_size, max_iter=int(5e3), thr=1e-3):
    """
    Main latent variable repair method using EM algorithm.
    
    Args:
        train_data: Training data to repair
        constraint: Tuple of (S,I,A,W,Y) constraint variables
        L_size: Size of latent variable space
        max_iter: Maximum iterations for EM
        thr: Convergence threshold
        
    Returns:
        Repaired fair dataset
    """
    print(f'Repair_latent in processing ...')
    print(f'Info: L_size={L_size}, max_iter={max_iter}, thr={thr}', flush=True)
    S, I, A, W, Y = constraint

    ### step-1: initialization
    print(f'Start initialization at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}...', flush=True)
    np.random.seed(42)
    conditioning_set_size = 2
    # Identify parents of I variables
    paI = identify_parent_I(train_data, constraint, max_size=conditioning_set_size)
    npI = list(set(I)-set(paI))

    # Partition parents into two groups
    paI_1, paI_2, state_bound = parent_I_partition(train_data, paI, S+npI+A, L_size)
    if paI_1 is None:
        print(f'[ERROR] Do not have enough attributes for partition in paI.', flush=True)
        return None 
    
    if L_size > state_bound:
        print(f'[ERROR] L_size={L_size} is greater than its upper bound={state_bound}.', flush=True)
        # return None
    
    # Create combined columns for more efficient processing
    column_mappings = [('S', S), ('I0', npI), ('I1', paI_1), ('I2', paI_2), ('A', A), ('W', W), ('Y', Y)]
    columns_to_create = [(name, cols) for name, cols in column_mappings if cols]
    
    for name, cols in columns_to_create:
        train_data[name] = train_data[cols].apply(tuple, axis=1)
    
    # Select only created columns for dense representation
    created_columns = [name for name, _ in columns_to_create]
    train_data_dense = train_data[created_columns]
    train_data = train_data.drop(columns=created_columns)
    
    # Initialize probability distributions
    P_L = np.random.dirichlet(np.ones(L_size))
    
    # Get unique values for each variable
    unique_I1 = train_data_dense['I1'].unique()
    unique_I2 = train_data_dense['I2'].unique()
    unique_Y = train_data_dense['Y'].unique()
    
    # Create column combinations for grouping
    s_i0_a_columns = [col for col in ['S', 'I0', 'A'] if col in train_data_dense.columns]
    a_w_columns = [col for col in ['A', 'W'] if col in train_data_dense.columns]
    
    unique_S_I0_A = train_data_dense[s_i0_a_columns].apply(tuple, axis=1).unique()
    unique_A_W = train_data_dense[a_w_columns].apply(tuple, axis=1).unique()

    # Initialize conditional probability matrices
    P_I1_given_L_S_I0_A = np.random.dirichlet(np.ones(len(unique_I1)), size=(len(P_L) * len(unique_S_I0_A)))
    P_I2_given_L_S_I0_A = np.random.dirichlet(np.ones(len(unique_I2)), size=(len(P_L) * len(unique_S_I0_A)))
    P_Y_given_L_A_W = np.random.dirichlet(np.ones(len(unique_Y)), size=(len(P_L) * len(unique_A_W)))

    # Create lookup dictionaries for efficient indexing
    I1_dict = {val: idx for idx, val in enumerate(unique_I1)}
    I2_dict = {val: idx for idx, val in enumerate(unique_I2)}
    Y_dict = {val: idx for idx, val in enumerate(unique_Y)}
    
    L_S_I0_A_dict = {(l,) + combo: idx for idx, (l, combo) in 
                       enumerate([(l, combo) for l in range(L_size) for combo in unique_S_I0_A])}
    L_A_W_dict = {(l,) + combo: idx for idx, (l, combo) in 
                  enumerate([(l, combo) for l in range(L_size) for combo in unique_A_W])}
    
    ### step-2: Vectorized EM algorithm
    max_iterations = max_iter
    tolerance = thr
    print(f'Start EM algorithm (iter={max_iterations}, tol={tolerance}) at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}...', flush=True)

    log_likelihoods = []
    data_values = train_data_dense.values
    
    # Helper function to get combinations from rows
    def get_combo_from_row(row, columns):
        return tuple(row[train_data_dense.columns.get_loc(col)] for col in columns)
    
    # Pre-compute indices for faster lookup
    s_i0_a_indices = np.array([L_S_I0_A_dict[(l,) + get_combo_from_row(row, s_i0_a_columns)] 
                                for l in range(L_size) for row in data_values]).reshape(L_size, -1)
    i1_indices = np.array([I1_dict[row[train_data_dense.columns.get_loc('I1')]] for row in data_values])
    i2_indices = np.array([I2_dict[row[train_data_dense.columns.get_loc('I2')]] for row in data_values])
    a_w_indices = np.array([L_A_W_dict[(l,) + get_combo_from_row(row, a_w_columns)] 
                           for l in range(L_size) for row in data_values]).reshape(L_size, -1)
    y_indices = np.array([Y_dict[row[train_data_dense.columns.get_loc('Y')]] for row in data_values])

    # Start EM iterations
    for iteration in range(max_iterations):
        # E-step: Compute posterior probabilities P(L|X)
        P_I1_probs = np.maximum(P_I1_given_L_S_I0_A[s_i0_a_indices, i1_indices], 1e-30)
        P_I2_probs = np.maximum(P_I2_given_L_S_I0_A[s_i0_a_indices, i2_indices], 1e-30)
        P_Y_probs = np.maximum(P_Y_given_L_A_W[a_w_indices, y_indices], 1e-30)
        
        # Calculate log probabilities with numerical stability
        log_probs = (np.log(P_L).reshape(-1,1) + np.log(P_I1_probs) + np.log(P_I2_probs) + np.log(P_Y_probs))
        max_log_probs = np.max(log_probs, axis=0, keepdims=True)
        exp_probs = np.exp(log_probs - max_log_probs)
        posterior_probs = exp_probs / np.sum(exp_probs, axis=0)

        # M-step: Update model parameters
        P_L = np.mean(posterior_probs, axis=1)
        
        # Update P(I1|L,S,I0,A)
        counts_I1 = np.zeros((L_size * len(unique_S_I0_A), len(unique_I1)))
        np.add.at(counts_I1, (s_i0_a_indices.ravel(), np.tile(i1_indices, L_size)), posterior_probs.ravel())
        norm_sums = np.sum(counts_I1, axis=1, keepdims=True)
        norm_sums[norm_sums == 0] = 1
        P_I1_given_L_S_I0_A = counts_I1 / norm_sums
        P_I1_global = counts_I1.sum(axis=0) / counts_I1.sum()
        P_I1_given_L_S_I0_A[P_I1_given_L_S_I0_A.sum(axis=1) == 0] = P_I1_global

        # Update P(I2|L,S,I0,A)
        counts_I2 = np.zeros((L_size * len(unique_S_I0_A), len(unique_I2)))
        np.add.at(counts_I2, (s_i0_a_indices.ravel(), np.tile(i2_indices, L_size)), posterior_probs.ravel())
        norm_sums = np.sum(counts_I2, axis=1, keepdims=True)
        norm_sums[norm_sums == 0] = 1
        P_I2_given_L_S_I0_A = counts_I2 / norm_sums
        P_I2_global = counts_I2.sum(axis=0) / counts_I2.sum()
        P_I2_given_L_S_I0_A[P_I2_given_L_S_I0_A.sum(axis=1) == 0] = P_I2_global

        # Update P(Y|L,A,W)
        counts_Y = np.zeros((L_size * len(unique_A_W), len(unique_Y)))
        np.add.at(counts_Y, (a_w_indices.ravel(), np.tile(y_indices, L_size)), posterior_probs.ravel())
        norm_sums = np.sum(counts_Y, axis=1, keepdims=True)
        norm_sums[norm_sums == 0] = 1
        P_Y_given_L_A_W = counts_Y / norm_sums
        P_Y_global = counts_Y.sum(axis=0) / counts_Y.sum()
        P_Y_given_L_A_W[P_Y_given_L_A_W.sum(axis=1) == 0] = P_Y_global
        
        # Compute log-likelihood for convergence check
        P_I1_probs = np.maximum(P_I1_given_L_S_I0_A[s_i0_a_indices, i1_indices], 1e-30)
        P_I2_probs = np.maximum(P_I2_given_L_S_I0_A[s_i0_a_indices, i2_indices], 1e-30)
        P_Y_probs = np.maximum(P_Y_given_L_A_W[a_w_indices, y_indices], 1e-30)
        log_probs = (np.log(P_L).reshape(-1,1) + np.log(P_I1_probs) + np.log(P_I2_probs) + np.log(P_Y_probs))
        max_log_probs = np.max(log_probs, axis=0, keepdims=True)
        exp_probs = np.exp(log_probs - max_log_probs)
        log_likelihood = np.sum(max_log_probs + np.log(np.sum(exp_probs, axis=0)))
        log_likelihoods.append(log_likelihood)

        if np.isnan(np.array(log_likelihoods)).any(): break

        if iteration % 10 == 0:
            print(f'Iteration {iteration}: Log-Likelihood = {log_likelihood}, at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}...', flush=True)

        # Check convergence
        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tolerance:
            print(f"Converged after {iteration} iterations. The final log-likelihood is {log_likelihoods[-1]}.")
            break

    ### step-3: store results

    ### step-4: compute P(SIAWY) and sample
    new_constraint = (S, npI, paI_1, paI_2, A, W, Y)
    sample_args = (train_data_dense, len(train_data_dense), new_constraint, s_i0_a_columns, L_size, P_L, 
                   P_I1_given_L_S_I0_A, P_I2_given_L_S_I0_A, P_Y_given_L_A_W, L_S_I0_A_dict, L_A_W_dict, 
                   unique_I1, unique_I2, unique_Y)
    return sample_from_cond_prob(sample_args)




def sample_from_cond_prob(args):
    """
    Sample from learned conditional probability distributions to generate fair data.
    
    Args:
        args: Tuple containing all necessary parameters and probability distributions
        
    Returns:
        data_fair: Generated fair dataset
    """
    print(f'Start sampling at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}...', flush=True)
    (train_data_dense, num_samples, new_constraint, s_i0_a_columns, L_size, P_L, 
     P_I1_given_L_S_I0_A, P_I2_given_L_S_I0_A, P_Y_given_L_A_W, L_S_I0_A_dict, L_A_W_dict, 
     unique_I1, unique_I2, unique_Y) = args
    S, npI, paI_1, paI_2, A, W, Y = new_constraint
    
    # Calculate empirical distribution of observed variables
    s_i0_a_w_columns = [col for col in ['S', 'I0', 'A', 'W'] if col in train_data_dense.columns]
    P_SI0AW = train_data_dense.groupby(s_i0_a_w_columns).size().div(len(train_data_dense))
    P_SI0AW = P_SI0AW.reset_index().rename(columns={0: 'P(SI0AW)'})

    # Sample from joint distribution
    np.random.seed(42) 
    si0aw_indices = np.random.choice(P_SI0AW.index, size=num_samples, p=P_SI0AW['P(SI0AW)'].values)
    sampled_data = P_SI0AW.loc[si0aw_indices, s_i0_a_w_columns]
    sampled_data['L'] = np.random.choice(range(L_size), size=num_samples, p=P_L)

    # Sample I1 and I2 values
    l_s_i0_a_idx = [L_S_I0_A_dict[(l,) + tuple(row)] 
                      for l, *row in sampled_data[['L'] + s_i0_a_columns].itertuples(index=False)]
    sampled_data['I1'] = [np.random.choice(unique_I1, p=P_I1_given_L_S_I0_A[idx].ravel()) for idx in l_s_i0_a_idx]
    sampled_data['I2'] = [np.random.choice(unique_I2, p=P_I2_given_L_S_I0_A[idx].ravel()) for idx in l_s_i0_a_idx]

    # Sample Y values
    a_w_columns = [col for col in ['A', 'W'] if col in train_data_dense.columns]
    l_a_w_idx = [L_A_W_dict[(l,) + tuple(row)] for l, *row in sampled_data[['L'] + a_w_columns].itertuples(index=False)]
    sampled_data['Y'] = [np.random.choice(unique_Y, p=P_Y_given_L_A_W[idx].ravel()) for idx in l_a_w_idx]

    # Drop latent variable and recover original format
    data_fair = sampled_data.drop('L', axis=1)
    
    # Expand combined columns back to original format
    column_expansions = [('S', S), ('I0', npI), ('I1', paI_1), ('I2', paI_2), ('A', A), ('W', W), ('Y', Y)]
    columns_to_expand = [(name, cols) for name, cols in column_expansions if cols and name in data_fair.columns]
    
    for name, cols in columns_to_expand:
        expanded_cols = pd.DataFrame(data_fair[name].tolist(), columns=cols)
        data_fair = pd.concat([data_fair.drop(columns=[name]).reset_index(drop=True), 
                              expanded_cols.reset_index(drop=True)], axis=1)
    return data_fair




def process_independence_test(args):
    """
    Helper function to process independence tests in parallel.
    
    Args:
        args: Tuple containing test parameters
        
    Returns:
        Tuple of (variable, is_independent, conditioning_set, p_value)
    """
    i, data, col_to_index, Y, alpha, size, A, W, I, removed_I = args
    if size == 0:
        # Test marginal independence
        contingency_table = pd.crosstab(data[:, col_to_index[i]], data[:, col_to_index[Y[0]]])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        return (i, p_value > alpha, None, p_value)
    else:
        # Test conditional independence
        for combo in itertools.combinations(A+W+removed_I, size):
            cit = CIT(data, method='gsq')
            p_value = cit(col_to_index[i], col_to_index[Y[0]], [col_to_index[x] for x in combo])
            if p_value > alpha:  # independent
                return (i, True, combo, p_value)
        return (i, False, None, None)

def identify_parent_I(train_data, constraint, alpha=0.05, max_size=3):
    """
    Identify parent variables of I using parallel processing.
    Tests both marginal and conditional independence.
    Avoids mutual screening by only conditioning on confirmed independent variables.
    
    Args:
        train_data: Training data
        constraint: Tuple of (S,I,A,W,Y) constraint variables
        alpha: Significance level for independence tests
        max_size: Maximum size of conditioning sets
        
    Returns:
        List of identified parent variables
    """
    S, I, A, W, Y = constraint
    candidate_paI = I.copy()  # Start with all I variables as candidates
    removed_I = []  # Track I variables confirmed as independent
    data = train_data.to_numpy()
    col_to_index = {name: idx for idx, name in enumerate(train_data.columns)}
    
    # Test independence at each conditioning set size
    for size in range(max_size+1):
        iteration_ele = candidate_paI.copy()
        if not iteration_ele:  # Exit if no candidates remain
            break
            
        # Prepare arguments for parallel processing
        args_list = [(i, data, col_to_index, Y, alpha, size, A, W, I, removed_I) 
                    for i in iteration_ele]
        
        # Process independence tests in parallel
        with Pool(processes=16) as pool:    # with Pool(processes=16) as pool
            results = pool.map(process_independence_test, args_list)
            
        # Process results and update candidate set
        for result in results:
            i, is_independent, combo, p_value = result
            if is_independent:
                if size == 0:
                    print(f'{i} is independent of {Y[0]}: {p_value}')
                else:
                    print(f'{i} is independent of {Y[0]} given {combo}: {p_value}')
                candidate_paI.remove(i)
                removed_I.append(i)  # Add to removed set
                
        print(f'Round {size}: remaining candidates = {candidate_paI}')
    
    print(f'Complete paI identification at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}...', flush=True)
    return candidate_paI




def total_cmi(train_data, partition1, partition2, cond_attrs):
    """
    Calculate total conditional mutual information between two partitions.
    
    Args:
        train_data: Training data
        partition1: First set of variables
        partition2: Second set of variables
        cond_attrs: Conditioning variables
        
    Returns:
        Total CMI value
    """
    total = 0
    for x in partition1:
        for y in partition2:
            x_data = train_data[x].values
            y_data = train_data[y].values
            if not cond_attrs:
                total += information_mutual(x_data, y_data)
            else:
                if len(cond_attrs) == 1:
                    z_data = train_data[cond_attrs[0]].values
                else:
                    # Combine multiple conditioning variables into single array
                    z_data = train_data[cond_attrs].apply(tuple, axis=1).values
                    # Convert tuples to integer codes for pyitlib
                    z_unique = {val: idx for idx, val in enumerate(np.unique(z_data))}
                    z_data = np.array([z_unique[val] for val in z_data])
                total += information_mutual_conditional(x_data, y_data, z_data)
    return total

def parent_I_partition(train_data, paI, cond, L_size, epsilon=1e-5):
    """
    Partition paI into two disjoint subsets paI_1 and paI_2 to minimize CMI.
    Uses local search optimization.
    
    Args:
        train_data: Training data
        paI: List of parent variables to partition
        cond: Conditioning variables
        L_size: Size of latent variable space
        epsilon: Convergence threshold
        
    Returns:
        paI_1, paI_2: Two disjoint subsets of paI
        state_bound: Upper bound on latent variable states
    """
    # Handle edge cases
    if len(paI) < 2:
        return None, None
    
    # Calculate domain sizes for each attribute
    domain_sizes = {attr: train_data[attr].nunique() for attr in paI}
    
    # Define helper functions
    tau = np.log(L_size)
    w = lambda x: np.log(domain_sizes[x])
    W = lambda attr_set: sum([w(i) for i in attr_set])
    
    # Check if tau is reasonable (should be less than total log domain)
    total_log_domain = W(paI)
    if tau >= total_log_domain / 2:
        print(f'[WARNING] tau={tau:.2f} is not reasonable (too large)')
        tau = min(w(i) for i in paI)
        print(f'[INFO] Now enforce tau={tau:.2f}.', flush=True)
        # return None, None, None
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize with random bipartition
    paI_shuffled = paI.copy()
    np.random.shuffle(paI_shuffled)
    split_idx = np.random.randint(1, len(paI_shuffled))
    paI_1 = set(paI_shuffled[:split_idx])
    paI_2 = set(paI_shuffled[split_idx:])
    
    # Check feasibility and adjust if needed
    while W(paI_1) < tau or W(paI_2) < tau:
        if len(paI_1) == 0 or len(paI_2) == 0:
            return None, None, None
        
        # Move attribute with largest domain from larger partition to smaller
        if W(paI_1) > W(paI_2):
            largest = np.random.choice(list(paI_1))
            paI_1 = paI_1 - {largest}
            paI_2 = paI_2 | {largest}
        else:
            largest = np.random.choice(list(paI_2))
            paI_2 = paI_2 - {largest}
            paI_1 = paI_1 | {largest}
    
    # Calculate initial CMI
    current_cmi = total_cmi(train_data, paI_1, paI_2, cond)
    
    # Main optimization loop
    while True:
        best_delta = 0
        best_partition = None
        
        # Try 1-moves: move each attribute to opposite partition
        for v in paI:
            if v in paI_1:
                new_paI_1 = paI_1 - {v}
                new_paI_2 = paI_2 | {v}
            else:
                new_paI_1 = paI_1 | {v}
                new_paI_2 = paI_2 - {v}
            
            # Check feasibility
            if W(new_paI_1) >= tau and W(new_paI_2) >= tau:
                new_cmi = total_cmi(train_data, new_paI_1, new_paI_2, cond)
                delta = current_cmi - new_cmi
                
                if delta > best_delta:
                    best_delta = delta
                    best_partition = (new_paI_1, new_paI_2)
        
        # If no improving 1-move, try 2-swaps
        if best_delta == 0:
            for u in paI_1:
                for v in paI_2:
                    new_paI_1 = (paI_1 - {u}) | {v}
                    new_paI_2 = (paI_2 - {v}) | {u}
                    
                    # Check feasibility
                    if W(new_paI_1) >= tau and W(new_paI_2) >= tau:
                        new_cmi = total_cmi(train_data, new_paI_1, new_paI_2, cond)
                        delta = current_cmi - new_cmi
                        
                        if delta > best_delta:
                            best_delta = delta
                            best_partition = (new_paI_1, new_paI_2)
        
        # No improvement found, terminate
        if best_delta < epsilon:
            break
        
        # Update with best partition found
        paI_1, paI_2 = best_partition
        current_cmi -= best_delta
    
    # Convert sets back to lists for return
    return list(paI_1), list(paI_2), int(round(np.exp(min(W(paI_1), W(paI_2)))))



