import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from utils_tools import *
from repair import *
from OTClean.main import randomize_test_data

def main(dataset_name, constraint_type, L_size, max_iter, thr):
    """
    Main function to run fairness experiments on different datasets
    
    Args:
        dataset_name: Name of dataset to use ('Adult', 'Compas', or 'Census-KDD')
        constraint_type: Type of fairness constraint to apply ('ambiguous', 'absent', or dataset name)
        L_size: Size of latent variable space
        max_iter: Maximum number of iterations for optimization
        thr: Convergence threshold
    """
    print(f'\nExecuted code: {get_current_code_info():}\n')

    # Load and preprocess dataset
    print('Start load dataset ...', flush=True)
    _, dataRaw = load_dataset(dataset_name)
    # Get feature groupings (S: sensitive, I: intermediate, A: admissible, W: world features, Y: target)
    S, I, A, W, Y = specify_constraint(dataset_name, constraint_type)  
    print(f'Dataset: {dataset_name}; \nConstraint: {S},{I},{A},{W},{Y}')

    # Convert categorical variables to numeric using factorization
    data = dataRaw[S+I+A+W+Y].copy()
    for column in data.columns:
        data[column] = pd.factorize(data[column])[0]
    X = data[S+I+A+W].values  # Features
    y = data[Y].values.flatten()  # Target variable

    # Setup 5-fold cross validation with 1 repeat
    print('Start split dataset ...', flush=True)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

    # Define methods and models to evaluate based on dataset
    # Methods: Latent (LatentPre, our method), Lazy (CausalPre), Cap-MS/MF (from capuchin), OTClean/OTClean_RT (from OTClean), Original (no repair), Dropped (remove S+I features)
    # Models: Random Forest and Multi-Layer Perceptron
    if dataset_name == 'Compas':
        methods = ['Latent', 'Lazy', 'Cap-MS', 'Cap-MF', 'OTClean', 'OTClean_RT', 'Original', 'Dropped']
        models = ['RF', 'MLP']
    elif dataset_name == 'Census-KDD':
        methods = ['Latent', 'Lazy', 'Cap-MF', 'Original', 'Dropped']
        models = ['RF', 'MLP']
    elif dataset_name == 'Adult':
        methods = ['Latent', 'Lazy', 'Cap-MS', 'Cap-MF', 'Original', 'Dropped']
        models = ['RF', 'MLP']
    else:
        print('ERROR! Please identify the method list and model list.')
        methods = []
        models = []

    # Initialize results storage arrays
    # Format: [method][model][fold_results]
    auc_list = [[[] for _ in range(len(models))] for _ in range(len(methods))]  # Store AUC scores
    rod_list = [[[] for _ in range(len(models))] for _ in range(len(methods))]  # Store ROD (discrimination) scores
    repairtime_list = [[] for _ in range(len(methods))]  # Store repair times
    round_cnt = 0

    # Run cross validation experiments
    for train, test in cv.split(X, y):
        round_cnt += 1
        print(f"Round {round_cnt} Starts ...", flush=True)
        for method_i, method in enumerate(methods):
            # Create deep copies of data for this fold
            train_data = data.copy(deep=True).iloc[train]
            test_data = data.copy(deep=True).iloc[test]
            print(f"  =>=> #{method_i}:{method} is processing ", flush=True)
            repairtime = 0

            # Apply fairness repair method based on selected approach
            if method in ['Latent', 'Lazy', 'Cap-MS', 'Cap-MF', 'OTClean']:
                data_fair, repairtime = repair_nav(method, train_data, [S, I, A, W, Y], L_size=L_size, max_iter=max_iter, thr=thr)
                X_train = data_fair[S+I+A+W].values
                y_train = data_fair[Y].values.flatten()
                X_test = test_data[S+I+A+W].values
                y_test = test_data[Y].values.flatten()
            elif method in ['OTClean_RT']:
                data_fair, repairtime, Gs_marginal, x_domain_marginal, target_domain_marginal = repair_nav(method, train_data, [S, I, A, W, Y], dataset_name, alpha)
                X_train = data_fair[I+A+W].values
                y_train = data_fair[Y].values.flatten()
                mapping_cols = S + I + A
                test_data_rep = randomize_test_data(test_data, Gs_marginal, mapping_cols, x_domain_marginal, target_domain_marginal, most_likely=False)
                X_test = test_data_rep[I+A+W].values
                y_test = test_data_rep[Y].values.flatten()
                test_data = test_data_rep.copy() 
            elif method == 'Dropped':
                # Drop both S and I features
                X_train = train_data[A+W].values
                y_train = train_data[Y].values.flatten()
                X_test = test_data[A+W].values
                y_test = test_data[Y].values.flatten()
            else:
                # Original data without repair
                X_train = train_data[S+I+A+W].values
                y_train = train_data[Y].values.flatten()
                X_test = test_data[S+I+A+W].values
                y_test = test_data[Y].values.flatten()

            repairtime_list[method_i].append(repairtime)
            print(f"Repair Time: {repairtime} ")

            # Train and evaluate models
            constraint = specify_constraint(dataset_name, constraint_type) if 'absent' in constraint_type else specify_constraint(dataset_name, dataset_name)
            for m_i, m in enumerate(models):
                auc, rod = RunClassifier(m, X_train, y_train, X_test, y_test, test_data, [S, I, A, W, Y], constraint, dataset_name)
                auc_list[method_i][m_i].append(auc)
                rod_list[method_i][m_i].append(rod)
    
    sum_all_results(methods, models, auc_list, rod_list, repairtime_list)


if __name__ == "__main__":
    # Command line arguments (commented out)
    # dataset_name, constraint_type, L_size, max_iter, thr = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    # print(f'\nInput parameters:\n dataset_name: {dataset_name}, constraint_type: {constraint_type}, L_size: {L_size}, max_iter: {max_iter}, thr: {thr}\n')
    
    # Default parameters for testing
    dataset_name = 'Adult'  # Dataset to use
    constraint_type = 'ambiguous'  # Type of fairness constraint
    L_size = 6  # Size of latent space
    max_iter = 800  # Maximum optimization iterations
    thr = 0.001  # Convergence threshold
    main(dataset_name, constraint_type, L_size, max_iter, thr)