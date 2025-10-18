import numpy as np
import pandas as pd
import inspect
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

from utils_clean import *


def load_dataset(name, filepath='./data/'):
    """
    Load and preprocess a dataset based on the given name.
    
    Args:
        name: Name of dataset to load ('Adult', 'Compas', or 'Census-KDD')
        filepath: Path to data directory
        
    Returns:
        Tuple of (raw_data, preprocessed_data)
    """
    dataRaw0, dataRaw = None, None
    if name == "Adult":
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        dataRaw0 = pd.read_csv(url, header=None, 
                            names=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", 
                                        "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", 
                                        "native-country", "income"], 
                            sep=r',\s*', engine='python')
        dataRaw = clean_adult(dataRaw0)
    elif name == 'Compas':
        dataRaw0 = pd.read_csv(filepath + 'compas-scores-two-years.csv', index_col=0)
        dataRaw = clean_compas(dataRaw0)
    elif name == 'Census-KDD':
        url = 'https://archive.ics.uci.edu/static/public/117/data.csv'
        dataRaw0 = pd.read_csv(url, sep=r',\s*', engine='python')
        dataRaw = clean_census(dataRaw0)
    else: 
        print("ERROR: Please input a resonable dataset name")
    # data = dataRaw[S+I+A+Y]
    return dataRaw0, dataRaw


def specify_constraint(dataset):
    """
    Specify fairness constraint variables for a given dataset.
    
    Args:
        dataset: Name of dataset ('Adult', 'Compas', or 'Census-KDD')
        
    Returns:
        Tuple of (S, I, A, W, Y) constraint variables where:
        S: Protected attributes
        I: Intermediate variables
        A: Actionable attributes
        W: Non-actionable attributes
        Y: Target variable
    """
    S, I, A, W, Y = None, None, None, None, None
    if dataset == 'Adult':
        if type == 'Adult':
            Y = ['income']
            S = ['sex', 'race']
            I = ['marital-status', 'relationship', 'native-country']
            A = ['hours-per-week', 'occupation', 'education-num', 'workclass']
            W = ['age', 'capital-gain', 'capital-loss'] 
        elif type == 'ambiguous':
            Y = ['income']
            S = ['sex', 'race']
            I = ['marital_status', 'relationship', 'native_country', 'age']
            A = ['hours_per_week', 'occupation', 'education_num', 'workclass']
            W = ['capital_gain', 'capital_loss'] 
        elif type == 'absent':
            Y = ['income']
            S = ['sex', 'race']
            I = ['marital_status', 'relationship', 'native_country']
            A = ['hours_per_week', 'occupation', 'education_num', 'workclass']
            W = ['capital_gain', 'capital_loss'] 

    elif dataset == 'Compas':
        if type == 'Compas':
            Y = ['two_year_recid']
            S = ['race', 'sex']
            I = ['age_cat', 'score_text']
            A = ['priors_count']
            W = ['length_of_stay', 'c_charge_degree']
        elif type == 'ambiguous':
            Y = ['two_year_recid']
            S = ['race', 'sex']
            I = ['age_cat', 'score_text', 'c_charge_degree']
            A = ['priors_count']
            W = ['length_of_stay']
        elif type == 'absent':
            Y = ['two_year_recid']
            S = ['race', 'sex']
            I = ['age_cat', 'score_text']
            A = ['priors_count']
            W = ['length_of_stay']
    
    elif dataset == 'Census-KDD':
        if type == 'Census-KDD':
            Y = ['income']
            S = ['sex', 'race']
            I = ['marital_status', 'region_of_previous_residence', 'detailed_household_summary_in_household', 
                'country_of_birth_self', 'citizenship', 'full_or_part_time_employment_stat', 
                'family_members_under_18', 'num_persons_worked_for_employer', 'veterans_benefits', 'age',
                'reason_for_unemployment', 'major_industry_code']
            A = ['class_of_worker', 'education', 'wage_per_hour', 
                'occupation_code', 'weeks_worked_in_year']
            W = ['dividends_from_stocks', 'capital_gains',
                'fill_inc_questionnaire_for_veteran_admin', 'year', 'instance_weight',
                'own_business_or_self_employed', 'capital_loss',
                'member_of_a_labor_union']
        elif type == 'ambiguous':
            Y = ['income']
            S = ['sex', 'race']
            I = ['marital_status', 'region_of_previous_residence', 'detailed_household_summary_in_household', 
                'country_of_birth_self', 'citizenship', 'full_or_part_time_employment_stat', 
                'family_members_under_18', 'num_persons_worked_for_employer', 'veterans_benefits', 'age',
                'reason_for_unemployment', 'major_industry_code', 
                'weeks_worked_in_year', 'member_of_a_labor_union', 
                'education', 'year', 'instance_weight', 'occupation_code']
            A = ['class_of_worker', 'wage_per_hour']
            W = ['dividends_from_stocks', 'capital_gains',
                'fill_inc_questionnaire_for_veteran_admin',
                'own_business_or_self_employed', 'capital_loss']
        elif type == 'absent':
            Y = ['income']
            S = ['sex', 'race']
            I = ['marital_status', 'region_of_previous_residence', 'detailed_household_summary_in_household', 
                'country_of_birth_self', 'citizenship', 'full_or_part_time_employment_stat', 
                'family_members_under_18', 'num_persons_worked_for_employer', 'veterans_benefits', 'age',
                'reason_for_unemployment', 'major_industry_code']
            A = ['class_of_worker', 'wage_per_hour']
            W = ['dividends_from_stocks', 'capital_gains',
                'fill_inc_questionnaire_for_veteran_admin',
                'own_business_or_self_employed', 'capital_loss']

    else: 
        print("ERROR: Please input a resonable dataset name")
    return S, I, A, W, Y


def RunClassifier(m, X_train, y_train, X_test, y_test, test_data, constraint, dataset):
    """
    Train and evaluate a classifier model.
    
    Args:
        m: Model type ('LR', 'RF', or 'MLP')
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        test_data: Full test dataset
        constraint: Tuple of (S,I,A,W,Y) constraint variables
        dataset: Name of dataset
        
    Returns:
        Tuple of (AUC score, ROD discrimination score)
    """
    [S, I, A, W, Y] = constraint
    if m == 'LR':
        print(f"\nLogistic Regression Performance: ")
        model = LogisticRegression(class_weight='balanced', max_iter=10000, random_state=42)
        model.fit(X_train, y_train)
    elif m == 'RF':
        print(f"\nRandom Forest Performance: ")
        if dataset == 'Adult': model = RandomForestClassifier(class_weight='balanced', max_depth=9, n_estimators=50, max_features='log2', random_state=42)
        elif dataset == 'Compas': model = RandomForestClassifier(class_weight='balanced', max_depth=5, n_estimators=10, max_features="log2", random_state=42)
        elif dataset == 'Census-KDD': model = RandomForestClassifier(class_weight='balanced', max_depth=7, n_estimators=100, max_features='log2', random_state=42)
        else: model = RandomForestClassifier(class_weight='balanced', max_depth=9, n_estimators=50, max_features='log2', random_state=42)
        model.fit(X_train, y_train)
    elif m == 'MLP':
        print(f"\nMLP Performance: ")
        if dataset == 'Adult': model = MLPClassifier(alpha=1, random_state=42)
        elif dataset == 'Compas': model = MLPClassifier(alpha=1, random_state=42)
        elif dataset == 'Census-KDD': model = MLPClassifier(alpha=0.1, max_iter=10000, random_state=42)
        else: model = MLPClassifier(alpha=1, random_state=42)
        model.fit(X_train, y_train)
    else:
        print(f"\nERROR: Please input legal model name.")
        return None, None
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f">>>>>>>>>>Model parameter: {model.get_params()}<<<<<<<<<<")
    print(f"AUC: {auc} ")
    dfProbs = test_data[S+I+A+W+Y]
    dfProbs['pred'] = y_prob
    rod = ComputeDiscrimination(dfProbs, S, A, 2)
    print(f"ROD: {rod} ")
    return auc, rod


def get_current_code_info():
    """
    Get information about the current code execution context.
    
    Returns:
        Tuple of (filename, function_name, line_number)
    """
    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back
    filename = caller_frame.f_code.co_filename
    function_name = caller_frame.f_code.co_name
    line_number = caller_frame.f_lineno
    return filename, function_name, line_number


def ComputeDiscrimination(LRTestPreds, S_features, A_features, minGroupNum):
    """
    Compute discrimination score based on prediction ratios across protected groups.
    
    Args:
        LRTestPreds: DataFrame with predictions
        S_features: Protected attribute columns
        A_features: Actionable attribute columns
        minGroupNum: Minimum group size to consider
        
    Returns:
        Maximum discrimination score across groups
    """
    total = 0
    if len(S_features) <= 1:
        unique_combinations = LRTestPreds[S_features[0]].drop_duplicates().sort_values()
        index = pd.Index(unique_combinations, name=S_features[0])
    else:
        unique_combinations = LRTestPreds[S_features].drop_duplicates().sort_values(by=S_features)
        index = pd.MultiIndex.from_frame(unique_combinations)
    bias = pd.DataFrame(np.zeros((len(index), len(index))), index=index, columns=index)
    grouped = LRTestPreds.groupby(A_features)
    for name, group in grouped:
        size = len(group.index)
        mean = group.groupby(S_features)['pred'].mean()
        if size <= 0 or len(mean.values) < minGroupNum or mean.isnull().any(): continue
        epsilon = 0.05
        v1 = mean.values
        v1[v1 == 0] = epsilon
        v1[v1 == 1] = 1-epsilon
        v2 = 1 - v1
        v1 = v1.reshape(len(v1),1)
        v2 = v2.reshape(len(v2),1)
        ratio1 = pd.DataFrame(v1/v1.transpose(),index=mean.index,columns=mean.index)
        ratio2 = pd.DataFrame(v2/v2.transpose(),index=mean.index,columns=mean.index)
        ratio_df = ratio1 / ratio2
        ratio_df_expanded = ratio_df.reindex(index=index, columns=index, fill_value=1)
        if bias is None: bias = ratio_df_expanded * size
        else: bias += ratio_df_expanded * size
        total += size
    if (total == 0):
        print('ERROR: ', get_current_code_info(), '; [DETAILS]: something wrong when calculate discrimination.')
        return None
    bias /= total
    bias_corrected = np.abs(np.log2(np.asarray(bias)))     ### correct >1 and <1
    maxdisc=np.amax(bias_corrected)  ### 0 is the best; >0 and the smaller the better
    return maxdisc


def sum_all_results(methods, models, auc_list, rod_list, repairtime_list):
    """
    Summarize and print performance metrics across all methods and models.
    
    Args:
        methods: List of repair methods
        models: List of model types
        auc_list: List of AUC scores
        rod_list: List of ROD discrimination scores
        repairtime_list: List of repair times
    """
    ## list format: [method, mean for diff models]
    auc_mean = [[] for _ in range(len(methods))]
    rod_mean = [[] for _ in range(len(methods))]
    repairtime_mean = []

    print(f'\n\n\n======== Original Performance data ========\n')
    print(f'auc_list:\n{auc_list}\n')
    print(f'rod_list:\n{rod_list}\n')
    print(f'repair_time_list:\n{repairtime_list}\n')

    for method_i in range(len(methods)):
        for m_i in range(len(models)):
            auc_mean[method_i].append(sum(auc_list[method_i][m_i])/len(auc_list[method_i][m_i]))
            rod_mean[method_i].append(sum(rod_list[method_i][m_i])/len(rod_list[method_i][m_i]))
        repairtime_mean.append(sum(repairtime_list[method_i])/len(repairtime_list[method_i]))

    print("\n\n\n======== Overall Performance ========")
    for m_i in range(len(models)):
        ROD_mi = [rod_mean[i][m_i] for i in range(len(rod_mean))]
        min_ROD = min(ROD_mi)
        print(f'\n======== {models[m_i]} ========\n')
        for method_i in range(len(methods)):
            print(f'{methods[method_i]}: auc ({round(auc_mean[method_i][m_i], 6)}), '
                  f'rod ({round(rod_mean[method_i][m_i], 6)}), rod_ad ({round(rod_mean[method_i][m_i]-min_ROD, 4)})')

    print("\n\n\n======== Repair Time ========")
    for method_i in range(len(methods)):
            print(f'{methods[method_i]}: {repairtime_mean[method_i]}')
