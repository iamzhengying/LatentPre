import numpy as np
import pandas as pd

def age_cut(x):
    if x <= 25:
        return 20
    elif x <= 35:
        return 30
    elif x <= 45:
        return 40
    elif x <= 60:
        return 50
    else:
        return 70

def hours_cut(x):
    if x <= 20:
        return 20
    elif x <= 35:
        return 30
    elif x <= 40:
        return 40
    elif x <= 45:
        return 45
    elif x <= 60:
        return 60
    else:
        return 70

def capgain_cut(x):
    if x == 0:
        return -1
    elif x < 50000:
        return x // 1000
    else:
        return 50

def caploss_cut(x):
    if x == 0:
        return -1
    if x < 2400:
        return x // 100
    else:
        return 24

def clean_adult(df):
    dataRaw = df.copy()
    dataRaw['age'] = dataRaw['age'].apply(lambda x: age_cut(x))
    dataRaw['hours-per-week'] = dataRaw['hours-per-week'].apply(lambda x: hours_cut(x))
    dataRaw['capital-gain'] = dataRaw['capital-gain'].apply(lambda x: capgain_cut(x))
    dataRaw['capital-loss'] = dataRaw['capital-loss'].apply(lambda x: caploss_cut(x))
    return dataRaw

#######################   COMPAS   #########################


def quantizePrior(x):
    if x<=0: return 0
    elif x<=3: return 1
    elif x<=6: return 2
    elif x<=10: return 3
    elif x<=15: return 4
    elif x<=20: return 5
    else: return 6
    
def quantizeLOS(x):
    if x<=3: return 0
    elif x<=7: return 1
    elif x<=30: return 2
    elif x<=60: return 3
    elif x<=90: return 4
    elif x<=180: return 5
    else: return 6

def clean_compas(df):
    df = df[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count', 
                    'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out',
                    'juv_fel_count', 'juv_misd_count', 'juv_other_count']]
    ix = df['days_b_screening_arrest'] <= 30
    ix = (df['days_b_screening_arrest'] >= -30) & ix
    ix = (df['is_recid'] != -1) & ix
    ix = (df['c_charge_degree'] != "O") & ix
    ix = (df['score_text'] != 'N/A') & ix
    df = df.loc[ix,:]
    df['length_of_stay'] = (pd.to_datetime(df['c_jail_out'])-pd.to_datetime(df['c_jail_in'])).apply(lambda x: x.days)
    # Drop Asian, Native American due to lack of samples
    dfcut = df.loc[~df['race'].isin(['Native American','Asian']),:]
    dfcutQ = dfcut[['sex','race','age_cat','c_charge_degree','score_text','priors_count','is_recid',
                'two_year_recid','length_of_stay', 'days_b_screening_arrest', 'decile_score', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']].copy()
    dfcutQ['priors_count'] = dfcutQ['priors_count'].apply(lambda x: quantizePrior(x))
    dfcutQ['length_of_stay'] = dfcutQ['length_of_stay'].apply(lambda x: quantizeLOS(x))
    return dfcutQ


#######################   CENSUS   #########################


def clean_census(dataRaw):
    ### re-name
    dataRaw = dataRaw.rename(columns={
            'AAGE': 'age',
            'ACLSWKR': 'class_of_worker',
            'ADTINK': 'industry_code',
            'ADTOCC': 'occupation_code',
            'AHGA': 'education',
            'AHRSPAY': 'wage_per_hour',
            'AHSCOL': 'enrolled_in_edu_inst_last_wk',
            'AMARITL': 'marital_status',
            'AMJIND': 'major_industry_code',
            'AMJOCC': 'major_occupation_code',
            'ARACE': 'race',
            'AREORGN': 'hispanic_origin',
            'ASEX': 'sex',
            'AUNMEM': 'member_of_a_labor_union',
            'AUNTYPE': 'reason_for_unemployment',
            'AWKSTAT': 'full_or_part_time_employment_stat',
            'CAPGAIN': 'capital_gains',
            'GAPLOSS': 'capital_loss',
            'DIVVAL': 'dividends_from_stocks',
            'FILESTAT': 'tax_filer_status',
            'GRINREG': 'region_of_previous_residence',
            'HHDFMX': 'detailed_household_and_family_stat',
            'HHDREL': 'detailed_household_summary_in_household',
            'MARSUPWRT': 'instance_weight',
            'MIGMTR1': 'migration_code_change_in_msa',
            'MIGMTR3': 'migration_code_change_in_reg',
            'MIGMTR4': 'migration_code_move_within_reg',
            'MIGSAME': 'live_in_this_house_1_year_ago',
            'MIGSUN': 'migration_prev_res_in_sunbelt',
            'NOEMP': 'num_persons_worked_for_employer',
            'PARENT': 'family_members_under_18',
            'PEARNVAL': 'total_person_earnings',
            'PEFNTVTY': 'country_of_birth_father',
            'PEMNTVTY': 'country_of_birth_mother',
            'PENATVTY': 'country_of_birth_self',
            'PRCITSHP': 'citizenship',
            'SEOTR': 'own_business_or_self_employed',
            'TAXINC': 'taxable_income_amount',
            'VETQVA': 'fill_inc_questionnaire_for_veteran_admin',
            'VETYN': 'veterans_benefits',
            'WKSWORK': 'weeks_worked_in_year',
            'GRINST': 'state_of_previous_residence'})
    ### clean
    dataDrop = dataRaw.drop(columns=[
            'industry_code', 'major_occupation_code', 'enrolled_in_edu_inst_last_wk',
            'hispanic_origin', 'state_of_previous_residence',
            'detailed_household_and_family_stat',
            'migration_code_change_in_msa', 'migration_code_change_in_reg',
            'migration_code_move_within_reg', 'migration_prev_res_in_sunbelt',
            'country_of_birth_father', 'country_of_birth_mother',
            'tax_filer_status', 'live_in_this_house_1_year_ago'])
    dataClean = dataDrop.dropna()
    ### bin
    dataBin = dataClean.copy()
    dataBin['age'] = pd.cut(dataBin['age'], bins=10, labels=list(range(10)))
    dataBin['wage_per_hour'] = pd.cut(dataBin['wage_per_hour'], bins=15, labels=list(range(15)))
    dataBin['capital_gains'] = pd.cut(dataBin['capital_gains'], bins=10, labels=list(range(10)))
    dataBin['capital_loss'] = pd.cut(dataBin['capital_loss'], bins=10, labels=list(range(10)))
    dataBin['dividends_from_stocks'] = pd.cut(dataBin['dividends_from_stocks'], bins=20, labels=list(range(20)))
    dataBin['instance_weight'] = pd.cut(dataBin['instance_weight'], bins=30, labels=list(range(30)))
    dataBin['weeks_worked_in_year'] = pd.cut(dataBin['weeks_worked_in_year'], bins=10, labels=list(range(10)))
    return dataBin
