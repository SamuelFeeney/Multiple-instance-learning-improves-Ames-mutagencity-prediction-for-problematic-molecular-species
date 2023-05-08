import pandas as pd
import numpy as np
import os
import pickle
import sys
from IPython.display import clear_output
from tpot import TPOTClassifier
from sklearn.feature_selection import VarianceThreshold
try:
    import misvm 
except:
    sys.exit("please use command to install MIL modelling package \n pip install -e git+https://github.com/garydoranjr/misvm.git#egg=misvm")

def updater(mdl,splt,enc,fld):
    clear_output(wait=True)
    if fld !="": 
        print('splitting method:',splt,'    encoding:',enc,'   model:',mdl)
        print('fold:',fld%10,'    iteration:',fld//10) 
        print('')
        print('percentage: |'+fld//2*("*")+["0"+str(fld) if len(str(fld))<2 else str(fld)][0]+(100-fld)//2*("*")+'|')
    else:
        print('splitting method:',splt,'    encoding:',enc,'   model:',mdl) 

def pos_or_neg(x):  ## Simple function used to translate predictions between MIL and ML into a single form
    if x>0:
        return 1
    else:
        return 0

def build_test_mil_model(training_data,testing_data,MIL,encoding,suffix,save_model,model_name = "",save_name="MIL_total_results.pk1"):    ## Build and test a MIL model
    already_complete = check_if_tested(suffix=suffix,encoding=encoding,model_name=model_name,path=save_name)
    if not already_complete:
        ##      Building model, note encoding already performed    
        bags = training_data[encoding+"_MIL"].to_list()   
        labels = training_data["Ames"].apply(lambda x: x if x==1 else -1).to_list()
        model = MIL                                                                   
        model.fit(bags,labels)    
        ##      Testing model     
        bags = testing_data[encoding+"_MIL"].to_list()
        labels = testing_data["Ames"].apply(lambda x: x if x==1 else -1).to_list()   
        predictions = model.predict(bags)                                      
        predicted_labels = list(map(pos_or_neg,predictions))                            
        df = pd.DataFrame({
            'predicted' : predictions,
            'predicted labal' : predicted_labels,
            'true label' : testing_data["Ames"].to_list()
        })  
        if type(save_name) == str:
            save_results(df = df, suffix = suffix, model = model_name, encoding = encoding, save_path = save_name)
        elif save_name:
            save_results(df = df, suffix = suffix, model = model_name, encoding = encoding)
    else:
        print("Already tested   fold:",suffix["fold"],"   iteration:",suffix["iteration"],"   model:",model_name,"   encoding:",encoding)
        clear_output(wait=True)

def check_if_tested(suffix,model_name,encoding,path):    ## Checking if this build/test has already been done. Saves on time if a run crashes
    if not os.path.isfile(path): 
        already_complete = False
    else:
        results = pd.read_pickle(path)
        already_complete = ((results["fold"].isin([suffix["fold"]])) & (results["iteration"].isin([suffix["iteration"]])) & (results["model"].isin([model_name])) & (results["encoding"].isin([encoding]))).any()
    return already_complete

def format_results(df,suffix,model,encoding):  ## adds informative columns to the df used for saving results 
    df["fold"]  =   suffix["fold"]
    df["iteration"] =   suffix["iteration"]
    df['index'] = df.index
    df["model"] =   model
    df["encoding"] =   encoding
    return df

def save_results(df,suffix,model,encoding,save_path):     ## saves results to a single pickle, adding to it or generating it
    if not os.path.isfile(save_path):
        df_formatted = format_results(df=df,suffix=suffix,model=model,encoding=encoding)
        df_formatted.to_pickle(save_path)
    else:
        total_results = pd.read_pickle(save_path)
        df_formatted = format_results(df=df,suffix=suffix,model=model,encoding=encoding)
        total_results = pd.concat([total_results,df_formatted], ignore_index=True)
        total_results.to_pickle(save_path)
        
def save_models(model,path):                    ## Saves model to a path
    pickle.dump(model, open(path, 'wb'))

def develop_models(training_data,testing_data,suffix={"fold":"","iteration":""},encoding="MACCS",save_model=False,save_name="MIL_total_results.pk1",kernels=["linear", 'quadratic', 'polynomial']):     ## single function to complete whole pipeline for a set of data to all expected models
    ##      Step 0:     Checking that the encoding method described is expected
    if not encoding in ["MACCS","RDFP","Morgan"]:
        print('Please use expected encoding: ["MACCS","RDFP","Morgan"]')
        return
    for kernel_type in kernels:
        ##      Step 1:     Build and test models
        tested_mils =  [   
            ["MISVM_"+kernel_type, misvm.MISVM(kernel=kernel_type, C=1.0,verbose=False)],
            ['SIL_'+kernel_type, misvm.SIL(kernel=kernel_type,verbose=False)],
            ['NSK_'+kernel_type, misvm.NSK(kernel=kernel_type,verbose=False)],
            ["sbMIL-"+kernel_type, misvm.sbMIL(kernel=kernel_type,verbose=False,eta=0.5)],
            ['sMIL_'+kernel_type, misvm.sMIL(kernel=kernel_type,verbose=False)]
                        ]
        for mil in tested_mils:
            build_test_mil_model(training_data=training_data,testing_data=testing_data,suffix=suffix,MIL=mil[1],encoding=encoding,model_name=mil[0],save_model=save_model,save_name=save_name)

def build_test_ml_model(training_data,testing_data,encoding,suffix,save_name,save_model,tpot = False,splitting_name=''):                      ## Build and test a machine learning model
    already_complete = check_if_tested(suffix=suffix,encoding=encoding,model_name="TPOT",path=save_name)
    if not already_complete:
        ##      Building model, note encoding already performed
        instances = np.array(training_data[encoding].to_list())
        labels = training_data["Ames"].to_list() 
        if not tpot:   
            tpot_optimisation = TPOTClassifier(generations=10, population_size=500, cv=5, verbosity=0, n_jobs=8)
        else:
            tpot_optimisation = tpot
        tpot_optimisation.fit(instances,labels)    
        ##      Testing model
        model = tpot_optimisation.fitted_pipeline_  ## This takes the best fitted pipeline developed
        tpot_optimisation.export('tpot models/tpot_'+encoding+splitting_name+'_exported_pipeline.py')
        instances = np.array(testing_data[encoding].to_list())
        true_labels = testing_data["Ames"].to_list()       
        predictions = model.predict(instances) 
        predicted_probabilities = model.predict_proba(instances)                             
        predicted_labels = list(map(pos_or_neg,predictions))                            
        df = pd.DataFrame({
            'predicted' : [i[1] for i in predicted_probabilities],
            'predicted labal' : predicted_labels,
            'true label' : true_labels
        })   
        if type(save_name) == str:
            save_results(df = df, suffix = suffix, model = "TPOT", encoding = encoding, save_path = save_name)
        elif save_name:
            save_results(df = df, suffix = suffix, model = "TPOT", encoding = encoding)
    else:
        print("Already tested   fold:",suffix["fold"],"   iteration:",suffix["iteration"],"   model:","TPOT","   encoding:",encoding)

def check_rank(df):
    count = 0
    for x in df['PaDEL_MIL']:
        matrix=np.array(x)
        if not np.linalg.matrix_rank(matrix) == len(x):
            count+=1
    if count != 0:
        print("This will fail...",count)

def clean_data(df, remove_duplicates=False):
    def remove_duplicate_lists(x):
        duplicates = []; non_duplicates = []
        if type(x) == list:
            for i, test_list1 in enumerate(x):
                for j, test_list2 in enumerate(x):
                    if i < j:
                        if test_list1 == test_list2:
                            # print(i,'and',j,"are identical")
                            duplicates += [j]
            for i, list1 in enumerate(x):
                if i not in duplicates:
                    non_duplicates += [list1]
            return non_duplicates
        else:
            return np.nan
    working = df.copy()
    ## clean empty or NAN rows
    working = working.dropna(axis=0,how='any')
    ## Clean MIL duplicates
    if remove_duplicates:
        for MIL in ['MACCS_MIL','PaDEL_MIL']: 
            working[MIL] = working[MIL].apply(lambda x: remove_duplicate_lists(x))
    return working

def remove_zero_variance(inp,encoding='Morgan'):
    df = inp.copy()
    all_data = [lst for lists in df[encoding+'_MIL'].to_list() for lst in lists]
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(all_data)
    df[encoding+'_MIL'] = df[encoding+'_MIL'].apply(lambda x: constant_filter.transform(x))
    return df