""" I can't show my actual scripts from the Kaggle competition,
but here is my process for fitting a model. The data is split like 
the competition-- where each day is treated as one unit 
(i.e. all events occuring on 1-01-04 will be either in the testing
or training group). 

The scoring metric I use is a multivariate log-loss. The function 
itself is based on Kaggle's own, but model results need to be 
manipulated for it to work """

import pandas as pd
import pickle
import numpy as np

#################################
Functions 
#################################

def split_train_test(dataframe, size=1500):
    """ Returns X_train, X_test, y_train, y_test """
     
    tr_gp = dataframe.groupby(['date']).count()
    rows = np.random.choice(tr_gp.index.values, replace=False, size=size)
    tr = dataframe[dataframe.date.isin(rows)]
    ts = dataframe[-dataframe.date.isin(rows)]
    X_train = tr.drop(['category', 'date'], axis =1)
    X_test = ts.drop(['category', 'date'], axis=1)
    y_train = np.array(tr['category'])
    y_test = np.array(ts['category'])
    return X_train, X_test, y_train, y_test

def llfun(act, pred):
    """ Log loss function. Takes two n x 39 matrices. """
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = (act*sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred))).sum()
    ll = ll * -1.0/len(act)
    return ll

def get_preds_df(model):
    
    fit = model().fit(X_train, y_train)
    pred = fit.predict_proba(X_test)
    these_cols = list(fit.classes_)
    pred_df = pd.DataFrame(pred, columns = these_cols)
    missing_no = len([obj for obj in all_cols if obj not in these_cols])
    if missing_no > 0:
        extra = pd.DataFrame(np.zeros((len(pred), missing_no)), columns = [obj for obj in all_cols if obj not in these_cols])
        y_pred = pd.concat([pred_df, extra], axis=1)
    else:
        y_pred = pred_df
    return y_pred.as_matrix()
        
    
def get_actual_df(y_test):
    y_ts = pd.get_dummies(y_test)
    y_ts.reset_index(drop=True, inplace=True)
    these_cols = list(y_ts.columns)
    missing_no = len([obj for obj in all_cols if obj not in these_cols])
    if missing_no > 0:
        extra = pd.DataFrame(np.zeros((len(y_test), missing_no)), columns = [obj for obj in all_cols if obj not in these_cols])
        actual = pd.concat([y_ts, extra], axis=1)
    else:
        actual = y_ts
    return actual.as_matrix()


all_cols = ['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY',
       'DISORDERLY CONDUCT', 'DRIVING UNDER THE INFLUENCE',
       'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION',
       'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING',
       'KIDNAPPING', 'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING',
       'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES',
       'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE',
       'ROBBERY', 'RUNAWAY', 'SECONDARY CODES', 'SEX OFFENSES FORCIBLE',
       'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE',
       'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT',
       'WARRANTS', 'WEAPON LAWS']
