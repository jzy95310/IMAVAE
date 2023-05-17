import os, sys
sys.path.insert(0, '../../')
import numpy as np
import pickle as pkl
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import random
from scipy.stats import zscore
from functools import partial
from benchmarks.deep_LSEM import auxiliaryfunctions

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

random.seed(2020)
np.random.seed(2020)
tf.random.set_seed(2020)

def create_empty_df(num_runs,num_iters):
    # Create an empty dataFrame
    a = np.empty((num_runs,num_iters))
    a[:] = np.nan
    dataFrame = None
    parameters = ['alpha0','beta0','alpha','beta','gamma']
    for params in parameters:
        iter = ['iter_'+str(i) for i in range(num_iters)]
        pdindex = pd.MultiIndex.from_product([[params], iter], names=['parameters', 'runs']) 
        frame = pd.DataFrame(a, columns = pdindex,index = range(0,num_runs))
        dataFrame = pd.concat([dataFrame,frame],axis=1)
    return dataFrame

def create_shallow_dense_network(hidden_dim):
    # Create a one-layer dense network
    model = Sequential([
        Dense(hidden_dim, activation=partial(tf.nn.leaky_relu, alpha=0.01), name='dense1'), 
        Dense(1, activation=None, name='output')
    ])
    model.compile(loss=keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam(),metrics=['mse'])
    return model

def simulate_mediation(df,M,params_df,model,batch_size,epochs,tune,n_runs,iterations,patience,verbose):
    """
    """
    early_stopping=True
    # Initialize z with some phi
    print("Using initial random model parameters")

    z = model.predict(M) 
    try:
        z = np.concatenate(z)
    except:
        pass
    
    for i in range(0,iterations):
        print('Starting iteration... %s'%(i+1))
        #Check for sign change
        if np.corrcoef(z,df.Y)[0,1] < 0 :
            z = z*(-1)
        # Check for scaling
        z = zscore(z)

        lm = smf.ols(formula='z ~ X', data=df).fit()
        alpha0 = lm.params.loc['Intercept']
        alph = lm.params.loc['X'] 

        lm = smf.ols(formula='Y ~ z + X', data=df).fit()
        beta0 = lm.params.Intercept 
        bet = lm.params.z
        gam = lm.params.X 
        resid_std = np.std(lm.resid)
        e = df.Y - beta0 - (df.X*gam)
        h = alpha0 + df.X*alph
        d = (((bet*e)+h)/((bet**2)+1))
        
        if tune:
            adam = Adam()
            model.compile(loss='mean_absolute_error',optimizer=adam)
            if early_stopping:
                es = EarlyStopping(monitor='val_loss',mode='min',verbose=verbose,patience=patience)
                _ = model.fit(M,d,batch_size=batch_size,epochs=epochs,verbose=verbose,callbacks=[es],
                                  shuffle=True,validation_split = 0.3)
            else:
                _ = model.fit(M,d,batch_size=batch_size,epochs=epochs,verbose=verbose,
                              shuffle=True,validation_split = 0.3)
            z = model.predict(M)
            
   
        params_df.loc[n_runs]['alpha0','iter_'+str(i)]=alpha0
        params_df.loc[n_runs]['beta0','iter_'+str(i)]=beta0
        params_df.loc[n_runs]['beta','iter_'+str(i)]=bet
        params_df.loc[n_runs]['gamma','iter_'+str(i)]=gam
        params_df.loc[n_runs]['alpha','iter_'+str(i)]=alph

        try:
            z = np.concatenate(z)
        except:
            pass
       
    return params_df,z

def main():
    early_stopping = True
    epochs = 100
    batch_size = 128
    patience = 10
    verbose = 0
    result_path = './results/csv_files'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    num_iters = 20
    num_runs = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    output_df = create_empty_df(num_runs,num_iters)
    # Load the neural data
    with open('../../data/mediation_tst_shifted.pkl', 'rb') as fp:
        data = pkl.load(fp)
    df = pd.DataFrame({'X': data['T'].astype(int), 'Y': data['Y'].squeeze()})
    mediator = data['M']
    
    for runs in range(num_runs):
        K.clear_session()
        print("Running simulation with run %s" % (runs+1))

        # create a model
        model = create_shallow_dense_network(hidden_dim=128)

        # Fit the deep mediation LSEM framework
        params_df,z_final = simulate_mediation(df,mediator,output_df,model,batch_size,epochs,
                                      True,runs,num_iters,patience,verbose)
    params_df.to_csv(os.path.join(result_path,'tst_exp_with_covariate_shallow_LSEM.csv'))
    
    res = pd.read_csv('./results/csv_files/tst_exp_with_covariate_shallow_LSEM.csv')
    alpha_mean = np.mean(np.array(res['alpha.{}'.format(num_iters-1)][1:]).astype(np.float))
    alpha_std = np.std(np.array(res['alpha.{}'.format(num_iters-1)][1:]).astype(np.float))
    beta_mean = np.mean(np.array(res['beta.{}'.format(num_iters-1)][1:]).astype(np.float))
    beta_std = np.std(np.array(res['beta.{}'.format(num_iters-1)][1:]).astype(np.float))
    gamma_mean = np.mean(np.array(res['gamma.{}'.format(num_iters-1)][1:]).astype(np.float))
    gamma_std = np.std(np.array(res['gamma.{}'.format(num_iters-1)][1:]).astype(np.float))
    acme_c_mean = alpha_mean * beta_mean
    acme_c_std = ((alpha_std**2+alpha_mean**2)*(beta_std**2+beta_mean**2) - (alpha_mean**2)*(beta_mean**2))**(0.5)
    acme_t_mean = alpha_mean * beta_mean
    acme_t_std = ((alpha_std**2+alpha_mean**2)*(beta_std**2+beta_mean**2) - (alpha_mean**2)*(beta_mean**2))**(0.5)
    ade_c_mean, ade_c_std = gamma_mean, gamma_std
    ade_t_mean, ade_t_std = gamma_mean, gamma_std
    ate_mean = alpha_mean * beta_mean + gamma_mean
    ate_std = acme_c_std + gamma_std
    
    print("ACME (control) = {:.4f} +/- {:.4f}".format(acme_c_mean, acme_c_std))
    print("ACME (treatment) = {:.4f} +/- {:.4f}".format(acme_t_mean, acme_t_std))
    print("ADE (control) = {:.4f} +/- {:.4f}".format(ade_c_mean, ade_c_std))
    print("ADE (treatment) = {:.4f} +/- {:.4f}".format(ade_t_mean, ade_t_std))
    print("ATE = {:.4f} +/- {:.4f}".format(ate_mean, ate_std))
    print("-------------------------------------")
    print("True ACME (control) = {:.4f}".format(data['acme_c_true']))
    print("True ACME (treatment) = {:.4f}".format(data['acme_t_true']))
    print("True ADE (control) = {:.4f}".format(data['ade_c_true']))
    print("True ADE (treatment) = {:.4f}".format(data['ade_t_true']))
    print("True ATE = {:.4f}".format(data['ate_true']))
    
    res = {
        'acme_c': {'mean': acme_c_mean, 'std': acme_c_std, 'true': data['acme_c_true']}, 
        'acme_t': {'mean': acme_t_mean, 'std': acme_t_std, 'true': data['acme_t_true']}, 
        'ade_c': {'mean': ade_c_mean, 'std': ade_c_std, 'true': data['ade_c_true']}, 
        'ade_t': {'mean': ade_t_mean, 'std': ade_t_std, 'true': data['ade_t_true']}, 
        'ate': {'mean': ate_mean, 'std': ate_std, 'true': data['ate_true']}
    }
    with open('./results/tst_exp_with_covariate_shallow_LSEM.pkl', 'wb') as fp:
        pkl.dump(res, fp)

if __name__ == '__main__':
    main()