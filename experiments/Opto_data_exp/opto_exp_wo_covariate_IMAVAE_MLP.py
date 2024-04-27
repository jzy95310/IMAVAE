import sys, os
sys.path.insert(0, '../../')
import argparse
import numpy as np
import torch
import pickle as pkl
from models.imavae import IMAVAE

np.random.seed(2020)
torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
torch.cuda.manual_seed_all(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(args):
    with open('../../data/mediation_opto_fosit_shifted_wo_covariate.pkl', 'rb') as fp:
        data = pkl.load(fp)
    X, Z, T, Y = data['M'], data['Z'], data['T'][:,np.newaxis], data['Y']
    print("Dimension of X: ({}, {})".format(X.shape[0], X.shape[1]))
    print("Dimension of Z: ({}, {})".format(Z.shape[0], Z.shape[1]))

    imavae = IMAVAE(
        n_components=args.n_components, 
        n_sup_networks=args.n_sup_networks, 
        n_hidden_layers=args.n_hidden_layers, 
        hidden_dim=args.hidden_dim, 
        n_sup_hidden_layers=args.n_sup_hidden_layers,
        n_sup_hidden_dim=args.n_sup_hidden_dim, 
        optim_name=args.optim_name, 
        weight_decay=args.weight_decay, 
        recon_weight=args.recon_weight, 
        elbo_weight=args.elbo_weight, 
        sup_weight=args.sup_weight
    )
    _ = imavae.fit(
        X, T, Y,
        lr=args.lr, 
        n_epochs=args.n_epochs, 
        batch_size=args.batch_size,
        pretrain=False, 
        verbose=args.verbose
    )
    acme_c_mean, acme_c_std = imavae.acme_score(T, treatment=False)
    acme_t_mean, acme_t_std = imavae.acme_score(T, treatment=True)
    ade_c_mean, ade_c_std = imavae.ade_score(T, treatment=False)
    ade_t_mean, ade_t_std = imavae.ade_score(T, treatment=True)
    ate_mean, ate_std = imavae.ate_score(T)
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
    print("-------------------------------------")
    print("Error on ACME (control) = {:.4f} +/- {:.4f}".format(np.abs(acme_c_mean-data['acme_c_true']), acme_c_std))
    print("Error on ACME (treatment) = {:.4f} +/- {:.4f}".format(np.abs(acme_t_mean-data['acme_t_true']), acme_t_std))
    print("Error on ADE (control) = {:.4f} +/- {:.4f}".format(np.abs(ade_c_mean-data['ade_c_true']), ade_c_std))
    print("Error on ADE (treatment) = {:.4f} +/- {:.4f}".format(np.abs(ade_t_mean-data['ade_t_true']), ade_t_std))
    print("Error on ATE = {:.4f} +/- {:.4f}".format(np.abs(ate_mean-data['ate_true']), ate_std))
    
    res = {
        'acme_c': {'mean': acme_c_mean, 'std': acme_c_std, 'true': data['acme_c_true']}, 
        'acme_t': {'mean': acme_t_mean, 'std': acme_t_std, 'true': data['acme_t_true']}, 
        'ade_c': {'mean': ade_c_mean, 'std': ade_c_std, 'true': data['ade_c_true']}, 
        'ade_t': {'mean': ade_t_mean, 'std': ade_t_std, 'true': data['ade_t_true']}, 
        'ate': {'mean': ate_mean, 'std': ate_std, 'true': data['ate_true']}
    }
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    with open('./results/opto_exp_wo_covariate_IMAVAE_MLP.pkl', 'wb') as fp:
        pkl.dump(res, fp)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train IMAVAE on Opto data without covariates.')
    arg_parser.add_argument('--n_components', type=int, default=30)
    arg_parser.add_argument('--n_sup_networks', type=int, default=30)
    arg_parser.add_argument('--n_hidden_layers', type=int, default=2)
    arg_parser.add_argument('--hidden_dim', type=int, default=128)
    arg_parser.add_argument('--n_sup_hidden_layers', type=int, default=1)
    arg_parser.add_argument('--n_sup_hidden_dim', type=int, default=10)
    arg_parser.add_argument('--optim_name', type=str, default="Adam")
    arg_parser.add_argument('--recon_weight', type=float, default=0.1)
    arg_parser.add_argument('--elbo_weight', type=float, default=0.1)
    arg_parser.add_argument('--sup_weight', type=float, default=1.)
    arg_parser.add_argument('--lr', type=float, default=5e-6)
    arg_parser.add_argument('--weight_decay', type=float, default=0.0)
    arg_parser.add_argument('--n_epochs', type=int, default=100)
    arg_parser.add_argument('--batch_size', type=int, default=128)
    arg_parser.add_argument('--verbose', type=int, default=1)
    args = arg_parser.parse_known_args()[0]
    main(args)