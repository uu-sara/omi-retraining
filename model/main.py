# -*- coding: utf-8 -*-
"""Train and evaluate ECG classification models."""
__author__ = "Stefan Gustafsson"
__email__ = "stefan.gustafsson@medsci.uu.se"

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from warnings import warn

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataloader import BatchDataloader, OMIDataset
from metrics import EcgMetrics
from model import ECGModel, EnsembleECGModel
import utils

logger = logging.getLogger(__name__)


class LossConfig:
    def __init__(self, outcomes_cat, outcomes_bin, outcome_columns, w_bin_cat_ratio=1.0):
        self.loss_fn_cat = nn.CrossEntropyLoss()
        self.loss_fn_bin = nn.BCEWithLogitsLoss()
        self.indx_cat = [outcome_columns.get_loc(col) for col in outcomes_cat]
        self.indx_bin = [outcome_columns.get_loc(col) for col in outcomes_bin]
        if len(outcomes_cat) == 0:
            self.w_bin, self.w_cat = 1.0, 0.0
        elif len(outcomes_bin) == 0:
            self.w_bin, self.w_cat = 0.0, 1.0
        else:
            self.w_bin = w_bin_cat_ratio / (w_bin_cat_ratio + 1)
            self.w_cat = 1 / (w_bin_cat_ratio + 1)

    def compute_loss(self, pred_logits, outcomes):
        loss_cat = self.loss_fn_cat(pred_logits[:, self.indx_cat], outcomes[:, self.indx_cat]) if self.w_cat > 0 else None
        loss_bin = self.loss_fn_bin(pred_logits[:, self.indx_bin], outcomes[:, self.indx_bin]) if self.w_bin > 0 else None
        total = torch.tensor(0.0)
        if loss_cat is not None: total = total + self.w_cat * loss_cat
        if loss_bin is not None: total = total + self.w_bin * loss_bin
        return total, loss_cat, loss_bin


def prepare_batch(ecgs_cpu, outcomes_cpu, age_cpu, male_cpu, device):
    ecgs = ecgs_cpu.to(device)
    outcomes = outcomes_cpu.to(device)
    age_sex = torch.stack([male_cpu, age_cpu], dim=1).to(device)
    return (age_sex, ecgs), outcomes


def train_epoch(epoch_nr, dataloader, optimizer, model, loss_config, device):
    model.train()
    total_loss, total_loss_bin, total_loss_cat, n_samples = 0.0, 0.0, 0.0, 0
    pbar = tqdm(dataloader, desc=f"Training, epoch {epoch_nr:2d}", leave=False, position=0)
    for ecgs_cpu, outcomes_cpu, age_cpu, male_cpu in pbar:
        inputs, outcomes = prepare_batch(ecgs_cpu, outcomes_cpu, age_cpu, male_cpu, device)
        optimizer.zero_grad()
        pred_logits = model(inputs)
        loss, loss_cat, loss_bin = loss_config.compute_loss(pred_logits, outcomes)
        loss.backward()
        optimizer.step()
        bs = ecgs_cpu.size(0)
        total_loss += loss.detach().cpu().item() * bs
        if loss_cat is not None: total_loss_cat += loss_cat.detach().cpu().item() * bs
        if loss_bin is not None: total_loss_bin += loss_bin.detach().cpu().item() * bs
        n_samples += bs
    pbar.close()
    metrics = {'tot': total_loss / n_samples}
    if loss_config.w_bin > 0 and loss_config.w_cat > 0:
        metrics['bin'] = total_loss_bin / n_samples
        metrics['cat'] = total_loss_cat / n_samples
    return metrics


def evaluate_epoch(epoch_nr, dataloader, n_records, n_outcomes, model, loss_config, device):
    model.eval()
    total_loss, total_loss_bin, total_loss_cat, n_samples = 0.0, 0.0, 0.0, 0
    pred_prob = np.zeros((n_records, n_outcomes))
    batch_end = 0
    pbar = tqdm(dataloader, desc=f"Evaluation, epoch {epoch_nr:2d}", leave=False, position=0)
    with torch.no_grad():
        for ecgs_cpu, outcomes_cpu, age_cpu, male_cpu in pbar:
            inputs, outcomes = prepare_batch(ecgs_cpu, outcomes_cpu, age_cpu, male_cpu, device)
            batch_start = batch_end
            bs = ecgs_cpu.size(0)
            pred_logits = model(inputs)
            loss, loss_cat, loss_bin = loss_config.compute_loss(pred_logits, outcomes)
            batch_end = min(batch_start + bs, n_records)
            if loss_config.indx_cat:
                pred_prob[batch_start:batch_end, loss_config.indx_cat] = torch.softmax(pred_logits[:, loss_config.indx_cat], dim=1).cpu().numpy()
            if loss_config.indx_bin:
                pred_prob[batch_start:batch_end, loss_config.indx_bin] = torch.sigmoid(pred_logits[:, loss_config.indx_bin]).cpu().numpy()
            total_loss += loss.cpu().item() * bs
            if loss_cat is not None: total_loss_cat += loss_cat.cpu().item() * bs
            if loss_bin is not None: total_loss_bin += loss_bin.cpu().item() * bs
            n_samples += bs
    pbar.close()
    metrics = {'tot': total_loss / n_samples}
    if loss_config.w_bin > 0 and loss_config.w_cat > 0:
        metrics['bin'] = total_loss_bin / n_samples
        metrics['cat'] = total_loss_cat / n_samples
    return metrics, pred_prob


def predict_ensemble(dataloader, n_records, n_outcomes, model, device):
    model.eval()
    pred_prob = np.zeros((n_records, n_outcomes))
    pred_std = np.zeros((n_records, n_outcomes))
    batch_end = 0
    pbar = tqdm(dataloader, desc="Ensemble prediction", leave=False, position=0)
    with torch.no_grad():
        for ecgs_cpu, outcomes_cpu, age_cpu, male_cpu in pbar:
            ecgs = ecgs_cpu.to(device)
            age_sex = torch.stack([male_cpu, age_cpu], dim=1).to(device)
            batch_start = batch_end
            bs = ecgs_cpu.size(0)
            mean_probs, std_probs = model.forward_with_uncertainty((age_sex, ecgs))
            batch_end = min(batch_start + bs, n_records)
            pred_prob[batch_start:batch_end, :] = mean_probs.cpu().numpy()
            pred_std[batch_start:batch_end, :] = std_probs.cpu().numpy()
    pbar.close()
    return pred_prob, pred_std


def run_train_validate(args, loss_config, log, metrics_calc):
    tqdm.write('Setting up data...', end='')
    dset = OMIDataset(path_to_h5=args.hdf5, path_to_txt=args.txt, col_outcome=args.col_outcome, split_col=args.split_col, age_col=args.age_col, male_col=args.male_col, age_mean=args.age_mean, age_sd=args.age_sd)
    n_train, n_valid = sum(dset.train), sum(dset.valid)
    utils.save_dset(dset.outcomes[dset.train], os.path.join(args.folder, 'observed_data_train.csv'))
    utils.save_dset(dset.outcomes[dset.valid], os.path.join(args.folder, 'observed_data_valid.csv'))
    tqdm.write('done')
    tqdm.write(f'Found {n_train} training and {n_valid} validation records\n')
    for ensemble_nr in range(1, args.n_ensembles + 1):
        _train_single_ensemble(args, dset, n_train, n_valid, ensemble_nr, loss_config, log, metrics_calc)


def _train_single_ensemble(args, dset, n_train, n_valid, ensemble_nr, loss_config, log, metrics_calc):
    best_loss = float('inf')
    train_loader = BatchDataloader(dset, args.batch_size, mask=dset.train)
    valid_loader = BatchDataloader(dset, args.batch_size, mask=dset.valid)
    true_outcomes_valid = dset.outcomes[dset.valid]
    model = ECGModel(args)
    model.to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) if args.optim_algo.upper() == 'SGD' else optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, min_lr=args.lr_factor * args.min_lr, factor=args.lr_factor)
    if args.n_ensembles > 1:
        tqdm.write(f'\nRunning ensemble {ensemble_nr} out of {args.n_ensembles}\n')
        model_out, tensorboard_suffix, csv_out = os.path.join(args.folder, f'model_{ensemble_nr}.pth'), str(ensemble_nr), f'performance_metrics_{ensemble_nr}.csv'
    else:
        model_out, tensorboard_suffix, csv_out = os.path.join(args.folder, 'model.pth'), '', 'performance_metrics.csv'
    log.init_tensorboardlog(suffix=tensorboard_suffix)
    log.init_csvlog()
    for epoch_nr in range(1, args.epochs + 1):
        train_loss = train_epoch(epoch_nr, train_loader, optimizer, model, loss_config, args.device)
        valid_loss, valid_pred = evaluate_epoch(epoch_nr, valid_loader, n_valid, args.n_outcomes, model, loss_config, args.device)
        if args.save_progress: log.predictions_tofile(valid_pred, 'valid', epoch_nr, ensemble_nr)
        valid_metrics = metrics_calc.compute_metrics(valid_pred, true_outcomes_valid) if not args.no_metrics else {}
        is_best = valid_loss['tot'] < best_loss
        if is_best:
            torch.save({'epoch': epoch_nr, 'ensemble': ensemble_nr, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'valid_loss': valid_loss['tot'], 'config': vars(args)}, model_out)
            best_loss = valid_loss['tot']
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < args.min_lr:
            tqdm.write("Stopped: minimum learning rate reached")
            break
        tqdm.write(f"Epoch {epoch_nr:2d}: \033[91mTrain Loss\033[0m [tot={train_loss['tot']:.5f}] | \033[91mValid Loss\033[0m [tot={valid_loss['tot']:.5f}] | \033[91mLR\033[0m [{current_lr:.7f}] | \033[91mBest\033[0m [{is_best}]")
        log_data = {'ensemble': ensemble_nr, 'epoch': epoch_nr, 'lr': current_lr, **{f'train_{k}': v for k, v in train_loss.items()}, **{f'valid_{k}': v for k, v in valid_loss.items()}, **valid_metrics}
        log.tensorboardlog_tofile(log_data, epoch_nr)
        log.csvlog_tofile(log_data, csv_out)
        scheduler.step(valid_loss['tot'])
    log.close()


def run_test(args, loss_config, log, metrics_calc):
    tqdm.write('Setting up data...', end='')
    dset = OMIDataset(path_to_h5=args.hdf5, path_to_txt=args.txt, col_outcome=args.col_outcome, split_col=args.split_col, test=True, test_name=args.test_name, age_col=args.age_col, male_col=args.male_col, age_mean=args.age_mean, age_sd=args.age_sd)
    n_test = sum(dset.test)
    utils.save_dset(dset.outcomes[dset.test], os.path.join(args.folder, f'observed_data_{args.test_name}.csv'))
    true_outcomes_test = dset.outcomes[dset.test]
    tqdm.write('done')
    tqdm.write(f'Found {n_test} test records for "{args.test_name}"\n')
    test_loader = BatchDataloader(dset, args.batch_size, mask=dset.test)
    tqdm.write('Loading model...', end='')
    is_ensemble = args.n_ensembles > 1
    if is_ensemble:
        model = EnsembleECGModel(args, args.model_path)
    else:
        model = ECGModel(args)
        checkpoint = torch.load(os.path.join(args.model_path, 'model.pth'), map_location=args.device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        model.to(args.device)
        model.eval()
    tqdm.write('done\n')
    test_uncertainty = None
    if is_ensemble:
        test_pred, test_uncertainty = predict_ensemble(test_loader, n_test, args.n_outcomes, model, args.device)
        test_loss = {}
    else:
        test_loss, test_pred = evaluate_epoch(0, test_loader, n_test, args.n_outcomes, model, loss_config, args.device)
    test_metrics = metrics_calc.compute_metrics(test_pred, true_outcomes_test) if not args.no_metrics else {}
    pred_df = pd.DataFrame(data=test_pred, index=true_outcomes_test.index.tolist(), columns=[f'pr_{col}' for col in true_outcomes_test.columns])
    pred_df.index.name = true_outcomes_test.index.name
    log.predictions_tofile(pred_df, os.path.join(args.folder, f'predicted_data_{args.test_name}'))
    if test_uncertainty is not None:
        uncertainty_df = pd.DataFrame(data=test_uncertainty, index=true_outcomes_test.index.tolist(), columns=[f'std_{col}' for col in true_outcomes_test.columns])
        uncertainty_df.index.name = true_outcomes_test.index.name
        log.predictions_tofile(uncertainty_df, os.path.join(args.folder, f'uncertainty_{args.test_name}'))
        tqdm.write(f'Uncertainty estimates saved to uncertainty_{args.test_name}.csv')
    log.init_csvlog()
    log.csvlog_tofile({**test_loss, **test_metrics}, f'{args.test_name}_metrics.csv')
    tqdm.write(metrics_calc.get_summary(test_metrics))


def parse_arguments():
    sys_parser = argparse.ArgumentParser(add_help=False)
    sys_parser.add_argument('--model_path', type=str)
    sys_parser.add_argument('--hdf5', type=str, default='')
    sys_parser.add_argument('--hdf5_dset_ecg')
    sys_parser.add_argument('--hdf5_dset_recordid')
    sys_parser.add_argument('--txt', type=str, default='')
    sys_parser.add_argument('--split_col', type=str, default='split')
    sys_parser.add_argument('--age_col', type=str, default='age')
    sys_parser.add_argument('--male_col', type=str, default='male')
    sys_parser.add_argument('--test', action='store_true')
    sys_parser.add_argument('--test_name', type=str, default='test')
    sys_parser.add_argument('--out_dir', type=str)
    sys_parser.add_argument('--cpu', action='store_true')
    sys_parser.add_argument('--gpu_index', type=int, default=0)
    sys_parser.add_argument('--save_progress', action='store_true')
    sys_parser.add_argument('--outcomes_json', type=str, default=None)
    sys_parser.add_argument('--no_metrics', action='store_true')
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument('--seed', type=int, default=1234567)
    model_parser.add_argument('--outcomes_cat', type=str, nargs='+', default=[])
    model_parser.add_argument('--outcomes_bin', type=str, nargs='+', default=[])
    model_parser.add_argument('--epochs', type=int, default=70)
    model_parser.add_argument('--batch_size', type=int, default=32)
    model_parser.add_argument('--lr', type=float, default=0.001)
    model_parser.add_argument('--patience', type=int, default=7)
    model_parser.add_argument('--min_lr', type=float, default=1e-7)
    model_parser.add_argument('--lr_factor', type=float, default=0.1)
    model_parser.add_argument('--weight_decay', type=float, default=0.01)
    model_parser.add_argument('--seq_length', type=int, default=4096)
    model_parser.add_argument('--n_residual_block', type=int)
    model_parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320])
    model_parser.add_argument('--net_seq_length', type=int, nargs='+', default=[4096, 1024, 256, 64, 16])
    model_parser.add_argument('--dropout_rate', type=float, default=0.8)
    model_parser.add_argument('--kernel_size', type=int, default=17)
    model_parser.add_argument('--activation_function', type=str, default='ReLU')
    model_parser.add_argument('--optim_algo', type=str, default='ADAM')
    model_parser.add_argument('--w_bin_cat_ratio', type=float, default=1.0)
    model_parser.add_argument('--n_ensembles', type=int, default=1)
    model_parser.add_argument('--n_leads', type=int, default=8)
    model_parser.add_argument('--agesex_dim', type=int, default=64)
    model_parser.add_argument('--age_mean', type=float, default=None)
    model_parser.add_argument('--age_sd', type=float, default=None)
    arg_sys, arg_remaining = sys_parser.parse_known_args()
    if arg_sys.model_path:
        config_path = os.path.join(arg_sys.model_path, 'model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                model_parser.set_defaults(**json.load(f))
    arg_model, arg_unknown = model_parser.parse_known_args(arg_remaining)
    combined_parser = argparse.ArgumentParser(parents=[model_parser, sys_parser], description='Train or test ECG model')
    _, arg_unknown = combined_parser.parse_known_args(arg_unknown)
    if arg_unknown: warn(f"Unknown arguments: {arg_unknown}")
    arg_sys.folder = arg_sys.out_dir if arg_sys.test else utils.set_output_folder(arg_sys.out_dir)
    arg_model.device = 'cpu' if arg_sys.cpu or not torch.cuda.is_available() else f'cuda:{arg_sys.gpu_index}'
    if not arg_sys.test and arg_sys.outcomes_json:
        with open(arg_sys.outcomes_json, 'r') as f:
            outcomes_data = json.load(f)
        arg_model.outcomes_cat = outcomes_data.get('outcomes_cat', arg_model.outcomes_cat)
        arg_model.outcomes_bin = outcomes_data.get('outcomes_bin', arg_model.outcomes_bin)
    if arg_sys.test and arg_sys.model_path:
        with open(os.path.join(arg_sys.model_path, 'model_config.json'), 'r') as f:
            saved = json.load(f)
        arg_model.outcomes_cat = saved.get('outcomes_cat', [])
        arg_model.outcomes_bin = saved.get('outcomes_bin', [])
    arg_model.col_outcome = arg_model.outcomes_cat + arg_model.outcomes_bin
    arg_model.n_outcomes = len(arg_model.col_outcome)
    return arg_sys, arg_model, arg_unknown


def main():
    tqdm.write('Initializing...', end='')
    arg_sys, arg_model, _ = parse_arguments()
    if not arg_sys.test:
        utils.save_config(arg_model, os.path.join(arg_sys.folder, 'model_config.json'))
        utils.save_config(arg_sys, os.path.join(arg_sys.folder, 'sys.json'))
    else:
        utils.save_config(arg_sys, os.path.join(arg_sys.folder, f'sys_{arg_sys.test_name}.json'))
    args = argparse.Namespace(**vars(arg_sys), **vars(arg_model))
    if args.n_residual_block is not None and args.n_residual_block in [2, 4, 8, 16]:
        args.net_filter_size, args.net_seq_length = utils.net_param_map(args.n_residual_block)
    utils.seed_everything(args.seed, deterministic=True, warn_only=True)
    tqdm.write('done')
    tqdm.write(f'Using seed: {args.seed}')
    tqdm.write(f'Using device: {args.device}')
    if not arg_sys.test:
        tqdm.write(f'Model config: {os.path.join(args.folder, "model_config.json")}')
        tqdm.write(f'System config: {os.path.join(args.folder, "sys.json")}\n')
    tqdm.write('Setting up logger and metrics...', end='')
    metrics_calc = EcgMetrics(args.col_outcome)
    log = utils.Logger(args.folder)
    tqdm.write('done')
    tqdm.write(f'Output directory: {args.folder}\n')
    temp_dset = OMIDataset(path_to_h5=args.hdf5, path_to_txt=args.txt, col_outcome=args.col_outcome, split_col=args.split_col, test=args.test, test_name=args.test_name if args.test else 'test', age_col=args.age_col, male_col=args.male_col, age_mean=args.age_mean, age_sd=args.age_sd)
    loss_config = LossConfig(outcomes_cat=args.outcomes_cat, outcomes_bin=args.outcomes_bin, outcome_columns=temp_dset.outcomes.columns, w_bin_cat_ratio=args.w_bin_cat_ratio)
    args.w_bin, args.w_cat = loss_config.w_bin, loss_config.w_cat
    del temp_dset
    if args.test:
        tqdm.write('Testing model...\n')
        run_test(args, loss_config, log, metrics_calc)
    else:
        tqdm.write('Training and validating model...\n')
        run_train_validate(args, loss_config, log, metrics_calc)
    tqdm.write('\nAll done')


if __name__ == "__main__":
    main()
