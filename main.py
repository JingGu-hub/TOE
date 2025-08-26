import argparse
import os
import torch
import random
import numpy as np
from exp.exp_main import Exp_Main
import warnings

warnings.filterwarnings("ignore")


def fix_random_seed(seed):
    """Fixes the random seed"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


if __name__ == '__main__':
    fix_seed = 1
    fix_random_seed(fix_seed)

    # ================= Key Hyperparameters for TOE-Base =================
    # --- Data & Task Settings ---
    dataset = 'ETT-small'
    sub_dataset = 'ETTm2'
    seq_len = 96
    pred_len = 96

    # --- Model Architecture ---
    d_model = 128
    d_ff = 256
    dropout = 0.2
    d_core = 128
    d_pick = 128

    # --- Training Strategy ---
    batch_size = 128
    lr_main = 0.001
    lr_retrain = 0.01
    warmup_ratio = 0.2

    # --- Branch Control Thresholds ---
    time_threshold = 0.3
    frequency_threshold = 0.3
    space_threshold = 0.3

    # --- Initial Branch Loss (for adaptive threshold init) ---
    time_loss_init = 0.2
    frequency_loss_init = 0.2
    space_loss_init = 0.2
    # ====================================================================

    dim = 7  # Default dimension for ETTm2
    parser = argparse.ArgumentParser(description='Online Transformer Experts (TOE) for Time Series Forecasting')

    # Basic Config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default=f'{sub_dataset}_SL{seq_len}_PL{pred_len}', help='model id')
    # If you want to run Crossformer or DLinear, you need to close the frequency and space branches
    # To turn off TOE, you only need to set the hot start ratio to 1
    parser.add_argument('--model', type=str, default='Online',
                        help='model name, options: [Online, DLinear, Crossformer]')

    # Data Loader
    parser.add_argument('--data', type=str, default=f'{sub_dataset}', help='dataset type')
    parser.add_argument('--root_path', type=str, default=f'/usr/local/lzlconda/file/ssh_4090_2/Anomaly_Detection/dataset/{dataset}/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=f'{sub_dataset}.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='t', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--dataset', type=str, default=f'{sub_dataset}', help='dataset name')

    # Forecasting Task
    parser.add_argument('--seq_len', type=int, default=seq_len, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=pred_len, help='prediction sequence length')

    #Crossformer
    parser.add_argument('--win_size', type=int, default=4, help='attn factor')
    parser.add_argument('--seg_len', type=int, default=8, help='input sequence length')

    # Model Define
    parser.add_argument('--enc_in', type=int, default=dim, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=dim, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=dim, help='output size')
    parser.add_argument('--d_model', type=int, default=d_model, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=d_ff, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=dropout, help='dropout')
    parser.add_argument('--d_core', type=int, default=d_core, help='dimension of core for STAR module')
    parser.add_argument('--d_pick', type=int, default=d_pick, help='kernel size of filter in Freq_branch')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--train_only', type=bool, required=False, default=False,
                        help='perform training on full input dataset without validation and testing')
    parser.add_argument('--use_amp', type=bool, default=False, help='use automatic mixed precision training')


    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # Time_branch
    parser.add_argument('--Time_branch_period', type=int, default=24, help='Period for Time_branch')

    # Freq_branch
    parser.add_argument('--beta', type=float, default=0.5, help='beta value for low-pass filter')
    parser.add_argument('--initial', type=int, default=1, help='use prediction linear initialization')
    parser.add_argument('--layers', type=int, default=2, help='number of layers in Freq_branch')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=lr_main, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multiple gpus')

    # Online Learning & Multi-Branch
    parser.add_argument('--use_time', type=bool, default=True, help='if use time branch')
    parser.add_argument('--use_frequency', type=bool, default=True, help='if use frequency branch')
    parser.add_argument('--use_space', type=bool, default=True, help='if use space branch')
    parser.add_argument('--branches_num', type=int, default=0, help='the total number of used branches')
    parser.add_argument('--warmup_ratio', type=float, default=warmup_ratio, help='warmup data ratio')
    parser.add_argument('--retrain_epochs', type=int, default=250, help='epochs for retraining')
    parser.add_argument('--retrain_lr', type=float, default=lr_retrain, help='learning rate for retraining')
    parser.add_argument('--time_num', type=int, default=96, help='input length for time branch')
    parser.add_argument('--frequency_num', type=int, default=96, help='input length for frequency branch')
    parser.add_argument('--space_num', type=int, default=96, help='input length for space branch')
    parser.add_argument('--time_threshold', type=float, default=time_threshold, help='threshold for time branch')
    parser.add_argument('--frequency_threshold', type=float, default=frequency_threshold,
                        help='threshold for frequency branch')
    parser.add_argument('--space_threshold', type=float, default=space_threshold, help='threshold for space branch')
    parser.add_argument('--time_loss_init', type=float, default=time_loss_init, help='initial loss for time branch')
    parser.add_argument('--frequency_loss_init', type=float, default=frequency_loss_init,
                        help='initial loss for frequency branch')
    parser.add_argument('--space_loss_init', type=float, default=space_loss_init, help='initial loss for space branch')
    parser.add_argument('--max_test_count', type=int, default=20, help='max test count')
    parser.add_argument('--cur_state', type=str, default='', help='current state: warmup, train, or test')

    parser.add_argument('--visualize', type=bool, default=False, help='if visualize the results')

    args = parser.parse_args()

    # Dynamically set the number of branches based on flags
    args.branches_num = int(args.use_time) + int(args.use_frequency) + int(args.use_space)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # Setting record
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_{}_{}'.format(
                args.model_id, args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
                'run', ii)

            exp = Exp(args)  # set experiments
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)

            if args.do_predict:
                print(f'>>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_{}_{}'.format(
            args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
            'test', ii)

        exp = Exp(args)  # set experiments
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        torch.cuda.empty_cache()