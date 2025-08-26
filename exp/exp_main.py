import csv
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
import os
import time
import warnings
from tqdm import tqdm

from models.TOE import Model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):

    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.args = args

    def _build_model(self):
        model = Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        # 0. Initialization and Timers
        time_branch_train_time, frequency_branch_train_time, space_branch_train_time = 0, 0, 0
        time_branch_test_time, frequency_branch_test_time, space_branch_test_time = 0, 0, 0
        total_train_start_time = time.time()

        # Data Preparation
        train_data_original, _ = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_data_len = len(train_data_original)
        adjusted_train_len = (train_data_len // self.args.batch_size) * self.args.batch_size
        train_data = Subset(train_data_original, range(adjusted_train_len))

        # 1. Warmup Phase: Train on a subset of data to initialize branch memory pools.
        self.args.cur_state = "warmup"
        warmup_size = int(self.args.warmup_ratio * adjusted_train_len)
        all_indices = list(range(adjusted_train_len))
        np.random.shuffle(all_indices)
        warmup_indices = all_indices[:warmup_size]
        online_indices = all_indices[warmup_size:]

        print(f"Total training samples: {adjusted_train_len}")
        print(f"Warmup samples: {len(warmup_indices)}")
        print(f"Online training samples: {len(online_indices)}")

        if len(warmup_indices) > 0:
            warmup_dataset = Subset(train_data, warmup_indices)
            warmup_loader = DataLoader(warmup_dataset, batch_size=self.args.batch_size, shuffle=True,
                                       num_workers=self.args.num_workers)
            warmup_loop = tqdm(warmup_loader, desc='Warmup Phase')
            self.model.train()
            warmup_start_time = time.time()
            for batch_x, batch_y, batch_x_mark, batch_y_mark in warmup_loop:
                model_optim.zero_grad()
                batch_x, batch_y, batch_x_mark = batch_x.float().to(self.device), batch_y.float().to(
                    self.device), batch_x_mark.float().to(self.device)
                outputs = self.model(batch_x, x_mark_enc=batch_x_mark, mode="train", y=batch_y)
                loss = criterion(outputs, batch_y[:, -self.args.pred_len:, :])
                loss.backward()
                model_optim.step()
                warmup_loop.set_postfix(loss=loss.item())

            warmup_duration = time.time() - warmup_start_time
            num_active_branches = self.args.use_time + self.args.use_frequency + self.args.use_space
            shared_warmup_time = warmup_duration / num_active_branches if num_active_branches > 0 else 0
            if self.args.use_time: time_branch_train_time += shared_warmup_time
            if self.args.use_frequency: frequency_branch_train_time += shared_warmup_time
            if self.args.use_space: space_branch_train_time += shared_warmup_time

        # 2. Online Training Phase: Adaptively train branches or fusion weights.
        self.args.cur_state = "train"
        if len(online_indices) > 0:
            online_dataset = Subset(train_data, online_indices)
            online_loader = DataLoader(online_dataset, batch_size=self.args.batch_size, shuffle=True,
                                       num_workers=self.args.num_workers)
            online_loop = tqdm(online_loader, desc='Online Training Phase')

            for batch_x, batch_y, batch_x_mark, batch_y_mark in online_loop:
                batch_x, batch_y, batch_x_mark = batch_x.float().to(self.device), batch_y.float().to(
                    self.device), batch_x_mark.float().to(self.device)

                # Evaluate performance of each branch on the current batch.
                self.model.eval()
                with torch.no_grad():
                    time_loss, _, time_output = self.model._predict_branch(x=batch_x, y=batch_y,
                                                                           x_mark_enc=batch_x_mark,
                                                                           branch="time") if self.args.use_time else (
                    torch.tensor(torch.inf), 0, None)
                    space_loss, _, space_output = self.model._predict_branch(x=batch_x, y=batch_y,
                                                                             branch="space") if self.args.use_space else (
                    torch.tensor(torch.inf), 0, None)
                    frequency_loss, _, frequency_output = self.model._predict_branch(x=batch_x, y=batch_y,
                                                                                     x_mark_enc=batch_x_mark,
                                                                                     branch="frequency") if self.args.use_frequency else (
                    torch.tensor(torch.inf), 0, None)
                    online_loop.set_postfix(time=f"{time_loss.item():.4f}", frequency=f"{frequency_loss.item():.4f}",
                                            space=f"{space_loss.item():.4f}")

                self.model.train()
                is_time_poor = self.args.use_time and time_loss > self.model.time_threshold
                is_space_poor = self.args.use_space and space_loss > self.model.space_threshold
                is_frequency_poor = self.args.use_frequency and frequency_loss > self.model.frequency_threshold

                # If any branch performs poorly, retrain it. Otherwise, train the fusion weights.
                if is_time_poor or is_space_poor or is_frequency_poor:
                    if is_time_poor:
                        start_t = time.time();
                        self.model._time_term_forward(batch_x, batch_y, batch_x_mark);
                        time_branch_train_time += time.time() - start_t
                    if is_space_poor:
                        start_t = time.time();
                        self.model._space_term_forward(batch_x, batch_y);
                        space_branch_train_time += time.time() - start_t
                    if is_frequency_poor:
                        start_t = time.time();
                        self.model._frequency_term_forward(batch_x, batch_x_mark, batch_y);
                        frequency_branch_train_time += time.time() - start_t
                else:
                    fused_output = self.model.fuse_branches(time_output=time_output, frequency_output=frequency_output,
                                                            space_output=space_output)
                    if fused_output is not None:
                        start_fuse_train = time.time()
                        fused_loss = criterion(fused_output, batch_y[:, -self.args.pred_len:, :])
                        model_optim.zero_grad()
                        fused_loss.backward()
                        model_optim.step()
                        fuse_train_duration = time.time() - start_fuse_train
                        if num_active_branches > 0:
                            shared_fuse_time = fuse_train_duration / num_active_branches
                            if self.args.use_time: time_branch_train_time += shared_fuse_time
                            if self.args.use_frequency: frequency_branch_train_time += shared_fuse_time
                            if self.args.use_space: space_branch_train_time += shared_fuse_time

        total_train_duration_min = (time.time() - total_train_start_time) / 60
        print(f"\n--- Total Training Time: {total_train_duration_min:.2f} minutes ---")

        # 3. Online Testing Phase: Predict and then update based on the previous sample.
        self.args.cur_state = "test"
        total_test_start_time = time.time()
        test_loop = tqdm(test_loader, desc='Online Testing Phase')
        previous_batch = None
        batch_metrics = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loop):
            # After the first prediction, use the previous ground truth to update the model.
            if i > 0 and previous_batch is not None:
                self.model.train()
                prev_x, prev_y, prev_x_mark, _ = previous_batch
                with torch.no_grad():
                    time_loss, _, _ = self.model._predict_branch(x=prev_x, y=prev_y, x_mark_enc=prev_x_mark,
                                                                 branch="time") if self.args.use_time else (
                    torch.tensor(torch.inf), 0, None)
                    space_loss, _, _ = self.model._predict_branch(x=prev_x, y=prev_y,
                                                                  branch="space") if self.args.use_space else (
                    torch.tensor(torch.inf), 0, None)
                    frequency_loss, _, _ = self.model._predict_branch(x=prev_x, y=prev_y, x_mark_enc=prev_x_mark,
                                                                      branch="frequency") if self.args.use_frequency else (
                    torch.tensor(torch.inf), 0, None)
                if self.args.use_time and time_loss > self.model.time_threshold:
                    start_t = time.time();
                    self.model._time_term_forward(prev_x, prev_y, prev_x_mark);
                    time_branch_test_time += time.time() - start_t
                if self.args.use_space and space_loss > self.model.space_threshold:
                    start_t = time.time();
                    self.model._space_term_forward(prev_x, prev_y);
                    space_branch_test_time += time.time() - start_t
                if self.args.use_frequency and frequency_loss > self.model.frequency_threshold:
                    start_t = time.time();
                    self.model._frequency_term_forward(prev_x, prev_x_mark, prev_y);
                    frequency_branch_test_time += time.time() - start_t

            # Make a prediction on the current batch.
            self.model.eval()
            batch_x, batch_y, batch_x_mark = batch_x.float().to(self.device), batch_y.float().to(
                self.device), batch_x_mark.float().to(self.device)
            with torch.no_grad():
                if self.args.use_time:
                    start_t = time.time();
                    time_mse, time_mae, time_output = self.model._predict_branch(x=batch_x, y=batch_y,
                                                                                 x_mark_enc=batch_x_mark,
                                                                                 branch="time");
                    time_branch_test_time += time.time() - start_t
                else:
                    time_mse, time_mae, time_output = 0, 0, None
                if self.args.use_space:
                    start_t = time.time();
                    space_mse, space_mae, space_output = self.model._predict_branch(x=batch_x, y=batch_y,
                                                                                    branch="space");
                    space_branch_test_time += time.time() - start_t
                else:
                    space_mse, space_mae, space_output = 0, 0, None
                if self.args.use_frequency:
                    start_t = time.time();
                    frequency_mse, frequency_mae, frequency_output = self.model._predict_branch(x=batch_x, y=batch_y,
                                                                                                x_mark_enc=batch_x_mark,
                                                                                                branch="frequency");
                    frequency_branch_test_time += time.time() - start_t
                else:
                    frequency_mse, frequency_mae, frequency_output = 0, 0, None

                fused_output = self.model.fuse_branches(time_output=time_output, frequency_output=frequency_output,
                                                        space_output=space_output)
                final_mae, final_mse = (0, 0)
                if fused_output is not None:
                    true, pred = batch_y[:, -self.args.pred_len:, :].cpu().numpy(), fused_output.cpu().numpy()
                    final_mae, final_mse = mean_absolute_error(true.flatten(), pred.flatten()), mean_squared_error(
                        true.flatten(), pred.flatten())
                batch_metrics.append({'time_mae': time_mae, 'time_mse': time_mse, 'frequency_mae': frequency_mae,
                                      'frequency_mse': frequency_mse, 'space_mae': space_mae, 'space_mse': space_mse,
                                      'final_mae': final_mae, 'final_mse': final_mse})
                test_loop.set_postfix(MAE=final_mae, MSE=final_mse)
            previous_batch = (batch_x, batch_y, batch_x_mark, batch_y_mark)

        total_test_duration_min = (time.time() - total_test_start_time) / 60
        print(f"\n--- Total Test Time: {total_test_duration_min:.2f} minutes ---")

        # 4. Results Saving
        final_metrics = pd.DataFrame(batch_metrics).mean().to_dict()
        print(
            f"Final Average Time Branch - MAE: {final_metrics.get('time_mae', 0):.4f}, MSE: {final_metrics.get('time_mse', 0):.4f}")
        print(
            f"Final Average Frequency Branch - MAE: {final_metrics.get('frequency_mae', 0):.4f}, MSE: {final_metrics.get('frequency_mse', 0):.4f}")
        print(
            f"Final Average Space Branch - MAE: {final_metrics.get('space_mae', 0):.4f}, MSE: {final_metrics.get('space_mse', 0):.4f}")
        print(
            f"Final Average Fused Output - MAE: {final_metrics.get('final_mae', 0):.4f}, MSE: {final_metrics.get('final_mse', 0):.4f}")

        # Create a unique directory for this experiment's results.
        base_info = f"{self.args.dataset}/{self.args.model}_{self.args.warmup_ratio}/time:{self.args.time_threshold}_freq:{self.args.frequency_threshold}_space:{self.args.space_threshold}"
        main_folder_name = f"MAE_{final_metrics.get('final_mae', 0):.4f}_MSE_{final_metrics.get('final_mse', 0):.4f}_in{self.args.seq_len}_pred{self.args.pred_len}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_path = os.path.join('./test_results/', base_info, main_folder_name)
        os.makedirs(save_path, exist_ok=True)

        # Save model components, metrics, and hyperparameters.
        checkpoint_path = os.path.join(save_path, 'checkpoints')
        os.makedirs(checkpoint_path, exist_ok=True)
        if self.args.use_frequency: torch.save(self.model.frequency_term_mem,
                                               os.path.join(checkpoint_path, 'frequency_branch_mem.pth'))
        if self.args.use_time: torch.save(self.model.time_term_mem,
                                          os.path.join(checkpoint_path, 'time_branch_mem.pth'))
        if self.args.use_space: torch.save(self.model.space_term_mem,
                                           os.path.join(checkpoint_path, 'space_branch_mem.pth'))
        torch.save(self.model.fusion_weights, os.path.join(checkpoint_path, 'fusion_weights.pth'))

        pd.DataFrame(batch_metrics).to_csv(os.path.join(save_path, 'batch_metrics.csv'), index_label="Batch")
        with open(os.path.join(save_path, 'final_metrics.txt'), 'w') as f:
            for key, value in final_metrics.items(): f.write(f'{key}: {value}\n')
        with open(os.path.join(save_path, 'hparams.yaml'), 'w') as f:
            yaml.dump(vars(self.args), f)

        # Save a comprehensive summary of the experiment's performance and cost.
        summary_file_path = os.path.join(save_path, 'experiment_summary.csv')
        fieldnames = [
            'datetime', 'dataset', 'model', 'warmup_ratio',
            'time_mae', 'time_mse', 'frequency_mae', 'frequency_mse', 'space_mae', 'space_mse',
            'final_mae', 'final_mse',
            'time_mem_size', 'frequency_mem_size', 'space_mem_size',
            'total_train_time_min', 'time_branch_train_time_min', 'frequency_branch_train_time_min',
            'space_branch_train_time_min',
            'total_test_time_min', 'time_branch_test_time_min', 'frequency_branch_test_time_min',
            'space_branch_test_time_min'
        ]
        summary_data = {
            'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset': self.args.dataset,
            'model': self.args.model,
            'warmup_ratio': self.args.warmup_ratio,
            'time_mae': final_metrics.get('time_mae', 0), 'time_mse': final_metrics.get('time_mse', 0),
            'frequency_mae': final_metrics.get('frequency_mae', 0),
            'frequency_mse': final_metrics.get('frequency_mse', 0),
            'space_mae': final_metrics.get('space_mae', 0), 'space_mse': final_metrics.get('space_mse', 0),
            'final_mae': final_metrics.get('final_mae', 0), 'final_mse': final_metrics.get('final_mse', 0),
            'time_mem_size': len(self.model.time_term_mem) if self.args.use_time else 0,
            'frequency_mem_size': len(self.model.frequency_term_mem) if self.args.use_frequency else 0,
            'space_mem_size': len(self.model.space_term_mem) if self.args.use_space else 0,
            'total_train_time_min': total_train_duration_min,
            'time_branch_train_time_min': time_branch_train_time / 60,
            'frequency_branch_train_time_min': frequency_branch_train_time / 60,
            'space_branch_train_time_min': space_branch_train_time / 60,
            'total_test_time_min': total_test_duration_min,
            'time_branch_test_time_min': time_branch_test_time / 60,
            'frequency_branch_test_time_min': frequency_branch_test_time / 60,
            'space_branch_test_time_min': space_branch_test_time / 60,
        }
        try:
            with open(summary_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(summary_data)
            print(f"\nExperiment summary saved to: {summary_file_path}")
        except IOError as e:
            print(f"Error writing to summary file {summary_file_path}: {e}")

        return self.model