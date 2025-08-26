import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import warnings
import matplotlib.font_manager as fm

warnings.filterwarnings("ignore", message="No.*samples*. Your perplexity.*")

import matplotlib
from models.space_branch.Space import Space_Branch_Model
from models.freq_branch.Freq import Freq_Branch_Model
from models.time_branch.Time import Time_Branch_Model
from models.DLinear.DLinear import DLinear
from models.Crossformer.cross_former import Crossformer

MODEL_REGISTRY = {
    'Crossformer': Crossformer,
    'DLinear': DLinear,
    'Online': Time_Branch_Model,
}


class Model(nn.Module):
    """
    TOE-Base for Online Time Series Forecasting.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.batch_size = configs.batch_size
        self.time_num = configs.time_num
        self.frequency_num = configs.frequency_num
        self.space_num = configs.space_num
        self.vis_counter = 0

        self.time_term_mem = []
        self.frequency_term_mem = []
        self.space_term_mem = []
        self.fusion_weights = nn.Parameter(torch.ones(3).float() / configs.branches_num)

        # Initialize adaptive thresholds and their upper caps
        # The adaptive threshold is dynamic, initialized by the branch's initial loss
        self.time_threshold = configs.time_loss_init
        self.frequency_threshold = configs.frequency_loss_init
        self.space_threshold = configs.space_loss_init

        # The threshold cap is a fixed upper limit
        self.time_threshold_cap = configs.time_threshold
        self.frequency_threshold_cap = configs.frequency_threshold
        self.space_threshold_cap = configs.space_threshold

        self.retrain_epochs = configs.retrain_epochs
        self.retrain_lr = configs.retrain_lr

        self.time_model = MODEL_REGISTRY[configs.model](configs)
        self.space_model = Space_Branch_Model(configs)
        self.frequency_model = Freq_Branch_Model(configs)

        if self.configs.visualize:
            self.vis_batch_idx = 0
            self.vis_output_path = f"./visualization_results/{self.configs.data}/pred_len:{self.configs.pred_len}_noise:{self.configs.noise_level}/"
            if not os.path.exists(self.vis_output_path):
                os.makedirs(self.vis_output_path)

    def _get_output(self, x=None, x_mark_enc=None, branch="time", params=None):
        if params:
            self.load_best_params(params, branch=branch)
        if branch == "time":
            output = self.time_model(x_enc=x, x_mark_enc=x_mark_enc)
        elif branch == "frequency":
            output = self.frequency_model(x_enc=x, x_mark_enc=x_mark_enc)
        elif branch == "space":
            output = self.space_model(x)
        return output.float()

    def forward(self, x, x_mark_enc=None, mode="train", y=None, batch_idx=None):
        x = x.float()
        if y is not None:
            y = y.float()

        if mode != 'train' and batch_idx is not None and self.configs.visualize:
            self.vis_batch_idx = batch_idx

        if mode == "train":
            time_output, frequency_output, space_output = None, None, None
            if self.configs.use_time:
                time_output = self._time_term_forward(x, y, x_mark_enc=x_mark_enc)
            if self.configs.use_frequency:
                frequency_output = self._frequency_term_forward(x, x_mark_enc=x_mark_enc, y=y)
            if self.configs.use_space:
                space_output = self._space_term_forward(x, y)
            outputs = self.fuse_branches(time_output, frequency_output, space_output)
            return outputs
        else:
            with torch.no_grad():
                time_output, frequency_output, space_output = None, None, None
                if self.configs.use_time:
                    time_output = self._time_term_forward(x, y, x_mark_enc=x_mark_enc)
                if self.configs.use_frequency:
                    frequency_output = self._frequency_term_forward(x, x_mark_enc=x_mark_enc, y=y)
                if self.configs.use_space:
                    space_output = self._space_term_forward(x, y)
                outputs = self.fuse_branches(time_output, frequency_output, space_output)
                return outputs

    def _visualize_adaptation(self, x_lookback, y_truth, pred_before_retrain, pred_after_retrain, branch_name):
        lookback_np = x_lookback[0].cpu().numpy()
        truth_np = y_truth[0].cpu().numpy()
        pred_before_np = pred_before_retrain[0].cpu().numpy()
        pred_after_np = pred_after_retrain[0].cpu().numpy()
        num_plots = 3
        seq_len = self.seq_len
        pred_len = self.pred_len
        time_history = np.arange(0, seq_len)
        time_future = np.arange(seq_len, seq_len + pred_len)
        color_history = '#377eb8'
        color_truth = '#B2DF8A'
        color_pred_before = '#FB9A99'
        color_pred_after = '#E31A1C'
        color_vline = '#cccccc'
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(40, 10), sharex=True, sharey=False)
        for i in range(num_plots):
            ax = axes[i]
            all_y_values_channel = np.concatenate([
                lookback_np[:, i], truth_np[:, i], pred_before_np[:, i], pred_after_np[:, i]
            ])
            if all_y_values_channel.size > 0:
                y_min, y_max = np.nanmin(all_y_values_channel), np.nanmax(all_y_values_channel)
                y_range = y_max - y_min
                y_padding = y_range * 0.05 if y_range > 1e-6 else 0.1
                ax.set_ylim(y_min - y_padding, y_max + y_padding)
            ax.plot(time_history, lookback_np[:, i], color=color_history, label='History', linewidth=3)
            ax.plot(time_future, truth_np[:, i], color=color_truth, linewidth=5, label='Ground Truth')
            ax.plot(time_future, pred_before_np[:, i], color=color_pred_before, linestyle='--', linewidth=4,
                    label='Before Retrain')
            ax.plot(time_future, pred_after_np[:, i, ], color=color_pred_after, linestyle='-', linewidth=4.5,
                    label='After Retrain')
            ax.axvline(x=seq_len - 0.5, color=color_vline, linestyle=':', linewidth=5)
            ax.set_title(f'Channel {i + 1}', fontsize=32, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=24)
            ax.grid(True, which='both', linestyle='--', linewidth=1.0)
            if i == 0:
                ax.set_ylabel('Value', fontsize=32, fontweight='bold')
        top_y_level = 0.98
        fig.suptitle(
            f'{branch_name.upper()} Branch (Prediction Window: {self.configs.pred_len})',
            x=0.07, y=top_y_level, ha='left', va='top', fontsize=48, fontweight='bold'
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc='upper right', bbox_to_anchor=(0.95, top_y_level),
            ncol=4, fontsize=30, frameon=False
        )
        fig.text(0.5, 0.04, 'Time Steps', ha='center', va='center', fontsize=40, fontweight='bold')
        plt.tight_layout(rect=[0.05, 0.08, 1, 0.90])
        save_path = os.path.join(self.vis_output_path, f'{branch_name}_adapt_id_{self.vis_counter:05d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.vis_counter += 1
        print(f"\n[Visualized] Adaptation proof saved to {save_path}")

    def _time_term_forward(self, x, y, x_mark_enc=None):
        time_x = self._get_segment(x, data_num=self.time_num)
        time_x_mark = self._get_segment(x_mark_enc, data_num=self.time_num)
        y_true = y[:, -self.pred_len:, :].to(x.device)
        branch_name = "time"

        if self.configs.cur_state == "warmup":
            time_output = self._get_output(time_x, x_mark_enc=time_x_mark, branch=branch_name)
            loss = F.mse_loss(time_output, y_true)
            current_params = [copy.deepcopy(self.time_model.state_dict())]
            if not self.time_term_mem or loss.item() < self.time_term_mem[0]["loss"]:
                self.time_term_mem = [{"params": current_params, "loss": loss.item()}]
        else:
            min_loss = float('inf')
            best_params = None
            with torch.no_grad():
                if self.time_term_mem:
                    for params_dict in self.time_term_mem:
                        temp_output = self._get_output(x=time_x, x_mark_enc=time_x_mark, branch=branch_name,
                                                       params=params_dict["params"])
                        loss = F.mse_loss(temp_output, y_true)
                        if loss.item() < min_loss:
                            min_loss = loss.item()
                            best_params = params_dict["params"]
                        # If loss is below adaptive threshold, stop searching early
                        if min_loss < self.time_threshold:
                            break
                        del temp_output, loss

            if best_params is None and self.time_term_mem: best_params = self.time_term_mem[0]['params']

            # Core decision logic: take action based on the minimum loss found
            if min_loss < self.time_threshold:
                # Case 1: Performance is good enough, use the best historical model
                with torch.no_grad():
                    time_output = self._get_output(x=time_x, x_mark_enc=time_x_mark, branch=branch_name,
                                                   params=best_params)
            elif self.configs.cur_state == "train":
                # Case 2: Poor performance in training mode, trigger retraining and threshold adaptation
                visualize_this_step = self.configs.visualize
                pred_before = None
                if visualize_this_step:
                    with torch.no_grad():
                        pred_before = self._get_output(x=time_x, x_mark_enc=time_x_mark, branch=branch_name,
                                                       params=best_params)

                # Retrain the branch on the current sample
                time_output = self._train_branch(x=time_x, y=y, x_mark_enc=time_x_mark, branch=branch_name,
                                                 initial_params=best_params)

                # Update the adaptive threshold
                self.time_threshold = min(min_loss, self.time_threshold_cap)

                if visualize_this_step:
                    self._visualize_adaptation(
                        x_lookback=x, y_truth=y_true, pred_before_retrain=pred_before,
                        pred_after_retrain=time_output, branch_name=branch_name
                    )
            else:
                # Case 3: Poor performance in test/validation mode, use the best available model without retraining
                with torch.no_grad():
                    time_output = self._get_output(x=time_x, x_mark_enc=time_x_mark, branch=branch_name,
                                                   params=best_params)

        if time_output is None:
            with torch.no_grad():
                params = self.time_term_mem[0]['params'] if self.time_term_mem else None
                time_output = self._get_output(x=time_x, x_mark_enc=time_x_mark, branch=branch_name, params=params)

        del time_x, time_x_mark, y_true
        return time_output

    def _frequency_term_forward(self, x, x_mark_enc=None, y=None):
        frequency_x = self._get_segment(x, data_num=self.frequency_num)
        frequency_x_mark_enc = self._get_segment(x_mark_enc, data_num=self.frequency_num)
        y_true = y[:, -self.pred_len:, :].to(x.device)
        branch_name = "frequency"

        if self.configs.cur_state == "warmup":
            frequency_output = self._get_output(frequency_x, x_mark_enc=frequency_x_mark_enc, branch=branch_name)
            loss = F.mse_loss(frequency_output, y_true)
            current_params = [copy.deepcopy(self.frequency_model.state_dict())]
            if not self.frequency_term_mem or loss.item() < self.frequency_term_mem[0]["loss"]:
                self.frequency_term_mem = [{"params": copy.deepcopy(current_params), "loss": loss.item()}]
        else:
            min_loss = float('inf')
            best_params = None
            with torch.no_grad():
                if self.frequency_term_mem:
                    for params_dict in self.frequency_term_mem:
                        temp_output = self._get_output(frequency_x, x_mark_enc=frequency_x_mark_enc, branch=branch_name,
                                                       params=params_dict["params"][0])
                        loss = F.mse_loss(temp_output, y_true)
                        if loss.item() < min_loss:
                            min_loss, best_params = loss.item(), params_dict["params"]
                        if min_loss < self.frequency_threshold:
                            break
                        del temp_output, loss

            if best_params is None and self.frequency_term_mem: best_params = self.frequency_term_mem[0]["params"]

            if min_loss < self.frequency_threshold:
                with torch.no_grad():
                    frequency_output = self._get_output(frequency_x, x_mark_enc=frequency_x_mark_enc,
                                                        branch=branch_name, params=best_params[0])
            elif self.configs.cur_state == "train":
                visualize_this_step = self.configs.visualize
                pred_before = None
                if visualize_this_step:
                    with torch.no_grad():
                        pred_before = self._get_output(frequency_x, x_mark_enc=frequency_x_mark_enc, branch=branch_name,
                                                       params=best_params[0])

                frequency_output = self._train_branch(x=frequency_x, x_mark_enc=frequency_x_mark_enc, y=y,
                                                      branch=branch_name, initial_params=best_params)

                self.frequency_threshold = min(min_loss, self.frequency_threshold_cap)

                if visualize_this_step:
                    self._visualize_adaptation(x, y_true, pred_before, frequency_output, branch_name)

            else:
                with torch.no_grad():
                    frequency_output = self._get_output(frequency_x, x_mark_enc=frequency_x_mark_enc,
                                                        branch=branch_name, params=best_params[0])

        if frequency_output is None:
            with torch.no_grad():
                params = self.frequency_term_mem[0]['params'][0] if self.frequency_term_mem else None
                frequency_output = self._get_output(frequency_x, x_mark_enc=frequency_x_mark_enc, branch=branch_name,
                                                    params=params)

        del frequency_x, frequency_x_mark_enc, y_true
        return frequency_output

    def _space_term_forward(self, x, y):
        space_x = self._get_segment(x, data_num=self.space_num)
        y_true = y[:, -self.pred_len:, :].to(x.device)
        branch_name = "space"

        if self.configs.cur_state == "warmup":
            space_output = self._get_output(x=space_x, branch=branch_name)
            loss = F.mse_loss(space_output, y_true)
            current_params = [copy.deepcopy(self.space_model.state_dict())]
            if not self.space_term_mem or loss.item() < self.space_term_mem[0]["loss"]:
                self.space_term_mem = [{"params": current_params, "loss": loss.item()}]
        else:
            min_loss = float('inf')
            best_params = None
            with torch.no_grad():
                if self.space_term_mem:
                    for params_dict in self.space_term_mem:
                        temp_output = self._get_output(x=space_x, params=params_dict["params"][0], branch=branch_name)
                        loss = F.mse_loss(temp_output, y_true)
                        if loss.item() < min_loss:
                            min_loss, best_params = loss.item(), params_dict["params"]
                        if min_loss < self.space_threshold:
                            break
                        del temp_output, loss

            if best_params is None and self.space_term_mem: best_params = self.space_term_mem[0]["params"]

            if min_loss < self.space_threshold:
                with torch.no_grad():
                    space_output = self._get_output(x=space_x, params=best_params[0], branch=branch_name)
            elif self.configs.cur_state == "train":
                visualize_this_step = self.configs.visualize
                pred_before = None
                if visualize_this_step:
                    with torch.no_grad():
                        pred_before = self._get_output(x=space_x, params=best_params[0], branch=branch_name)

                space_output = self._train_branch(x=space_x, y=y, branch=branch_name, initial_params=best_params)

                self.space_threshold = min(min_loss, self.space_threshold_cap)

                if visualize_this_step:
                    self._visualize_adaptation(x, y_true, pred_before, space_output, branch_name)
            else:
                with torch.no_grad():
                    space_output = self._get_output(x=space_x, params=best_params[0], branch=branch_name)

        if space_output is None:
            with torch.no_grad():
                params = self.space_term_mem[0]['params'][0] if self.space_term_mem else None
                space_output = self._get_output(x=space_x, params=params, branch=branch_name)

        del space_x, y_true
        return space_output

    def fuse_branches(self, time_output=None, frequency_output=None, space_output=None):
        target_shape, device = None, None
        if time_output is not None:
            target_shape, device = time_output.shape, time_output.device
        elif frequency_output is not None:
            target_shape, device = frequency_output.shape, frequency_output.device
        elif space_output is not None:
            target_shape, device = space_output.shape, space_output.device
        else:
            return None
        fused_output = torch.zeros(target_shape, device=device)
        if time_output is not None and self.configs.use_time: fused_output += self.fusion_weights[0] * time_output
        if frequency_output is not None and self.configs.use_frequency: fused_output += self.fusion_weights[
                                                                                            1] * frequency_output
        if space_output is not None and self.configs.use_space: fused_output += self.fusion_weights[2] * space_output
        return fused_output

    def _get_segment(self, x, data_num=96):
        if x is None: return None
        data_num = min(data_num, x.shape[1])
        return x[:, -data_num:, :].clone()

    def load_best_params(self, initial_params=None, branch="time"):
        if initial_params is None: return
        params_to_load = initial_params[0] if isinstance(initial_params, list) else initial_params
        model_map = {"time": self.time_model, "frequency": self.frequency_model, "space": self.space_model}
        model_map[branch].load_state_dict(params_to_load, strict=True)

    def _predict_branch(self, x=None, x_mark_enc=None, y=None, branch="time"):
        with torch.no_grad():
            criterion_mse = nn.MSELoss()
            criterion_mae = nn.L1Loss()
            y_true = y[:, -self.pred_len:, :]
            if branch == "time":
                time_x = self._get_segment(x, data_num=self.time_num)
                time_x_mark = self._get_segment(x_mark_enc, data_num=self.time_num)
                time_output = self._get_output(x=time_x, x_mark_enc=time_x_mark, branch="time")
                mse_loss = criterion_mse(time_output, y_true.to(time_output.device))
                mae_loss = criterion_mae(time_output, y_true.to(time_output.device))
                outputs = time_output
            elif branch == "frequency":
                frequency_x = self._get_segment(x, data_num=self.frequency_num)
                frequency_x_mark_enc = self._get_segment(x_mark_enc, data_num=self.frequency_num)
                frequency_output = self._get_output(x=frequency_x, x_mark_enc=frequency_x_mark_enc, branch="frequency")
                mse_loss = criterion_mse(frequency_output, y_true.to(frequency_output.device))
                mae_loss = criterion_mae(frequency_output, y_true.to(frequency_output.device))
                outputs = frequency_output
            elif branch == "space":
                space_x = self._get_segment(x, data_num=self.space_num)
                space_output = self._get_output(x=space_x, branch="space")
                mse_loss = criterion_mse(space_output, y_true.to(space_output.device))
                mae_loss = criterion_mae(space_output, y_true.to(space_output.device))
                outputs = space_output
            else:
                raise ValueError(f"Invalid branch name: {branch}")
            mse_val = mse_loss.item()
            mae_val = mae_loss.item()
            del y_true, mse_loss, mae_loss
            return torch.tensor(mse_val), torch.tensor(mae_val), outputs

    def _train_branch(self, x, y, x_mark_enc=None, branch="time", initial_params=None):
        model_map = {"time": self.time_model, "frequency": self.frequency_model, "space": self.space_model}
        model = model_map[branch]
        optimizer = optim.AdamW(model.parameters(), lr=self.retrain_lr)
        criterion_mse = nn.MSELoss()
        y_true = y[:, -self.pred_len:, :].to(x.device)
        best_loss = float("inf")
        best_params_state_dict = None
        self.load_best_params(initial_params, branch=branch)
        progress_bar = tqdm(range(self.retrain_epochs), desc=f"{branch.capitalize()}-Branch Retraining", leave=False)
        for epoch in progress_bar:
            optimizer.zero_grad()
            output = self._get_output(x, x_mark_enc=x_mark_enc, branch=branch)
            loss = criterion_mse(output, y_true)
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            progress_bar.set_postfix({"loss": f"{current_loss:.6f}"})
            if current_loss < best_loss:
                best_loss = current_loss
                best_params_state_dict = copy.deepcopy(model.state_dict())
            # Early stop retraining if the loss drops below the branch's adaptive threshold
            if current_loss < getattr(self, f"{branch}_threshold"):
                break
        del loss, output
        if best_params_state_dict is not None:
            mem_pool = getattr(self, f"{branch}_term_mem")
            mem_pool.append({"params": [best_params_state_dict], "loss": best_loss})
            if len(mem_pool) > self.configs.retrain_epochs: mem_pool.pop(0)
        final_params_to_load = [best_params_state_dict] if best_params_state_dict is not None else initial_params
        self.load_best_params(final_params_to_load, branch=branch)
        with torch.no_grad():
            final_output = self._get_output(x, x_mark_enc=x_mark_enc, branch=branch)
        del optimizer, y_true, best_params_state_dict, final_params_to_load
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return final_output.detach()