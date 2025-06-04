# utils.py
import torch
import numpy as np
from sklearn.utils import resample
from lifelines.utils import concordance_index as concordance_index_lifelines
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from statistics import median
import copy
import random
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_patient_data(
    patient_json_path: str,
    test_ratio: float = 0.5,
    k: int = 5
):
    with open(patient_json_path, 'r') as f:
        loaded = json.load(f)

    train_data = loaded["train_data"]
    test_data = loaded["test_data"]
    train_val_split = [
        (item["train"], item["val"]) for item in loaded["train_val_split"]
    ]
    data_dict = loaded["data_dict"]
    return train_data, test_data, train_val_split,data_dict

def bootstrap_c_index(y_data, predictions, n_iterations=1000):
    c_indices = []
    for _ in range(n_iterations):
        indices = resample(np.arange(len(y_data)))
        sample_y = y_data[indices]
        sample_preds = predictions[indices]
        c_index = cal_ci(sample_y, sample_preds)
        if c_index[0].device.type == 'cuda':
            c_indices.append(c_index[0].cpu().item())
        else:
            c_indices.append(c_index[0].item())

    lower_bound = np.percentile(c_indices, 2.5)
    upper_bound = np.percentile(c_indices, 97.5)

    return lower_bound, upper_bound, np.mean(c_indices)

def compute_concordance_index_ci(model, data, n_bootstraps=1000, seed=None):
    np.random.seed(seed)
    concordance_scores = []
    for _ in range(n_bootstraps):
        bootstrap_sample = data.sample(n=len(data), replace=True)
        c_index = concordance_index(bootstrap_sample['duration'],
                                    -model.predict_partial_hazard(bootstrap_sample), bootstrap_sample['event'])
        concordance_scores.append(c_index)
    lower_ci = np.percentile(concordance_scores, 2.5)
    upper_ci = np.percentile(concordance_scores, 97.5)
    return np.mean(concordance_scores), (lower_ci, upper_ci)

def cox_loss(y_true, y_pred):
    time_value = torch.squeeze(y_true[0:, 0])
    event = torch.squeeze(y_true[0:, 1]).type(torch.bool)
    score = torch.squeeze(y_pred)

    ix = torch.where(event)[0]
    if ix.nelement() == 0:
        return torch.tensor(1.0, requires_grad=True)

    sel_time = time_value[ix]
    sel_mat = (sel_time.unsqueeze(1).expand(1, sel_time.size()[0],
                                            time_value.size()[0]).squeeze() <= time_value).float()
    p_lik = score[ix] - torch.log(torch.sum(sel_mat * torch.exp(score), axis=-1))

    loss = -torch.mean(p_lik)

    return loss


def concordance_index_(y_true, y_pred):
    time_value = torch.squeeze(y_true[0:, 0])
    event = torch.squeeze(y_true[0:, 1]).type(torch.bool)

    time_1 = time_value.unsqueeze(1).expand(1, time_value.size()[0], time_value.size()[0]).squeeze()
    event_1 = event.unsqueeze(1).expand(1, event.size()[0], event.size()[0]).squeeze()
    ix = torch.where(torch.logical_and(time_1 < time_value, event_1))

    s1 = y_pred[ix[0]]
    s2 = y_pred[ix[1]]
    ci = torch.mean((s1 < s2).float())

    return ci



def cal_loss(y, pred, criterion):  # criterion=cox_loss
    loss_dict = {}
    for target_class in range(int(y.shape[1] / 2)):
        loss_dict[target_class] = criterion(y[:, target_class * 2:(target_class + 1) * 2].to(device), pred.to(device))
    return loss_dict


def cal_ci(y, pred):
    ci_dict = {}
    for target_class in range(int(y.shape[1] / 2)):
        ci_dict[target_class] = concordance_index_(y[:, target_class * 2:(target_class + 1) * 2].to(device),
                                                   -pred.to(device))
    return ci_dict


def concordance_index_test(y_true, y_pred):
    actual_durations = y_true[0:, 0].cpu().numpy()
    predicted_scores = y_pred.cpu().numpy()
    event_observed = y_true[0:, 1].cpu().numpy()
    if np.isnan(predicted_scores).any():
        return 0.5
    c_index = concordance_index(event_times=actual_durations,
                                predicted_scores=predicted_scores,
                                event_observed=event_observed)
    return c_index


def cal_ci_test(y, pred):
    ci_dict = {}
    for target_class in range(int(y.shape[1] / 2)):
        ci_dict[target_class] = concordance_index_test(y[:, target_class * 2:(target_class + 1) * 2].to(device),
                                                       -pred.to(device))
    return ci_dict


def prediction(model, dl, criterion):
    i = 0
    for data in dl:
        identifier, labels, inputs1, inputs2, clin, prompt, report_code = data[0], data[1].to(device), data[
            2].to(device), \
            data[
                3].to(
                device), data[4].to(device), data[5].to(device), data[6].to(device)
        mask = report_code['attention_mask'][:, 0, :].to('cuda')
        input_id = report_code['input_ids'][:, 0, :].to('cuda')
        pred = model(inputs2.float(), input_id, mask, clin.float(), prompt)[0]
        y_batch = labels
        if i == 0:
            pred_all = pred
            y_all = y_batch
            i = 1
        else:
            pred_all = torch.cat([pred_all, pred])
            y_all = torch.cat([y_all, y_batch])

    loss_dict = cal_loss(y_all, pred_all, criterion)
    ci_dict = cal_ci(y_all, pred_all)

    return loss_dict, ci_dict, pred_all


def _calculate_risk(h):
    r"""
    Take the logits of the model and calculate the risk for the patient

    Args:
        - h : torch.Tensor

    Returns:
        - risk : torch.Tensor

    """
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1)
    return risk, survival


def prediction_nll(model, dl, criterion):
    i = 0
    for data in dl:
        identifier, labels, inputs1, inputs2, clin, prompt, report_code = data[0], data[1].to(device), data[
            2].to(device), \
            data[
                3].to(
                device), data[4].to(device), data[5].to(device), data[6].to(device)
        mask = report_code['attention_mask'][:, 0, :].to('cuda')
        input_id = report_code['input_ids'][:, 0, :].to('cuda')
        pred = model(inputs2.float(), input_id, mask, clin.float(), prompt)[0]
        y_batch = labels
        if i == 0:
            pred_all = pred
            y_all = y_batch
            i = 1
        else:
            pred_all = torch.cat([pred_all, pred])
            y_all = torch.cat([y_all, y_batch])
    risk, _ = _calculate_risk(pred_all)
    y_all[:, 1] = 1 - y_all[:, 1]
    ci_dict = cal_ci(y_all, risk)

    return 0, ci_dict, pred_all


# TODO: document better and clean up
def nll_loss(h, y, c, alpha=0.5, eps=1e-7, reduction='sum'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
        The weight on uncensored loss
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    # print("h shape", h.shape)

    # make sure these are ints
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h)  # hazard function
    S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)

    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y + 1).clamp(min=eps)

    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss


def draw_risk(data, pictureName, cureName='Kaplan-Meier Survival Curve'):
    data.to_excel(pictureName + '.xlsx', index=False)
    data_copy = copy.deepcopy(data)
    data_copy['group'] = data_copy['group'].astype('category').cat.codes
    cph = CoxPHFitter()
    cph.fit(data_copy, duration_col='duration', event_col='event', formula="group")
    cph.print_summary()
    summary_df = cph.summary
    p_value_cox = summary_df.loc['group', 'p']
    print("Cox model p-value for 'group':", p_value_cox)
    mask1 = data['group'] == 'Group 1'
    time_to_event_group1 = data['duration'][mask1]
    event_status_group1 = data['event'][mask1]
    mask2 = data['group'] == 'Group 2'
    time_to_event_group2 = data['duration'][mask2]
    event_status_group2 = data['event'][mask2]
    result = logrank_test(time_to_event_group1, time_to_event_group2, event_observed_A=event_status_group1,
                          event_observed_B=event_status_group2)
    p_value = result.p_value
    print(f"-------------------------P-value between Group 1 and Group 2: {p_value:.8f}")

    exp_coef = summary_df.loc['group', 'exp(coef)']
    exp_coef_lower_95 = summary_df.loc['group', 'exp(coef) lower 95%']
    exp_coef_upper_95 = summary_df.loc['group', 'exp(coef) upper 95%']

    exp_coef_rounded = round(exp_coef, 2)
    exp_coef_lower_95_rounded = round(exp_coef_lower_95, 2)
    exp_coef_upper_95_rounded = round(exp_coef_upper_95, 2)

    if p_value_cox < 0.0001:
        label_ = f"High risk: HR = {exp_coef_rounded} (95% CI {exp_coef_lower_95_rounded}-{exp_coef_upper_95_rounded}), p<0.0001"
    else:
        label_ = f"High risk: HR = {exp_coef_rounded} (95% CI {exp_coef_lower_95_rounded}-{exp_coef_upper_95_rounded}), p={p_value_cox:.4f}"
    if p_value < 0.0001:
        label_logrank = "Log-rank test p<0.0001"
    else:
        label_logrank = "Log-rank test p={0:.4f}".format(p_value)

    specific_times = [0, 20, 40, 60, 80, 100, 120]
    labels = {}
    fig = plt.figure(constrained_layout=True, figsize=(9, 7))
    gs = GridSpec(2, 1, height_ratios=[4, 1], figure=fig)

    ax_kmf = fig.add_subplot(gs[0])
    ax_counts = fig.add_subplot(gs[1])

    groups = data['group'].unique()
    colors = ['#77eadc', '#bf3c3c']
    kmf_dict = {}
    for group, color in zip(groups, colors):
        kmf = KaplanMeierFitter()
        mask = data['group'] == group
        if group == 'Group 1':
            label = 'Low risk: reference'
        else:
            label = label_
        kmf.fit(data['duration'][mask], data['event'][mask], label=label)
        kmf.plot(ax=ax_kmf, ci_show=False, color=color, linewidth=4)
        kmf_dict[group] = kmf
    ax_kmf.plot([0, 0.1], [1, 1], color='white', label=label_logrank, linewidth=4)
    for group, color in zip(groups, colors):
        kmf = kmf_dict[group]
        event_table = kmf.event_table
        for time in specific_times:
            closest_time = event_table.index[event_table.index >= time].min()
            if not np.isnan(closest_time):
                at_risk = event_table.at[closest_time, 'at_risk']
                censored = event_table.loc[:closest_time, 'censored'].sum()
            else:
                closest_time = event_table.index.max()
                at_risk = 0
                censored = event_table.loc[:closest_time, 'censored'].sum()
            ax_counts.text(time, 0.6 if group == 'Group 1' else 0.3,
                           f'{int(at_risk)} ({int(censored)})',
                           ha='center', va='center', fontsize=16, color='black')
    ax_counts.add_patch(
        plt.Rectangle((- 10, 0.6), 4, 0.01, color='#77eadc', transform=ax_counts.transData, clip_on=False))
    ax_counts.add_patch(
        plt.Rectangle((- 10, 0.3), 4, 0.01, color='#bf3c3c', transform=ax_counts.transData, clip_on=False))
    ax_counts.text(- 10, 0.9, "Number at risk (number censored)", ha="left", va="center", color='black',
                   fontweight='bold', fontsize=18)

    ax_kmf.set_ylim([0.0, 1.1])
    ax_kmf.set_xlim([0.0, 120])
    ax_kmf.set_title(cureName, fontsize=24)
    ax_kmf.set_xlabel('Months', fontsize=18)
    ax_kmf.set_ylabel('Survival Probability', fontsize=18)
    ax_kmf.legend(loc='lower left', fontsize=18)
    ax_counts.set_xlim([0.0, 120])
    ax_counts.set_ylim([0, 1])
    ax_counts.axis('off')
    # plt.savefig(pictureName + '.png', dpi=300)
    plt.savefig(pictureName + '.pdf', format='pdf')