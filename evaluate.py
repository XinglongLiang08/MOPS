import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import argparse

from dataloader import Train_Dataset, Val_Dataset, test_Dataset
from utils import (
    bootstrap_c_index, cal_ci, _calculate_risk,
    draw_risk, prepare_patient_data
)
from models import MOPS
from config import PATIENT_JSON_PATH, MODEL_PATH, RESULT_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(device):
    model = MOPS().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def evaluate_dataloader(model, dataloader, data_dict=None, collect_roc=False):
    """Run the model on a DataLoader, return risks, identifiers, and optionally ROC data."""
    risks = []
    identifiers = []
    roc_entries = []
    all_preds = None
    all_labels = None

    with torch.no_grad():
        for batch in dataloader:
            ids, labels, _, inputs2, clin, prompt, report_code = batch
            ids = list(ids)
            inputs2 = inputs2.to(device)
            clin = clin.to(device)

            mask = report_code['attention_mask'][:, 0, :].to(device)
            input_id = report_code['input_ids'][:, 0, :].to(device)

            pred_raw = model(inputs2.float(), input_id, mask, clin.float(), prompt)[0]
            batch_risk, batch_score = _calculate_risk(pred_raw)

            # collect roc entries if requested (test set only)
            if collect_roc:
                for (tf, label), raw_score in zip(labels, pred_raw):
                    tf_val = tf.cpu().item()
                    # three-year label
                    y3 = 0 if tf_val > 36 else (1 if tf_val <= 36 and label == 0 else -1)
                    # five-year label
                    y5 = 0 if tf_val > 60 else (1 if tf_val <= 60 and label == 0 else -1)
                    # ten-year label
                    y10 = 0 if tf_val > 120 else (1 if tf_val <= 120 and label == 0 else -1)
                    roc_entries.append((y3, y5, y10, raw_score))

            # concatenate overall predictions
            if all_preds is None:
                all_preds = batch_risk
                all_labels = labels
            else:
                all_preds = torch.cat([all_preds, batch_risk])
                all_labels = torch.cat([all_labels, labels])

            # collect identifiers and risk values
            identifiers.extend(ids)
            risks.extend(batch_risk.tolist())

    return identifiers, risks, roc_entries, all_labels, all_preds


def save_roc_dataframe(roc_entries):
    df = pd.DataFrame({
        'y_3': [e[0] for e in roc_entries],
        'y_5': [e[1] for e in roc_entries],
        'y_10': [e[2] for e in roc_entries],
        'roc_scores': [e[3].cpu().numpy()[0] for e in roc_entries]
    })
    df.to_excel(RESULT_PATH + "all_ROC.xlsx", index=False)


def split_by_median(reference_risks, split_ids, split_risks):
    median_score = pd.Series(reference_risks).median()
    low  = [i for i, r in zip(split_ids, split_risks) if r <  median_score]
    high = [i for i, r in zip(split_ids, split_risks) if r >= median_score]
    return low, high, median_score


def build_dataframe(groups, data_dict):
    """
    Build data frames for group comparison and for all-subgroup plots.
    groups: dict of name->[ids]
    """
    records_all = []
    for group_name, ids in groups.items():
        for pid in ids:
            dd = data_dict[pid]
            records_all.append({
                'ID': pid,
                'duration': dd['time'],
                'event': 1 if dd['label'] == 0 else 0,
                'primary': 1 if dd['primary_treatment'] == 'neo_adjuvante' else 0,
                'age': 1 if dd['AGE'] > 50 else 0,
                'T_stage': 1 if dd['T_stage'] and any(x in dd['T_stage'] for x in ['1', 'is', 'IS']) else 0,
                'T_stage_2': 1 if not (dd['T_stage'] and any(x in dd['T_stage'] for x in ['1', 'is', 'IS'])) else 0,
                'EPH_surv1': 1 if (dd['EPH_surv'][0] > 0 and dd['EPH_surv'][2] < 2) else 0,
                'EPH_surv2': 1 if (dd['EPH_surv'][0] == 0 and dd['EPH_surv'][2] < 2) else 0,
                'EPH_surv3': 1 if dd['EPH_surv'][2] > 2 else 0,
                'N_stage1': 1 if dd['N_stage'] and '0' in dd['N_stage'] else 0,
                'N_stage2': 1 if dd['N_stage'] and '0' not in dd['N_stage'] else 0,
                'group': group_name
            })

    df_all = pd.DataFrame.from_records(records_all)
    # simplified df for overall plot
    df = df_all[['ID', 'duration', 'event', 'group']]
    return df_all, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pat_json', default=PATIENT_JSON_PATH)
    parser.add_argument('--model_path', default=MODEL_PATH)
    parser.add_argument('--result_path', default=RESULT_PATH)
    parser.add_argument('--test_ratio', type=float, default=0.5)
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = prepare_patient_data(args.pat_json, test_ratio=args.test_ratio, k=5)
    train_data, test_data, train_val_split, data_dict = data

    model = load_model(device)

    # Evaluate on train for risk distribution
    val_dataset = Val_Dataset(transform=None, fold=args.fold,
                              onlyReport=False, train_val_split=train_val_split,
                              data_dict=data_dict)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=16, shuffle=False)
    train_ids, train_risks, _, _, _ = evaluate_dataloader(model, val_loader)

    # Evaluate on test for ROC and overall CI
    test_dataset = test_Dataset(test_data=test_data, data_dict=data_dict)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=16, shuffle=False)
    test_ids, test_risks, roc_entries, y_all, pred_all = evaluate_dataloader(
        model, test_loader, data_dict, collect_roc=True
    )

    # Save ROC data
    save_roc_dataframe(roc_entries)

    # Compute CI
    y_all[:, 1] = 1 - y_all[:, 1]
    ci = cal_ci(y_all, pred_all)
    print(bootstrap_c_index(y_all, pred_all))
    print(ci)

    # Split groups by median
    low_ids, high_ids, median_score = split_by_median(
        train_risks,
        test_ids,
        test_risks  
    )
    print(f"Test-set median risk = {median_score:.4f}, low/hi on val-set = {len(low_ids)}/{len(high_ids)}")
    print(f"Low-risk N={len(low_ids)}, High-risk N={len(high_ids)}")

    groups = {'Group 1': low_ids, 'Group 2': high_ids}
    df_all, df = build_dataframe(groups, data_dict)

    # Draw risk plots
    draw_risk(df, RESULT_PATH + 'all', 'MOPS')
    # Subgroup Kaplan-Meier
    for col, title, suffix in [
        ('primary', 'Neoadjuvant', 'primary_neo_adjuvante'),
        ('T_stage_2', 'T2+', 'T2+'),
        ('T_stage', 'T1', 'T1'),
        ('age', 'Age>50', 'age_1'),
        ('age', 'Age<50', 'age_0'),
        ('EPH_surv1', 'Luminal', 'luminal'),
        ('EPH_surv2', 'Triple-negative', 'TN'),
        ('EPH_surv3', 'Her2+', 'Her2+'),
        ('N_stage1', 'N0', 'N0'),
        ('N_stage2', 'N1+', 'N+')
    ]:
        subset = df_all[df_all[col] == (1 if suffix in ['primary_neo_adjuvante', 'T2+', 'T1', 'age_1', 'luminal', 'TN', 'Her2+', 'N0', 'N+'] else 0)]
        draw_risk(subset, RESULT_PATH + f'{suffix}-Kaplan-Meier_report', title)


if __name__ == '__main__':
    main()
