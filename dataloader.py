from torch.utils.data import DataLoader
import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset as dataset
import nibabel as nib
import json
import torch.nn.functional as F
from pathlib import Path
from scipy.ndimage import binary_dilation
from scipy.ndimage import label as label_components
from transformers import BertTokenizer, BertModel, BertConfig, RobertaTokenizer, RobertaModel, \
    RobertaForTokenClassification, RobertaTokenizerFast
from config import (
    ORIGINAL_CT_DIR, ORIGINAL_PET_DIR, ORIGINAL_SEG_DIR,
    REPLACEMENT_CT_DIR, REPLACEMENT_PET_DIR, REPLACEMENT_SEG_DIR,
    DATASET_JSON_PATH, ROBERTA_TOKENIZER_PATH)

with open(DATASET_JSON_PATH, 'r') as f:
    dataSet = json.load(f)['training']
data_dict = {entry["ID_PI"]: entry for entry in dataSet}

tokenizer = RobertaTokenizerFast.from_pretrained(
    ROBERTA_TOKENIZER_PATH, return_tensors='pt')


def dilate_3d(mask, structure=None, iterations=5):
    if structure is None:
        structure = np.ones((3, 3, 3))
    dilated_mask = binary_dilation(mask, structure=structure, iterations=iterations)
    return dilated_mask


def generate_one_hot(T_stage_value, N_stage_value, M_stage_value, T_stage_post_value, N_stage_post_value,
                     M_stage_post_value, family_history, age, E, P, H, tumor_types):
    one_hot_matrix = np.zeros((25,
                               10))  # "T_stage","N_stage","M_stage","T_stage_post","N_stage_post","M_stage_post","family_history","age","E",“P"，”H","tumor_types"
    T_stage = ['0', '1A', '1B', '1C', '2', '3', '4', '4A', '4B', '4D', 'IS', 'None', 'X']
    N_stage = ['0', '0IS', '0S', '1', '1MS', '1S', '2A', '2B', '3A', '3B', '3BS', '3C', 'None', 'X']
    M_stage = ['0', '1', 'None', 'X']
    T_stage_post = ['1', '1A', '1B', '1C', '1MI', '2', '3', '4B', 'IS', 'None', 'X', 'Y0', 'Y1', 'Y1A', 'Y1B', 'Y1C',
                    'Y1MI',
                    'Y2', 'Y3', 'Y4A', 'Y4B', 'Y4D', 'YIS', 'YX']
    N_stage_post = ['0', '0I', '0IS', '0S', '1A', '1AS', '1B1', '1B2', '1B3', '1B4', '1M', '1MI', '1MS', '2A', '3A',
                    '3B', '3C', 'None', 'X']
    M_stage_post = ['-', '0', '1', 'None']
    try:
        index = T_stage.index(T_stage_value.strip())
        one_hot_matrix[index, 0] = 1
    except ValueError:
        print('t')
        print(T_stage_value.strip() + " is not in the list")

    try:
        index = N_stage.index(N_stage_value.strip())
        one_hot_matrix[index, 1] = 1
    except ValueError:
        print('n')
        print(N_stage_value.strip() + " is not in the list")
    try:
        index = T_stage_post.index(T_stage_post_value.strip())
        one_hot_matrix[index, 2] = 1
    except ValueError:
        print('tp')
        print(T_stage_post_value.strip() + " is not in the list")

    try:
        index = N_stage_post.index(N_stage_post_value.strip())
        one_hot_matrix[index, 3] = 1
    except ValueError:
        print('np')
        print(N_stage_post_value.strip() + " is not in the list")

    if 'kanker' in family_history:
        one_hot_matrix[0, 4] = 1
    one_hot_matrix[int(age / 4), 5] = 1
    one_hot_matrix[1 if int(E) > 0 and int(E / 4) < 25 else 0, 6] = 1
    one_hot_matrix[1 if int(P) > 0 and int(P / 4) < 25 else 0, 7] = 1
    one_hot_matrix[1 if int(H) > 2 and int(H / 4) < 25 else 0, 8] = 1
    # tumer type
    if tumor_types != None:
        tumor_types = tumor_types.lower()
        if 'ductaal' in tumor_types and 'infiltrerend ductaal' not in tumor_types and 'intraductaal carcinoom' not in tumor_types and 'ductaal carcinoma in situ' not in tumor_types:
            one_hot_matrix[0, 9] = 1
        if 'infiltrerend ductaal' in tumor_types and 'intraductaal carcinoom' not in tumor_types:
            one_hot_matrix[1, 9] = 1
        if 'lobulair' in tumor_types and 'infiltrerend lobulair' not in tumor_types:
            one_hot_matrix[2, 9] = 1
        if 'infiltrerend lobulair' in tumor_types:
            one_hot_matrix[3, 9] = 1
        if 'tubular' in tumor_types:
            one_hot_matrix[4, 9] = 1
        if 'mucineus' in tumor_types:
            one_hot_matrix[5, 9] = 1
        if 'micropapillair' in tumor_types:
            one_hot_matrix[6, 9] = 1
        if 'papillair' in tumor_types and 'micropapillair' not in tumor_types and 'intraductaal papillair adenocarcinoom' not in tumor_types:
            one_hot_matrix[7, 9] = 1
        if 'ductaal carcinoma in situ' in tumor_types or 'intraductaal carcinoom' in tumor_types or 'intraductaal papillair adenocarcinoom' in tumor_types:
            one_hot_matrix[8, 9] = 1
        if sum(one_hot_matrix[:, 9]) == 0:
            one_hot_matrix[9, 9] = 1

    return one_hot_matrix


def generate_tensors():
    a = torch.randint(150, 300, (1,)).item()
    tensor1 = torch.zeros(512, dtype=torch.int64)
    tensor1[:a] = 1
    tensor_2 = torch.ones(512, dtype=torch.int64)
    tensor_2[:a] = torch.randint(2, 30521, (a,), dtype=torch.int64)
    tensor_2[0] = 0
    tensor_2[a - 1] = 2
    return tensor1.unsqueeze(0), tensor_2.unsqueeze(0)


# Define the original and replacement directories
original_ct_dir = ORIGINAL_CT_DIR
replacement_ct_dir = REPLACEMENT_CT_DIR

original_pet_dir = ORIGINAL_PET_DIR
replacement_pet_dir = REPLACEMENT_PET_DIR

original_seg_dir = ORIGINAL_SEG_DIR
replacement_seg_dir = REPLACEMENT_SEG_DIR

class Train_Dataset(dataset):
    def __init__(self, transform=None, fold=0, onlyReport=False, train_val_split=[], data_dict=[]):
        self.dataSet, _ = train_val_split[fold]
        self.transform = transform
        self.onlyReport = onlyReport
        self.data_dict = data_dict

    def __getitem__(self, idx):

        identifier = self.dataSet[idx]
        report = torch.FloatTensor(torch.zeros(768))
        report_clean = str(self.data_dict[identifier]['reports_surv']).replace("\n", "")
        index = report_clean.find('Verslag')
        if index != -1:
            report_clean = report_clean[index:]
        report_code = tokenizer(report_clean, padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt")
        if self.onlyReport == False:

            ct = np.load(self.data_dict[identifier]["ct_image_names"].replace(original_ct_dir, replacement_ct_dir))
            ct = ct[np.newaxis, ...]

            pet = np.load(self.data_dict[identifier]["pet_image_names"].replace(original_pet_dir, replacement_pet_dir))
            pet = pet[np.newaxis, ...]

            seg = np.load(self.data_dict[identifier]["seg_image_names"].replace(original_seg_dir, replacement_seg_dir))

            expanded_mask = dilate_3d(seg)
            seg = expanded_mask
            seg = seg[np.newaxis, ...]
            combined_array = np.concatenate((pet, ct, seg), axis=0)
            if self.transform:
                combined_array = self.transform(combined_array)
        else:
            combined_array = torch.tensor([0])
        if self.data_dict[identifier]["label"] == 0:
            temp = 0
        else:
            temp = 1
        if self.data_dict[identifier]["time"] <= 28.23:
            y = 0
        elif self.data_dict[identifier]["time"] > 28.33 and self.data_dict[identifier]["time"] <= 49.6:
            y = 1
        elif self.data_dict[identifier]["time"] > 49.6 and self.data_dict[identifier]["time"] <= 72.06:
            y = 2
        else:
            y = 3
        label = torch.tensor([y, temp], dtype=torch.int64)

        item = self.data_dict[identifier]
        clinical_info = torch.tensor(generate_one_hot(str(item['T_stage']), str(item['N_stage']), str(item['M_stage']),
                                                      str(item['T_stage_post']),
                                                      str(item['N_stage_post']), str(item['M_stage_post']),
                                                      str(item['family_history']),
                                                      item['AGE'], item['EPH_surv'][0], item['EPH_surv'][1],
                                                      item['EPH_surv'][2], str(item['tumor_types'])))
        if str(item['primary_treatment']) == 'neo_adjuvante':  # "primary_treatment": "neo_adjuvante"
            primary_treatment = torch.tensor(1)
        else:
            primary_treatment = torch.tensor(0)
        return identifier, label, report, combined_array, clinical_info, primary_treatment, report_code

    def __len__(self):
        return len(self.dataSet)


class Val_Dataset(dataset):
    def __init__(self, transform=None, fold=0, onlyReport=False, train_val_split=[], data_dict=[]):
        _, self.dataSet = train_val_split[fold]
        self.transform = transform
        self.onlyReport = onlyReport

        self.data_dict = data_dict

    def __getitem__(self, idx):

        identifier = self.dataSet[idx]
        report = torch.FloatTensor(torch.zeros(768))
        report_clean = str(self.data_dict[identifier]['reports_surv']).replace("\n", "")
        index = report_clean.find('Verslag')
        if index != -1:
            report_clean = report_clean[index:]
        report_code = tokenizer(report_clean, padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt")

        if self.onlyReport == False:
            ct = np.load(self.data_dict[identifier]["ct_image_names"].replace(original_ct_dir, replacement_ct_dir))
            ct = ct[np.newaxis, ...]

            pet = np.load(self.data_dict[identifier]["pet_image_names"].replace(original_pet_dir, replacement_pet_dir))
            pet = pet[np.newaxis, ...]

            seg = np.load(self.data_dict[identifier]["seg_image_names"].replace(original_seg_dir, replacement_seg_dir))

            expanded_mask = dilate_3d(seg)
            seg = expanded_mask
            seg = seg[np.newaxis, ...]
            combined_array = np.concatenate((pet, ct, seg), axis=0)
            if self.transform:
                combined_array = self.transform(combined_array)
        else:
            combined_array = torch.tensor([0])
        if self.data_dict[identifier]["label"] == 0:
            temp = 0
        else:
            temp = 1
        if self.data_dict[identifier]["time"] <= 28.23:
            y = 0
        elif self.data_dict[identifier]["time"] > 28.33 and self.data_dict[identifier]["time"] <= 49.6:
            y = 1
        elif self.data_dict[identifier]["time"] > 49.6 and self.data_dict[identifier]["time"] <= 72.06:
            y = 2
        else:
            y = 3
        label = torch.tensor([y, temp], dtype=torch.int64)

        item = self.data_dict[identifier]
        clinical_info = torch.tensor(generate_one_hot(str(item['T_stage']), str(item['N_stage']), str(item['M_stage']),
                                                      str(item['T_stage_post']),
                                                      str(item['N_stage_post']), str(item['M_stage_post']),
                                                      str(item['family_history']),
                                                      item['AGE'], item['EPH_surv'][0], item['EPH_surv'][1],
                                                      item['EPH_surv'][2], str(item['tumor_types'])))
        if str(item['primary_treatment']) == 'neo_adjuvante':  # "primary_treatment": "neo_adjuvante"
            primary_treatment = torch.tensor(1)
        else:
            primary_treatment = torch.tensor(0)
        return identifier, label, report, combined_array, clinical_info, primary_treatment, report_code

    def __len__(self):
        return len(self.dataSet)


class test_Dataset(dataset):
    def __init__(self, transform=None, fold=0, onlyReport=False, test_data=[], data_dict=[]):
        self.dataSet = test_data
        self.transform = transform
        self.onlyReport = onlyReport

        self.data_dict = data_dict

    def __getitem__(self, idx):
        identifier = self.dataSet[idx]
        report = torch.FloatTensor(torch.zeros(768))
        report_clean = str(self.data_dict[identifier]['reports_surv']).replace("\n", "")
        index = report_clean.find('Verslag')
        if index != -1:
            report_clean = report_clean[index:]
        report_code = tokenizer(report_clean, padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt")
        if self.onlyReport == False:
            ct = np.load(self.data_dict[identifier]["ct_image_names"].replace(original_ct_dir, replacement_ct_dir))
            ct = ct[np.newaxis, ...]

            pet = np.load(self.data_dict[identifier]["pet_image_names"].replace(original_pet_dir, replacement_pet_dir))
            pet = pet[np.newaxis, ...]

            seg = np.load(self.data_dict[identifier]["seg_image_names"].replace(original_seg_dir, replacement_seg_dir))
            expanded_mask = dilate_3d(seg)
            seg = expanded_mask
            seg = seg[np.newaxis, ...]
            combined_array = np.concatenate((pet, ct, seg), axis=0)
            if self.transform:
                combined_array = self.transform(combined_array)
        else:
            combined_array = torch.tensor([0])
        if self.data_dict[identifier]["label"] == 0:
            temp = 0
        else:
            temp = 1
        if self.data_dict[identifier]["time"] <= 28.23:
            y = 0
        elif self.data_dict[identifier]["time"] > 28.33 and self.data_dict[identifier]["time"] <= 49.6:
            y = 1
        elif self.data_dict[identifier]["time"] > 49.6 and self.data_dict[identifier]["time"] <= 72.06:
            y = 2
        else:
            y = 3
        label = torch.tensor([y, temp], dtype=torch.int64)
        #### AUC
        # label = torch.tensor([self.data_dict[identifier]["time"], temp], dtype=torch.int64)
        item = self.data_dict[identifier]
        clinical_info = torch.tensor(generate_one_hot(str(item['T_stage']), str(item['N_stage']), str(item['M_stage']),
                                                      str(item['T_stage_post']),
                                                      str(item['N_stage_post']), str(item['M_stage_post']),
                                                      str(item['family_history']),
                                                      item['AGE'], item['EPH_surv'][0], item['EPH_surv'][1],
                                                      item['EPH_surv'][2], str(item['tumor_types'])))
        if str(item['primary_treatment']) == 'neo_adjuvante':  # "primary_treatment": "neo_adjuvante"
            primary_treatment = torch.tensor(1)
        else:
            primary_treatment = torch.tensor(0)
        return identifier, label, report, combined_array, clinical_info, primary_treatment, report_code

    def __len__(self):
        return len(self.dataSet)