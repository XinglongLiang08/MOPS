# Original directories
ORIGINAL_CT_DIR = '/projects/nki-breast-mri/snapshot/a_PETCT/firstDiagnose/beforeSurgery_nnunet/imagesTs_resize'
ORIGINAL_PET_DIR = ORIGINAL_CT_DIR
ORIGINAL_SEG_DIR = '/projects/nki-breast-mri/snapshot/a_PETCT/firstDiagnose/beforeSurgery_nnunet/imagesTs_predlowres_stunet_resize'

# Replacement directories
REPLACEMENT_CT_DIR = '/processing/x.liang/imagesTs_split_npy'
REPLACEMENT_PET_DIR = '/processing/x.liang/imagesTs_split_npy'
REPLACEMENT_SEG_DIR = '/processing/x.liang/imagesTs_predlowres_nki_breast_stunet_split_npy'

# JSON dataset mapping file
DATASET_JSON_PATH = '/projects/whole_body_PET_CT_segmentation/organizePatientsFile/output/dataset_test_id_mask.json'
PATIENT_JSON_PATH = '/home/x.liang/MyProject/survivalAnalysis/github/saved_models/data_splits.json'
# Tokenizer directory
ROBERTA_TOKENIZER_PATH = '/home/x.liang/MyProject/survivalPrediction/radiobert_BigDataset_epoch10'

#model path
MODEL_PATH = '/home/x.liang/MyProject/survivalAnalysis/github/saved_models/best_model.pth'#'/home/x.liang/MyProject/survivalAnalysis/new_dataloader/OS/best_model-NonCom-ab_0733multi_clin_pro-0.pth'

#Result path
RESULT_PATH = '/home/x.liang/MyProject/survivalAnalysis/github/saved_models/survival/'
