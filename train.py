import argparse
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import Train_Dataset, Val_Dataset, test_Dataset
from utils import nll_loss, prediction_nll, cox_loss, prepare_patient_data
from models import MOPS
from config import PATIENT_JSON_PATH, MODEL_PATH


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for MOPS model')
    parser.add_argument('--name', type=str, required=True, help='Name parameter')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--lr_image', type=float, required=True, help='Learning rate for image branch')
    parser.add_argument('--lr_report', type=float, required=True, help='Learning rate for report branch')
    parser.add_argument('--lr_clin', type=float, required=True, help='Learning rate for clinical branch')
    parser.add_argument('--lr_fused', type=float, required=True, help='Learning rate for classifier')
    parser.add_argument('--alph', type=float, required=True, help='Weight for similarity loss')
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    # numpy seed if needed
    # np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(batch_size: int, train_val_split, data_dict, test_data):
    train_ds = Train_Dataset(transform=None, fold=0, train_val_split=train_val_split, data_dict=data_dict)
    val_ds = Val_Dataset(fold=0, train_val_split=train_val_split, data_dict=data_dict)
    test_ds = test_Dataset(test_data=test_data, data_dict=data_dict)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False)
    return train_loader, val_loader, test_loader


def build_model_and_optimizer(device, lr_image, lr_report, lr_clin, lr_fused):
    model = MOPS().to(device)
    optimizer = optim.AdamW([
        {'params': model.image_branch.parameters(), 'lr': lr_image},
        {'params': model.report_branch.RadioLOGIC.parameters(), 'lr': 1e-6},
        {'params': model.report_branch.fc1.parameters(), 'lr': lr_report},
        {'params': model.report_branch.fc2.parameters(), 'lr': lr_report},
        {'params': model.clin_branch.parameters(), 'lr': lr_clin},
        {'params': model.classifier.parameters(), 'lr': lr_fused}
    ], weight_decay=0.01)
    return model, optimizer


def train_one_epoch(model, loader, optimizer, device, alph):
    model.train()
    for _, (identifier, labels, _ , inputs2, clin, prompt, report_code) in enumerate(tqdm(loader, desc='Training', leave=False)):
        labels = labels.to(device)
        inputs2 = inputs2.to(device)
        clin = clin.to(device)
        prompt = prompt.to(device)

        mask = report_code['attention_mask'][:, 0, :].to(device)
        input_id = report_code['input_ids'][:, 0, :].to(device)

        optimizer.zero_grad()
        outputs, similarity = model(inputs2.float(), input_id, mask, clin.float(), prompt)
        loss_pred = nll_loss(outputs, labels[:, 0].unsqueeze(1), labels[:, 1].unsqueeze(1))
        loss_sim = similarity * alph
        loss = loss_pred + loss_sim
        loss.backward()
        optimizer.step()


def evaluate(model, loader, criterion, device):
    model.eval()
    with torch.no_grad():
        loss_dict, ci_dict, score = prediction_nll(model, loader, criterion)
    return ci_dict, score


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_data, test_data, train_val_split, data_dict = \
        prepare_patient_data(PATIENT_JSON_PATH, test_ratio=0.5, k=5)

    print(f"Learning rates: image={args.lr_image}, report={args.lr_report}, "
          f"clin={args.lr_clin}, fused={args.lr_fused}, alpha={args.alph}")

    train_loader, val_loader, test_loader = build_dataloaders(batch_size=8,
                                                              train_val_split=train_val_split,
                                                              data_dict=data_dict,
                                                              test_data=test_data)
    model, optimizer = build_model_and_optimizer(device,
                                                 args.lr_image,
                                                 args.lr_report,
                                                 args.lr_clin,
                                                 args.lr_fused)

    best_val_ci = 0.0
    for epoch in range(1, 11):
        start = time.time()
        train_one_epoch(model, train_loader, optimizer, device, args.alph)

        train_ci, _ = evaluate(model, train_loader, cox_loss, device)
        val_ci, _ = evaluate(model, val_loader, cox_loss, device)
        total_val_ci = sum(val_ci.values()).cpu().item()

        print(f"Epoch {epoch}: train C-index={sum(train_ci.values()).cpu().item():.4f}, "
              f"val C-index={total_val_ci:.4f}, time={(time.time()-start):.1f}s")

        if total_val_ci > best_val_ci:
            best_val_ci = total_val_ci
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"New best model saved with C-index: {best_val_ci:.4f}")

    print('Finished Training')


if __name__ == '__main__':
    main()
