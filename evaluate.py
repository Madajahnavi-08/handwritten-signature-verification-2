"""
Evaluation and Metrics Analysis Module for Siamese Signature Verification.
Computes ROC, AUC, EER (Equal Error Rate), FAR, FRR, and Confusion Matrix.
Author: Mada Jahnavi
Domain: Deep Learning & Computer Vision
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

from model import SiameseSignatureNet
from dataset import SignaturePairDataset


def evaluate_model(model_path: str = None, num_test_pairs: int = 500):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "data")
    reports_dir = os.path.join(project_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    if model_path is None:
        model_path = os.path.join(project_dir, "saved_models", "best_siamese_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}. Please run train.py first!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading Siamese model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model = SiameseSignatureNet(embedding_dim=128).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test dataset with unseen seed
    test_dataset = SignaturePairDataset(data_dir=data_dir, num_pairs=num_test_pairs, is_train=False, seed=2026)
    
    distances = []
    ground_truth_labels = []

    print(f"[*] Running inference on {len(test_dataset)} test signature pairs...")
    with torch.no_grad():
        for i in range(len(test_dataset)):
            img1, img2, label = test_dataset[i]
            img1 = img1.unsqueeze(0).to(device)
            img2 = img2.unsqueeze(0).to(device)

            out1, out2 = model(img1, img2)
            dist = float(F.pairwise_distance(out1, out2).squeeze().cpu().item())
            
            distances.append(dist)
            ground_truth_labels.append(int(label.item()))

    distances = np.array(distances)
    y_true = np.array(ground_truth_labels)

    # Note: For Siamese distances, a lower distance means class 1 (Genuine).
    # To compute standard ROC where higher score = positive class:
    # similarity_scores = 1.0 - (distances / 2.0)
    scores = 1.0 - (distances / 2.0)
    scores = np.clip(scores, 0.0, 1.0)

    # 1. ROC & AUC
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    # 2. FAR, FRR, and EER computation across Euclidean Distance thresholds
    candidate_thresholds = np.linspace(0.05, 1.5, 300)
    far_list = []  # False Acceptance Rate: Forged accepted as Genuine (dist <= th)
    frr_list = []  # False Rejection Rate: Genuine rejected as Forged (dist > th)

    for th in candidate_thresholds:
        # Predictions based on distance: Genuine (1) if dist <= th, else Forged (0)
        y_pred = (distances <= th).astype(int)
        
        # Genuine pairs are y_true == 1
        genuine_mask = (y_true == 1)
        # Forged pairs are y_true == 0
        forged_mask = (y_true == 0)

        # FRR = (Genuine predicted as 0) / Total Genuine
        frr = np.sum((y_pred == 0) & genuine_mask) / np.sum(genuine_mask)
        # FAR = (Forged predicted as 1) / Total Forged
        far = np.sum((y_pred == 1) & forged_mask) / np.sum(forged_mask)

        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)

    # EER is point where FAR ~= FRR
    diff = np.abs(far_arr - frr_arr)
    eer_idx = np.argmin(diff)
    optimal_threshold = float(candidate_thresholds[eer_idx])
    eer_value = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0)

    # Metrics at optimal threshold
    final_preds = (distances <= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, final_preds)
    tn, fp, fn, tp = cm.ravel()  # 0: Forged, 1: Genuine
    # tn: Forged correctly detected as Forged
    # fp: Forged incorrectly accepted as Genuine (False Acceptance)
    # fn: Genuine incorrectly rejected (False Rejection)
    # tp: Genuine correctly verified (True Genuine)

    accuracy = float((tp + tn) / len(y_true)) * 100.0
    precision = float(tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("==================================================================")
    print("                MODEL EVALUATION RESULTS")
    print("==================================================================")
    print(f"  Test Samples Evaluated   : {len(y_true)} pairs")
    print(f"  Area Under Curve (AUC)   : {roc_auc:.4f}")
    print(f"  Optimal Distance Thresh  : {optimal_threshold:.4f}")
    print(f"  Equal Error Rate (EER)   : {eer_value * 100.0:.2f}%")
    print(f"  Overall Accuracy         : {accuracy:.2f}%")
    print(f"  Precision                : {precision:.2f}%")
    print(f"  Recall (Sensitivity)     : {recall:.2f}%")
    print(f"  F1-Score                 : {f1:.2f}%")
    print(f"  Confusion Matrix         :")
    print(f"    - True Genuine Verified  : {tp}")
    print(f"    - False Genuine Rejected : {fn} (FRR: {fn/(tp+fn)*100:.1f}%)")
    print(f"    - True Forgery Detected  : {tn}")
    print(f"    - False Forgery Accepted : {fp} (FAR: {fp/(tn+fp)*100:.1f}%)")
    print("==================================================================")

    # 1. Plot ROC & EER Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ROC
    ax1.plot(fpr, tpr, color='#2563eb', lw=2.5, label=f'Siamese ROC (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='#9ca3af', lw=1.5, linestyle='--', label='Random Chance')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11)
    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=11)
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=':', alpha=0.6)

    # FAR vs FRR / EER
    ax2.plot(candidate_thresholds, far_arr * 100.0, color='#dc2626', lw=2, label='FAR (False Acceptance)')
    ax2.plot(candidate_thresholds, frr_arr * 100.0, color='#16a34a', lw=2, label='FRR (False Rejection)')
    ax2.plot(optimal_threshold, eer_value * 100.0, 'ko', markersize=8, label=f'EER = {eer_value*100:.1f}% @ {optimal_threshold:.2f}')
    ax2.axvline(x=optimal_threshold, color='gray', linestyle=':')
    ax2.set_xlabel('Decision Distance Threshold (τ)', fontsize=11)
    ax2.set_ylabel('Error Rate (%)', fontsize=11)
    ax2.set_title('FAR vs FRR Curve & Equal Error Rate (EER)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    roc_plot_path = os.path.join(reports_dir, "roc_eer_curves.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()

    # 2. Plot Distance Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    gen_dists = distances[y_true == 1]
    forg_dists = distances[y_true == 0]
    
    ax.hist(gen_dists, bins=25, alpha=0.65, color='#16a34a', label=f'Genuine Pairs (Mean: {np.mean(gen_dists):.2f})', edgecolor='black')
    ax.hist(forg_dists, bins=25, alpha=0.65, color='#dc2626', label=f'Forged Pairs (Mean: {np.mean(forg_dists):.2f})', edgecolor='black')
    ax.axvline(x=optimal_threshold, color='#2563eb', linestyle='--', lw=2.5, label=f'Optimal Threshold ({optimal_threshold:.2f})')
    ax.set_title('Euclidean Distance Distribution in Siamese Feature Space', fontsize=12, fontweight='bold')
    ax.set_xlabel('Euclidean Distance ||f(x1) - f(x2)||')
    ax.set_ylabel('Pair Frequency')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    hist_plot_path = os.path.join(reports_dir, "distance_distribution.png")
    plt.savefig(hist_plot_path, dpi=300)
    plt.close()

    # Save summary report JSON
    eval_results = {
        "num_test_pairs": len(y_true),
        "roc_auc": round(float(roc_auc), 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "eer_pct": round(eer_value * 100.0, 2),
        "accuracy_pct": round(accuracy, 2),
        "precision_pct": round(precision, 2),
        "recall_pct": round(recall, 2),
        "f1_score_pct": round(f1, 2),
        "confusion_matrix": {
            "true_genuine": int(tp),
            "false_rejection": int(fn),
            "true_forged": int(tn),
            "false_acceptance": int(fp)
        },
        "mean_distance_genuine": round(float(np.mean(gen_dists)), 4),
        "mean_distance_forged": round(float(np.mean(forg_dists)), 4),
        "plots": {
            "roc_eer": roc_plot_path,
            "distribution": hist_plot_path
        }
    }

    report_json_path = os.path.join(reports_dir, "evaluation_report.json")
    with open(report_json_path, "w") as f:
        json.dump(eval_results, f, indent=4)

    print(f"Saved evaluation metrics to: {report_json_path}")
    print(f"Saved plots to: {reports_dir}")
    return eval_results


if __name__ == "__main__":
    evaluate_model()
