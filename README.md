# Handwritten Signature Verification using Siamese Neural Network
### With Bank Document-Verification Module
**Author:** Mada Jahnavi  
**Domain:** Deep Learning & Computer Vision  
**Degree Level:** B.Tech Mini Project  

---

## 📌 Project Overview
The proposed system introduces a Siamese neural network to verify genuine handwritten signatures against forged samples, addressing the challenging problem of offline handwritten signature verification through a data-driven deep metric learning approach. Scanned signature image pairs form the basis of the dataset, which is processed through a structured pipeline before model building and evaluation. Results are presented through a bank document-verification module, so that the insights generated are directly actionable.

---

## 🌟 Key Highlights
1. **Twin CNN Architecture:** Two weight-sharing branches project signature images into a 128-dimensional discriminative embedding space.
2. **Contrastive Loss Optimization:** Pulls genuine signature embeddings together and pushes forged signature embeddings beyond margin $m = 1.0$.
3. **Structured Image Preprocessing Pipeline:**
   - Grayscale conversion
   - Gaussian filtering & noise suppression
   - Otsu's adaptive binarization
   - Morphological stroke closing
   - Bounding box contour detection & aspect-ratio preserving canvas padding (100x100)
   - Intensity normalization to $[0, 1]$
4. **Bank Document-Verification Module (Web UI):**
   - **Bank Cheque Verification:** Automatic Region of Interest (ROI) extraction of the drawer signature from scanned bank cheques and transaction slips.
   - **Customer Specimen Database:** Registered customer accounts with specimen signatures.
   - **Explainable AI:** Spatial Stroke Divergence Heatmap highlighting pen tremors and curvature differences.
   - **Automated Audit Certificates:** Generates official downloadable PDF certificates with timestamps and examiner sign-offs.
5. **Comprehensive Evaluation Metrics:** ROC Curve, AUC, Equal Error Rate (EER), False Acceptance Rate (FAR), False Rejection Rate (FRR), and Confusion Matrix.

---

## 📁 Directory Structure
```
SignatureVerification_SiameseNet/
├── data/
│   ├── raw/
│   │   ├── genuine/               # Raw genuine signature images
│   │   └── forged/                # Raw forged signature images
│   ├── processed/                 # Standardized 100x100 inverted signatures
│   └── bank_cheques/              # Scanned Bank Cheques with Drawer Signatures
├── saved_models/
│   ├── best_siamese_model.pth     # Trained PyTorch Siamese weights
│   ├── training_metrics.json      # Training loss & accuracy records
│   └── training_curves.png        # Contrastive loss & distance separation plots
├── reports/
│   ├── evaluation_report.json     # ROC, AUC, EER, and confusion matrix metrics
│   ├── roc_eer_curves.png         # ROC & EER tradeoff curves
│   └── distance_distribution.png  # Genuine vs Forged distance histogram
├── static/
│   ├── css/style.css              # Bank corporate styling
│   └── js/app.js                  # Frontend UI logic
├── templates/
│   └── index.html                 # Bank Document Verification Portal
├── app.py                         # FastAPI Web Server Application
├── bank_verification_engine.py    # Inference, ROI extraction, and PDF engine
├── model.py                       # Siamese Architecture & Contrastive Loss
├── dataset.py                     # Balanced Pair Dataset & Dataloaders
├── preprocessing.py               # Structured Image Processing Pipeline
├── generate_dataset.py            # Signature & Bank Cheque Generator
├── train.py                       # Training pipeline
├── evaluate.py                    # Evaluation, ROC, EER analysis
├── run_app.bat                    # 1-Click Windows execution script
├── train_model.bat                # 1-Click Model training script
├── PROJECT_REPORT.md              # Complete B.Tech Academic Project Report
└── VIVA_VOCE_PREPARATION.md       # 35+ Viva Voce Questions & Answers
```

---

## 🚀 How to Run the Project

### 1. Start the Bank Document Verification Web App
Double-click `run_app.bat` or run:
```powershell
python app.py
```
Open your web browser and go to:
👉 **`http://127.0.0.1:8000`**

### 2. Retrain the Siamese Neural Network (Optional)
Double-click `train_model.bat` or run:
```powershell
python train.py
```

### 3. Run Evaluation & Generate ROC/EER Curves
```powershell
python evaluate.py
```

---

## 👩‍💻 Student Information
- **Name:** Mada Jahnavi
- **Project:** B.Tech Mini Project
- **Domain:** Deep Learning & Computer Vision
- **Topic:** A Siamese Neural Network-Based Approach to Handwritten Signature Verification
