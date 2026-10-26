"""
Bank Document Verification & Signature Matching Engine.
Provides bank document processing, signature extraction, Siamese inference,
stroke difference heatmap generation, and audit certificate creation.
Author: Mada Jahnavi
Domain: Deep Learning & Computer Vision
"""

import os
import io
import time
import uuid
import json
import base64
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

from model import SiameseSignatureNet, compute_similarity
from preprocessing import preprocess_signature_image, preprocess_to_tensor


class BankVerificationEngine:
    """
    Core backend inference engine for the Bank Document-Verification Module.
    """
    def __init__(self, model_path: str = None, threshold: float = None):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.project_dir, "data")
        self.reports_dir = os.path.join(self.project_dir, "reports")
        
        # Load optimal threshold from evaluation report if available
        eval_report_file = os.path.join(self.reports_dir, "evaluation_report.json")
        if threshold is not None:
            self.threshold = threshold
        elif os.path.exists(eval_report_file):
            try:
                with open(eval_report_file, "r") as f:
                    data = json.load(f)
                    self.threshold = float(data.get("optimal_threshold", 0.18))
            except Exception:
                self.threshold = 0.18
        else:
            self.threshold = 0.18
        
        if model_path is None:
            self.model_path = os.path.join(self.project_dir, "saved_models", "best_siamese_model.pth")
        else:
            self.model_path = model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()
        self.customers = self._init_customer_database()

    def _load_model(self):
        """Loads trained Siamese weights or initializes with pretrained weights if not yet trained."""
        self.model = SiameseSignatureNet(embedding_dim=128).to(self.device)
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"[Bank Engine] Loaded trained Siamese model from: {self.model_path}")
        else:
            print(f"[Bank Engine] Note: Model weights not found at {self.model_path}. Please run train.py!")
        self.model.eval()

    def _init_customer_database(self):
        """Pre-registered bank customer accounts linked to their genuine specimen signatures."""
        genuine_dir = os.path.join(self.data_dir, "raw", "genuine")
        
        customers = [
            {
                "id": "user_01",
                "name": "Mada Jahnavi",
                "account_no": "SB-1092847291",
                "account_type": "Corporate Current",
                "branch": "Financial District Branch, Hyderabad",
                "ifsc": "NTAB0001429",
                "specimen_signature_path": os.path.join(genuine_dir, "user_01_genuine_01.png")
            },
            {
                "id": "user_02",
                "name": "K. Rajesh Kumar",
                "account_no": "SB-2049182390",
                "account_type": "Premium Savings",
                "branch": "MG Road Branch, Bengaluru",
                "ifsc": "NTAB0002105",
                "specimen_signature_path": os.path.join(genuine_dir, "user_02_genuine_01.png")
            },
            {
                "id": "user_03",
                "name": "P. Priya Sharma",
                "account_no": "SB-3091827401",
                "account_type": "Salary Account",
                "branch": "BKC Branch, Mumbai",
                "ifsc": "NTAB0003022",
                "specimen_signature_path": os.path.join(genuine_dir, "user_03_genuine_01.png")
            },
            {
                "id": "user_04",
                "name": "V. Suresh Reddy",
                "account_no": "SB-4019283742",
                "account_type": "Business Current",
                "branch": "Banjara Hills, Hyderabad",
                "ifsc": "NTAB0001430",
                "specimen_signature_path": os.path.join(genuine_dir, "user_04_genuine_01.png")
            },
            {
                "id": "user_05",
                "name": "A. Venkat Rao",
                "account_no": "SB-5091823743",
                "account_type": "Savings Super",
                "branch": "Anna Salai, Chennai",
                "ifsc": "NTAB0004510",
                "specimen_signature_path": os.path.join(genuine_dir, "user_05_genuine_01.png")
            }
        ]
        return customers

    def get_customer_by_id(self, customer_id: str):
        for c in self.customers:
            if c["id"] == customer_id:
                return c
        return None

    def extract_signature_from_cheque(self, cheque_img_input):
        """
        Extracts the Drawer Signature region of interest (ROI) from a standard bank cheque.
        Cheque dimensions normalized: width ~ 750, height ~ 340.
        Signature box is located at x: 520 to 730, y: 180 to 280.
        """
        if isinstance(cheque_img_input, str):
            cheque_cv = cv2.imread(cheque_img_input)
        elif isinstance(cheque_img_input, Image.Image):
            cheque_cv = cv2.cvtColor(np.array(cheque_img_input), cv2.COLOR_RGB2BGR)
        else:
            cheque_cv = cheque_img_input

        h, w, _ = cheque_cv.shape
        # Relative coordinates for bank cheque drawer signature zone (above the signatory line)
        x1 = int(w * 0.72)
        y1 = int(h * 0.52)
        x2 = int(w * 0.98)
        y2 = int(h * 0.78)

        extracted_sig = cheque_cv[y1:y2, x1:x2].copy()
        return extracted_sig, (x1, y1, x2, y2)

    def generate_stroke_difference_heatmap(self, proc1: np.ndarray, proc2: np.ndarray) -> str:
        """
        Computes spatial pixel-level stroke variance and renders a color heatmap overlay
        indicating stroke agreement (green) vs divergence/discrepancies (red).
        Returns base64 PNG data string.
        """
        # Absolute difference between normalized signatures
        diff = np.abs(proc1 - proc2)
        diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
        
        # Color mapping: 0 difference -> dark blue/black, high difference -> bright red/yellow
        heatmap_norm = np.clip(diff_blur * 255.0 * 2.0, 0, 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        # Blend with grayscale background for clarity
        base_gray = ((proc1 + proc2) * 0.5 * 255).astype(np.uint8)
        base_bgr = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(base_bgr, 0.45, heatmap_color, 0.55, 0)

        # Encode to Base64
        _, buffer = cv2.imencode('.png', blended)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{b64_str}"

    def verify_signature_pair(self, reference_input, questioned_input, customer_info: dict = None) -> dict:
        """
        Executes end-to-end Siamese verification between Reference Signature and Questioned Signature.
        """
        # Preprocess both images through structured pipeline
        proc_ref = preprocess_signature_image(reference_input)
        proc_que = preprocess_signature_image(questioned_input)

        # Convert to model tensors
        tensor_ref = torch.from_numpy(proc_ref).unsqueeze(0).unsqueeze(0).to(self.device)
        tensor_que = torch.from_numpy(proc_que).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb_ref, emb_que = self.model(tensor_ref, tensor_que)
            euclidean_dist = float(F.pairwise_distance(emb_ref, emb_que).squeeze().cpu().item())

        # Banking metrics
        decision = compute_similarity(distance=euclidean_dist, threshold=self.threshold)
        
        # Stroke difference heatmap visualization
        heatmap_b64 = self.generate_stroke_difference_heatmap(proc_ref, proc_que)

        # Base64 encodings of preprocessed images
        _, buf1 = cv2.imencode('.png', (proc_ref * 255).astype(np.uint8))
        _, buf2 = cv2.imencode('.png', (proc_que * 255).astype(np.uint8))
        ref_proc_b64 = f"data:image/png;base64,{base64.b64encode(buf1).decode('utf-8')}"
        que_proc_b64 = f"data:image/png;base64,{base64.b64encode(buf2).decode('utf-8')}"

        verdict_badge = "VERIFIED - GENUINE SIGNATURE" if decision["is_genuine"] else "ALERT - FORGERY SUSPECTED"
        badge_class = "badge-success" if decision["is_genuine"] else "badge-danger"
        action_recommendation = (
            "Approve transaction. Specimen signature match confidence meets banking security thresholds."
            if decision["is_genuine"]
            else "Hold transaction. Critical stroke divergence detected exceeding allowed variance threshold. Escalate to branch supervisor."
        )

        verification_record = {
            "transaction_id": f"TXN-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "customer": customer_info or {
                "name": "General Verification",
                "account_no": "N/A",
                "account_type": "Cheque Document / Slip",
                "branch": "Central Clearing House"
            },
            "metrics": {
                "euclidean_distance": decision["distance"],
                "threshold": self.threshold,
                "similarity_pct": decision["similarity_pct"],
                "confidence_pct": decision["confidence"],
                "is_genuine": decision["is_genuine"]
            },
            "decision": {
                "verdict": verdict_badge,
                "badge_class": badge_class,
                "action": action_recommendation
            },
            "visuals": {
                "ref_proc_b64": ref_proc_b64,
                "que_proc_b64": que_proc_b64,
                "heatmap_b64": heatmap_b64
            }
        }
        return verification_record

    def generate_pdf_report(self, record: dict) -> str:
        """
        Creates an official Bank Document Verification Certificate / Audit Report PDF.
        """
        os.makedirs(self.reports_dir, exist_ok=True)
        filename = f"Bank_Audit_{record['transaction_id']}.pdf"
        filepath = os.path.join(self.reports_dir, filename)

        c = canvas.Canvas(filepath, pagesize=letter)
        w_page, h_page = letter

        # Bank Header
        c.setFillColor(colors.HexColor("#0f172a"))
        c.rect(0, h_page - 80, w_page, 80, fill=1, stroke=0)

        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, h_page - 40, "APEX NATIONAL TRUST BANK - AUDIT SYSTEM")
        c.setFont("Helvetica", 10)
        c.drawString(40, h_page - 60, "AUTOMATED AI SIGNATURE VERIFICATION AUDIT CERTIFICATE")

        # Metadata Box
        c.setFillColor(colors.HexColor("#f8fafc"))
        c.setStrokeColor(colors.HexColor("#cbd5e1"))
        c.rect(40, h_page - 180, w_page - 80, 85, fill=1, stroke=1)

        c.setFillColor(colors.HexColor("#1e293b"))
        c.setFont("Helvetica-Bold", 10)
        c.drawString(55, h_page - 110, f"Transaction ID: {record['transaction_id']}")
        c.drawString(320, h_page - 110, f"Audit Timestamp: {record['timestamp']}")

        cust = record["customer"]
        c.setFont("Helvetica", 9)
        c.drawString(55, h_page - 130, f"Account Holder: {cust['name']}")
        c.drawString(320, h_page - 130, f"Account Number: {cust['account_no']}")
        c.drawString(55, h_page - 150, f"Branch Location: {cust.get('branch', 'Main Branch')}")
        c.drawString(320, h_page - 150, f"System Operator: Mada Jahnavi (Officer ID: AGY-2026)")

        # Result Banner
        is_gen = record["metrics"]["is_genuine"]
        banner_color = colors.HexColor("#16a34a") if is_gen else colors.HexColor("#dc2626")
        c.setFillColor(banner_color)
        c.rect(40, h_page - 235, w_page - 80, 40, fill=1, stroke=0)

        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(w_page / 2.0, h_page - 222, record["decision"]["verdict"])

        # Verification Scores Grid
        c.setFillColor(colors.HexColor("#1e293b"))
        c.setFont("Helvetica-Bold", 11)
        c.drawString(55, h_page - 265, "SIAMESE NEURAL NETWORK METRIC EVALUATION:")

        c.setFont("Helvetica", 10)
        c.drawString(55, h_page - 290, f"• Match Similarity: {record['metrics']['similarity_pct']}%")
        c.drawString(55, h_page - 310, f"• Euclidean Distance ||f(x1) - f(x2)||: {record['metrics']['euclidean_distance']}")
        c.drawString(300, h_page - 290, f"• Verification Confidence: {record['metrics']['confidence_pct']}%")
        c.drawString(300, h_page - 310, f"• Decision Separation Threshold: {record['metrics']['threshold']}")

        c.drawString(55, h_page - 340, "Actionable Directive:")
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(55, h_page - 355, record["decision"]["action"])

        # Sign-off stamp
        c.setStrokeColor(colors.HexColor("#64748b"))
        c.rect(w_page - 220, 100, 170, 75, fill=0, stroke=1)
        c.setFont("Helvetica", 8)
        c.drawCentredString(w_page - 135, 160, "DOCUMENT EXAMINER SIGN-OFF")
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(w_page - 135, 125, "Mada Jahnavi")
        c.setFont("Helvetica", 8)
        c.drawCentredString(w_page - 135, 110, "Authorised Bank Examiner")

        # Footer
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#94a3b8"))
        c.drawCentredString(w_page / 2.0, 30, "Generated by Siamese Neural Network Bank Document Verification Module • Mini Project B.Tech")

        c.save()
        return filepath
