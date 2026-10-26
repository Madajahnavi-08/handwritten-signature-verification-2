"""
FastAPI Bank Document Verification Application.
Author: Mada Jahnavi
Domain: Deep Learning & Computer Vision
"""

import os
import glob
import json
import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import io

from bank_verification_engine import BankVerificationEngine


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(PROJECT_DIR, "static")
TEMPLATES_DIR = os.path.join(PROJECT_DIR, "templates")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")

app = FastAPI(
    title="Handwritten Signature Verification System",
    description="A Siamese Neural Network Approach with Bank Document Verification Module (Mada Jahnavi)",
    version="1.0.0"
)

# CORS & Static files
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize Bank Verification Engine
engine = BankVerificationEngine()

# In-memory transaction audit history
AUDIT_LOGS = []


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/customers")
async def list_customers():
    """Returns registered banking customers and their specimen signatures encoded as base64."""
    results = []
    for c in engine.customers:
        sig_path = c["specimen_signature_path"]
        b64 = ""
        if os.path.exists(sig_path):
            with open(sig_path, "rb") as f:
                b64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
        results.append({
            "id": c["id"],
            "name": c["name"],
            "account_no": c["account_no"],
            "account_type": c["account_type"],
            "branch": c["branch"],
            "ifsc": c["ifsc"],
            "specimen_signature_b64": b64
        })
    return JSONResponse(content={"customers": results})


@app.get("/api/cheques")
async def list_sample_cheques():
    """Lists generated realistic bank cheques for instant verification testing."""
    cheques_dir = os.path.join(DATA_DIR, "bank_cheques")
    files = sorted(glob.glob(os.path.join(cheques_dir, "*.png")))
    cheques = []
    for f in files:
        basename = os.path.basename(f)
        parts = basename.replace(".png", "").split("_")
        user_id = f"{parts[1]}_{parts[2]}" if len(parts) >= 3 else parts[1]
        status = parts[2] if len(parts) >= 4 else "TEST"
        cust = engine.get_customer_by_id(user_id)
        cust_name = cust["name"] if cust else "Unknown Customer"
        
        with open(f, "rb") as img_file:
            b64 = f"data:image/png;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"

        cheques.append({
            "filename": basename,
            "user_id": user_id,
            "customer_name": cust_name,
            "expected_status": status,
            "image_b64": b64
        })
    return JSONResponse(content={"cheques": cheques})


@app.post("/api/verify/pair")
async def verify_signature_pair(
    customer_id: Optional[str] = Form(None),
    ref_file: Optional[UploadFile] = File(None),
    que_file: Optional[UploadFile] = File(None),
    que_sample_path: Optional[str] = Form(None)
):
    """
    Verifies a pair: Reference Signature vs Questioned Signature.
    Can accept file uploads or customer registry selections.
    """
    try:
        # 1. Resolve Reference Image
        cust_info = None
        if customer_id and customer_id != "custom":
            cust_info = engine.get_customer_by_id(customer_id)
            if not cust_info:
                raise HTTPException(status_code=404, detail="Customer not found")
            ref_cv = cv2.imread(cust_info["specimen_signature_path"])
        elif ref_file:
            ref_bytes = await ref_file.read()
            ref_cv = cv2.imdecode(np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_COLOR)
        else:
            raise HTTPException(status_code=400, detail="Missing reference signature")

        # 2. Resolve Questioned Image
        if que_file:
            que_bytes = await que_file.read()
            que_cv = cv2.imdecode(np.frombuffer(que_bytes, np.uint8), cv2.IMREAD_COLOR)
        elif que_sample_path and os.path.exists(que_sample_path):
            que_cv = cv2.imread(que_sample_path)
        else:
            raise HTTPException(status_code=400, detail="Missing questioned signature")

        if ref_cv is None or que_cv is None:
            raise HTTPException(status_code=400, detail="Invalid image file format")

        # 3. Perform Siamese Verification
        record = engine.verify_signature_pair(
            reference_input=ref_cv,
            questioned_input=que_cv,
            customer_info=cust_info
        )

        # Raw image previews for UI
        _, b1 = cv2.imencode('.png', ref_cv)
        _, b2 = cv2.imencode('.png', que_cv)
        record["visuals"]["ref_raw_b64"] = f"data:image/png;base64,{base64.b64encode(b1).decode('utf-8')}"
        record["visuals"]["que_raw_b64"] = f"data:image/png;base64,{base64.b64encode(b2).decode('utf-8')}"

        # Save to in-memory audit log
        AUDIT_LOGS.insert(0, record)
        if len(AUDIT_LOGS) > 50:
            AUDIT_LOGS.pop()

        return JSONResponse(content=record)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/verify/cheque")
async def verify_bank_cheque(
    customer_id: str = Form(...),
    cheque_file: Optional[UploadFile] = File(None),
    cheque_filename: Optional[str] = Form(None)
):
    """
    Verifies a scanned bank cheque document:
      - Extracts the drawer signature ROI
      - Fetches registered account holder's specimen signature
      - Runs Siamese neural network verification
    """
    try:
        cust = engine.get_customer_by_id(customer_id)
        if not cust:
            raise HTTPException(status_code=404, detail="Customer account not found")

        # Read cheque image
        if cheque_file:
            chq_bytes = await cheque_file.read()
            cheque_cv = cv2.imdecode(np.frombuffer(chq_bytes, np.uint8), cv2.IMREAD_COLOR)
        elif cheque_filename:
            path = os.path.join(DATA_DIR, "bank_cheques", cheque_filename)
            cheque_cv = cv2.imread(path)
        else:
            raise HTTPException(status_code=400, detail="No cheque image provided")

        if cheque_cv is None:
            raise HTTPException(status_code=400, detail="Could not read cheque image")

        # Extract signature ROI
        extracted_sig, roi_coords = engine.extract_signature_from_cheque(cheque_cv)
        
        # Load specimen signature
        ref_cv = cv2.imread(cust["specimen_signature_path"])

        # Run verification
        record = engine.verify_signature_pair(
            reference_input=ref_cv,
            questioned_input=extracted_sig,
            customer_info=cust
        )

        # Draw bounding box on cheque preview
        cheque_annotated = cheque_cv.copy()
        x1, y1, x2, y2 = roi_coords
        box_color = (0, 200, 0) if record["metrics"]["is_genuine"] else (0, 0, 220)
        cv2.rectangle(cheque_annotated, (x1, y1), (x2, y2), box_color, 3)
        cv2.putText(cheque_annotated, "SIGNATURE ROI", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        _, b_cheque = cv2.imencode('.png', cheque_annotated)
        _, b_extracted = cv2.imencode('.png', extracted_sig)
        _, b_ref = cv2.imencode('.png', ref_cv)

        record["visuals"]["cheque_preview_b64"] = f"data:image/png;base64,{base64.b64encode(b_cheque).decode('utf-8')}"
        record["visuals"]["que_raw_b64"] = f"data:image/png;base64,{base64.b64encode(b_extracted).decode('utf-8')}"
        record["visuals"]["ref_raw_b64"] = f"data:image/png;base64,{base64.b64encode(b_ref).decode('utf-8')}"

        AUDIT_LOGS.insert(0, record)
        return JSONResponse(content=record)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reports/generate")
async def generate_pdf(record_data: dict):
    """Generates official Bank Document Verification Certificate PDF."""
    try:
        pdf_path = engine.generate_pdf_report(record_data)
        filename = os.path.basename(pdf_path)
        return JSONResponse(content={"pdf_url": f"/api/reports/download/{filename}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reports/download/{filename}")
async def download_pdf(filename: str):
    file_path = os.path.join(REPORTS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf", filename=filename)
    raise HTTPException(status_code=404, detail="Report file not found")


@app.get("/api/audit-logs")
async def get_audit_logs():
    return JSONResponse(content={"logs": AUDIT_LOGS})


@app.get("/api/metrics")
async def get_system_metrics():
    """Returns training & evaluation metrics report."""
    eval_file = os.path.join(REPORTS_DIR, "evaluation_report.json")
    train_file = os.path.join(PROJECT_DIR, "saved_models", "training_metrics.json")
    
    eval_data = {}
    train_data = {}
    
    if os.path.exists(eval_file):
        with open(eval_file, "r") as f:
            eval_data = json.load(f)
    if os.path.exists(train_file):
        with open(train_file, "r") as f:
            train_data = json.load(f)
            
    return JSONResponse(content={
        "evaluation": eval_data,
        "training": train_data,
        "author": "Mada Jahnavi",
        "project": "A Siamese neural network-Based Approach to Handwritten signature verification"
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
