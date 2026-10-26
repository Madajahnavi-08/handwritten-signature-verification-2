"""
Quick end-to-end integration test of the Bank Verification Engine.
Author: Mada Jahnavi
"""

import os
import cv2
from bank_verification_engine import BankVerificationEngine


def run_test():
    engine = BankVerificationEngine()
    print("Testing Bank Verification Engine...")

    cust = engine.customers[0]  # Mada Jahnavi
    ref_path = cust["specimen_signature_path"]

    # Test Genuine Match (User 01, Genuine 02)
    gen2_path = os.path.join(engine.data_dir, "raw", "genuine", "user_01_genuine_02.png")
    res_gen = engine.verify_signature_pair(ref_path, gen2_path, cust)
    print("\n--- Genuine Pair Test ---")
    print(f"Customer: {cust['name']}")
    print(f"Distance: {res_gen['metrics']['euclidean_distance']}")
    print(f"Similarity: {res_gen['metrics']['similarity_pct']}%")
    print(f"Verdict: {res_gen['decision']['verdict']}")
    assert res_gen['metrics']['is_genuine'] is True, "Genuine signature should be verified!"

    # Test Forged Match (User 01, Forged 10)
    forg_path = os.path.join(engine.data_dir, "raw", "forged", "user_01_forged_10.png")
    res_forg = engine.verify_signature_pair(ref_path, forg_path, cust)
    print("\n--- Forged Pair Test ---")
    print(f"Customer: {cust['name']}")
    print(f"Distance: {res_forg['metrics']['euclidean_distance']}")
    print(f"Similarity: {res_forg['metrics']['similarity_pct']}%")
    print(f"Verdict: {res_forg['decision']['verdict']}")
    assert res_forg['metrics']['is_genuine'] is False, "Forged signature should be flagged!"

    # Test PDF generation
    pdf_path = engine.generate_pdf_report(res_gen)
    print(f"\nPDF Report generated successfully at: {pdf_path}")
    assert os.path.exists(pdf_path), "PDF report must exist!"

    # Test Cheque Verification
    cheque_path = os.path.join(engine.data_dir, "bank_cheques", "cheque_user_01_GENUINE_01.png")
    if os.path.exists(cheque_path):
        extracted, _ = engine.extract_signature_from_cheque(cheque_path)
        res_cheque = engine.verify_signature_pair(ref_path, extracted, cust)
        print("\n--- Cheque Verification Test ---")
        print(f"Cheque Signature Distance: {res_cheque['metrics']['euclidean_distance']}")
        print(f"Cheque Signature Similarity: {res_cheque['metrics']['similarity_pct']}%")
        print(f"Cheque Verdict: {res_cheque['decision']['verdict']}")

    print("\n==========================================")
    print(" ALL TESTS PASSED SUCCESSFULLY! ")
    print("==========================================")


if __name__ == "__main__":
    run_test()
