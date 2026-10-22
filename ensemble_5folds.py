# RMIT Hackathon 2026 - Ensemble Script for 5 Folds
# Tác dụng: Gộp trung bình cộng xác suất từ 5 Fold thành bài nộp cuối cùng (Final Submission)

import os
import numpy as np
import pandas as pd

def ensemble():
    print("=======================================================")
    print("🔥 ĐANG GỘP (ENSEMBLE) XÁC SUẤT TỪ 5 FOLD...")
    print("=======================================================")
    
    # Tìm file test.csv
    test_path = "test.csv"
    if not os.path.exists(test_path):
        if os.path.exists("./data/test.csv"):
            test_path = "./data/test.csv"
        elif os.path.exists("/kaggle/input/rmit-hackathon-2026/test.csv"):
            test_path = "/kaggle/input/rmit-hackathon-2026/test.csv"
        elif os.path.exists("/kaggle/input"):
            for root, dirs, files in os.walk("/kaggle/input"):
                if "test.csv" in files:
                    test_path = os.path.join(root, "test.csv")
                    break

    test_df = pd.read_csv(test_path)
    print(f"--> Đọc tập test: {len(test_df)} dòng.")
    
    probs_list = []
    available_folds = []
    
    for fold in range(5):
        filename = f"test_probs_fold_{fold}.npy"
        # Tìm trong thư mục hiện tại hoặc các thư mục input
        found = False
        if os.path.exists(filename):
            prob = np.load(filename)
            probs_list.append(prob)
            available_folds.append(fold)
            found = True
        elif os.path.exists("/kaggle/input"):
            for root, dirs, files in os.walk("/kaggle/input"):
                if filename in files:
                    prob = np.load(os.path.join(root, filename))
                    probs_list.append(prob)
                    available_folds.append(fold)
                    found = True
                    break
        if not found:
            print(f"⚠️ Chưa tìm thấy file: {filename}")
            
    if len(probs_list) == 0:
        raise FileNotFoundError("Chưa tìm thấy bất kỳ file test_probs_fold_X.npy nào! Hãy đặt các file npy vào thư mục.")
        
    print(f"\n--> Đã tìm thấy {len(probs_list)}/5 Folds: {available_folds}")
    
    # Trung bình cộng xác suất
    avg_probs = np.mean(probs_list, axis=0)
    
    # Dự đoán nhãn
    pred_indices = np.argmax(avg_probs, axis=1)
    INV_LABEL_MAP = {0: "BENIGN", 1: "JAILBREAK", 2: "PROMPT_INJECTION"}
    
    submission = pd.DataFrame({
        "id": test_df["id"],
        "label": [INV_LABEL_MAP[i] for i in pred_indices]
    })
    
    output_filename = "submission_ensemble_5folds.csv"
    submission.to_csv(output_filename, index=False)
    
    print(f"\n=======================================================")
    print(f"🏆 ĐÃ TẠO XONG: {output_filename}")
    print(f"--> Tổng số dòng: {len(submission)} (Kỳ vọng: 6000)")
    print("--> Phân bố nhãn dự đoán sau khi gộp:")
    print(submission["label"].value_counts())
    print("=======================================================")
    print("Bấm nút nộp file này lên Kaggle để lấy điểm bứt phá!")

if __name__ == "__main__":
    ensemble()
