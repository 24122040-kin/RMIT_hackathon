# MODEL: FacebookAI/xlm-roberta-large
# RMIT Hackathon 2026 - Multilingual Prompt Classification
# Backbone: XLM-RoBERTa-large with Head+Tail Tokenization & Class-Weighted Loss

import os
import sys
import gc
import json
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup
)

# ==============================================================================
# 1. CẤU HÌNH SIÊU THAM SỐ (CONFIG)
# ==============================================================================
class Config:
    seed = 42
    num_classes = 3
    max_len = 512
    head_len = 256
    tail_len = 256
    
    # Model backbone: Tự động quét các đường dẫn Kaggle Models mount thực tế
    model_name_or_path = "FacebookAI/xlm-roberta-large"
    kaggle_candidate_paths = [
        "/kaggle/input/xlm-roberta-large/transformers/default/1",
        "/kaggle/input/models/aisuko/xlm-roberta-large/transformers/default/1",
        "/kaggle/input/models/aisuko/xlm-roberta-large",
        "/kaggle/input/xlm-roberta-large",
        "FacebookAI/xlm-roberta-large"
    ]
    
    # Training settings
    epochs = 3
    batch_size = 8          # T4/P100 (16GB VRAM)
    grad_accum_steps = 2    # Effective batch size = 16
    learning_rate = 1e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    max_grad_norm = 1.0
    fp16 = True
    
    # Loss weights: phạt nặng hơn khi đoán sai lớp hiếm PROMPT_INJECTION (7%)
    # Thứ tự lớp: [BENIGN, JAILBREAK, PROMPT_INJECTION]
    class_weights = [0.7, 1.2, 3.2]

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(Config.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> Using device: {device}")

# ==============================================================================
# 2. HỆ THỐNG TỰ ĐỘNG DÒ TÌM ĐƯỜNG DẪN DATA & MODEL TRÊN KAGGLE
# ==============================================================================
def find_file_recursively(filename, default_dir="."):
    # 1. Thử đường dẫn mặc định
    local_path = os.path.join(default_dir, filename)
    if os.path.exists(local_path):
        return local_path
    if os.path.exists(filename):
        return filename
    
    # 2. Quét toàn bộ cây thư mục /kaggle/input
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                found_path = os.path.join(root, filename)
                print(f"--> [AUTO-FOUND] Đã tìm thấy {filename} tại: {found_path}")
                return found_path
                
    raise FileNotFoundError(f"Không thể tìm thấy file '{filename}' trong /kaggle/input hay thư mục hiện tại! Hãy kiểm tra xem bạn đã add dataset vào notebook chưa.")

def find_model_dir():
    # Quét tìm thư mục model xlm-roberta-large trong /kaggle/input
    if os.path.exists("/kaggle/input"):
        print("\n=== CẤU TRÚC THƯ MỤC /kaggle/input HIỆN TẠI ===")
        for root, dirs, files in os.walk("/kaggle/input"):
            level = root.replace("/kaggle/input", "").count(os.sep)
            indent = " " * 4 * level
            print(f"{indent}📁 {os.path.basename(root)}/")
            for f in files[:4]:
                print(f"{indent}    📄 {f}")
        print("===================================================\n")
        
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files and ("pytorch_model.bin" in files or "model.safetensors" in files or "sentencepiece.bpe.model" in files):
                print(f"--> [AUTO-FOUND] Đã tìm thấy Model offline tại: {root}")
                return root
                
    print("--> [INFO] Không tìm thấy model offline trong /kaggle/input, sử dụng 'FacebookAI/xlm-roberta-large' tải trực tuyến.")
    return "FacebookAI/xlm-roberta-large"

TRAIN_PATH = find_file_recursively("train.csv", default_dir="./data")
TEST_PATH = find_file_recursively("test.csv", default_dir="./data")
SAMPLE_SUB_PATH = find_file_recursively("sample_submission.csv", default_dir="./data")

LABEL_MAP = {"BENIGN": 0, "JAILBREAK": 1, "PROMPT_INJECTION": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# ==============================================================================
# 3. DATASET VỚI KỸ THUẬT HEAD + TAIL TOKENIZATION
# ==============================================================================
class PromptHeadTailDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=512, head_len=256, tail_len=256):
        self.texts = list(texts)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_len = head_len
        self.tail_len = tail_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize không truncate để kiểm tra độ dài thật
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Nếu độ dài vượt quá max_len - 2 (trừ <s> và </s>): cắt Head + Tail
        if len(tokens) > (self.max_len - 2):
            head_tokens = tokens[:self.head_len - 1]
            tail_tokens = tokens[-(self.tail_len - 1):]
            input_ids = [self.tokenizer.bos_token_id] + head_tokens + tail_tokens + [self.tokenizer.eos_token_id]
        else:
            input_ids = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
            
        attention_mask = [1] * len(input_ids)
        
        # Padding thủ công đến max_len
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }
        
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item

# ==============================================================================
# 4. CHUẨN BỊ MÔ HÌNH VÀ TOKENIZER
# ==============================================================================
def get_model_and_tokenizer():
    model_source = find_model_dir()
    print(f"--> Đang nạp Model & Tokenizer từ: {model_source}")
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    config = AutoConfig.from_pretrained(model_source, num_labels=Config.num_classes)
    model = AutoModelForSequenceClassification.from_pretrained(model_source, config=config)
    
    return model, tokenizer

# ==============================================================================
# 5. HÀM HUẤN LUYỆN VÀ ĐÁNH GIÁ (TRAIN & EVAL)
# ==============================================================================
def train_and_predict():
    print("--> 1. Đang tải dữ liệu...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    train_df["target"] = train_df["label"].map(LABEL_MAP)
    
    # Chia tập Stratified theo cả Label và Ngôn ngữ để đánh giá uy tín
    train_df["strat_key"] = train_df["label"].astype(str) + "_" + train_df["language"].astype(str)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.seed)
    
    # Chỉ lấy Fold 0 làm Validation (tỷ lệ 80% train, 20% validation)
    train_idx, val_idx = next(skf.split(train_df, train_df["strat_key"]))
    trn_data = train_df.iloc[train_idx].reset_index(drop=True)
    val_data = train_df.iloc[val_idx].reset_index(drop=True)
    
    print(f"--> Train size: {len(trn_data)}, Val size: {len(val_data)}, Test size: {len(test_df)}")
    print("--> Phân bố nhãn trong tập Train:")
    print(trn_data["label"].value_counts(normalize=True))
    
    model, tokenizer = get_model_and_tokenizer()
    model.to(device)
    
    train_dataset = PromptHeadTailDataset(
        texts=trn_data["text"].values,
        labels=trn_data["target"].values,
        tokenizer=tokenizer,
        max_len=Config.max_len,
        head_len=Config.head_len,
        tail_len=Config.tail_len
    )
    val_dataset = PromptHeadTailDataset(
        texts=val_data["text"].values,
        labels=val_data["target"].values,
        tokenizer=tokenizer,
        max_len=Config.max_len,
        head_len=Config.head_len,
        tail_len=Config.tail_len
    )
    test_dataset = PromptHeadTailDataset(
        texts=test_df["text"].values,
        labels=None,
        tokenizer=tokenizer,
        max_len=Config.max_len,
        head_len=Config.head_len,
        tail_len=Config.tail_len
    )
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)
    
    # Thiết lập Loss có trọng số lớp (Class Weights)
    class_weights_tensor = torch.tensor(Config.class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Optimizer & Scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': Config.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=Config.learning_rate)
    
    total_steps = (len(train_loader) // Config.grad_accum_steps) * Config.epochs
    warmup_steps = int(total_steps * Config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=Config.fp16)
    
    # Vòng lặp huấn luyện (Training Loop)
    best_macro_f1 = 0.0
    best_val_preds = None
    best_model_weights = None
    
    print("\n--> 2. Bắt đầu huấn luyện...")
    for epoch in range(Config.epochs):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.cuda.amp.autocast(enabled=Config.fp16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
                loss = loss / Config.grad_accum_steps
                
            scaler.scale(loss).backward()
            total_train_loss += loss.item() * Config.grad_accum_steps
            
            if (step + 1) % Config.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
            pbar.set_postfix({"loss": f"{(total_train_loss / (step + 1)):.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Đánh giá trên Validation (Validation Phase)
        model.eval()
        val_preds_list = []
        val_targets_list = []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                with torch.cuda.amp.autocast(enabled=Config.fp16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                val_preds_list.append(probs)
                val_targets_list.append(labels.cpu().numpy())
                
        val_probs = np.vstack(val_preds_list)
        val_targets = np.concatenate(val_targets_list)
        val_pred_labels = np.argmax(val_probs, axis=1)
        
        epoch_macro_f1 = f1_score(val_targets, val_pred_labels, average="macro")
        f1_per_class = f1_score(val_targets, val_pred_labels, average=None)
        
        print(f"\n[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {(val_loss/len(val_loader)):.4f}")
        print(f" ⭐ MACRO F1: {epoch_macro_f1:.4f}")
        print(f"   -> F1 BENIGN: {f1_per_class[0]:.4f} | JAILBREAK: {f1_per_class[1]:.4f} | PROMPT_INJECTION: {f1_per_class[2]:.4f}")
        
        if epoch_macro_f1 > best_macro_f1:
            best_macro_f1 = epoch_macro_f1
            best_val_preds = val_probs
            print(f"   🔥 Điểm kỷ lục mới! Đang lưu checkpoint...")
            torch.save(model.state_dict(), "best_xlm_roberta_large.pt")
            
    print(f"\n==========================================")
    print(f"🏆 BEST VALIDATION MACRO F1: {best_macro_f1:.4f}")
    print(f"==========================================")
    
    # Tải lại trọng số tốt nhất để dự đoán Test
    model.load_state_dict(torch.load("best_xlm_roberta_large.pt"))
    model.eval()
    
    print("\n--> 3. Đang suy luận trên tập Test...")
    test_preds_list = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.cuda.amp.autocast(enabled=Config.fp16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            test_preds_list.append(probs)
            
    test_probs = np.vstack(test_preds_list)
    
    # Lưu xác suất test và validation để dùng ensemble ở các bước sau
    np.save("test_probs_large.npy", test_probs)
    np.save("val_probs_large.npy", best_val_preds)
    
    # Xuất file submission
    test_pred_labels = np.argmax(test_probs, axis=1)
    submission = pd.DataFrame({
        "id": test_df["id"],
        "label": [INV_LABEL_MAP[p] for p in test_pred_labels]
    })
    
    submission.to_csv("submission.csv", index=False)
    print("--> 4. Đã lưu submission.csv thành công!")
    print(f"--> Kiểm tra số dòng: {len(submission)} (kỳ vọng 6000)")
    print(submission["label"].value_counts())
    print("\n[Hoàn thành! Bạn có thể tải file submission.csv để nộp ngay lên Kaggle]")

if __name__ == "__main__":
    train_and_predict()
