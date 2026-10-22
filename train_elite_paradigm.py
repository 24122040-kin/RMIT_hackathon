# MODEL: FacebookAI/xlm-roberta-large
# RMIT Hackathon 2026 - Elite Paradigm Solver
# Innovations: Language Prefix + Concat Pooling (CLS + Mean + Max) + Multi-Sample Dropout + Focal Loss + Threshold Search
# Target: Fold 0 | Epochs: 3

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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_cosine_schedule_with_warmup
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==============================================================================
# 1. CẤU HÌNH SIÊU THAM SỐ (CONFIG)
# ==============================================================================
class Config:
    target_fold = 0         # Chạy duy nhất Fold 0
    n_splits = 5            # Chia 5 Fold chuẩn (Fold 0 = 20% validation, 80% train)
    epochs = 3              # 3 Epochs
    seed = 42
    num_classes = 3
    
    # Tối ưu max_len = 384 (Head 192 + Tail 192) giúp giảm 45% bộ nhớ attention O(L^2)
    max_len = 384
    head_len = 192
    tail_len = 192
    
    # Model backbone: XLM-RoBERTa Large
    model_name_or_path = "FacebookAI/xlm-roberta-large"
    
    # Giảm batch_size xuống 4 và bật Gradient Checkpointing -> Tiết kiệm 60% VRAM (chỉ dùng ~6GB)
    batch_size = 4
    grad_accum_steps = 4    # Effective batch = 16 (4 x 4)
    learning_rate = 1.5e-5  # Backbone LR
    classifier_lr = 5e-5    # Custom Head LR
    weight_decay = 0.01
    warmup_ratio = 0.1
    max_grad_norm = 1.0
    fp16 = True
    
    # Focal loss gamma
    focal_gamma = 2.0
    # Class weights: [BENIGN, JAILBREAK, PROMPT_INJECTION]
    class_weights = [0.8, 1.2, 3.5]

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
print(f"==========================================================")
print(f"🔥 KHỞI CHẠY ELITE PARADIGM: FOLD {Config.target_fold} | {Config.epochs} EPOCHS")
print(f"--> Thiết bị: {device}")
print(f"==========================================================")

# ==============================================================================
# 2. HỆ THỐNG DÒ TÌM ĐƯỜNG DẪN TỰ ĐỘNG
# ==============================================================================
def find_file_recursively(filename, default_dir="."):
    local_path = os.path.join(default_dir, filename)
    if os.path.exists(local_path): return local_path
    if os.path.exists(filename): return filename
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                found = os.path.join(root, filename)
                print(f"--> [AUTO-FOUND] {filename} tại: {found}")
                return found
    raise FileNotFoundError(f"Không tìm thấy {filename} trong /kaggle/input!")

def find_model_dir():
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files and ("pytorch_model.bin" in files or "model.safetensors" in files or "sentencepiece.bpe.model" in files):
                print(f"--> [AUTO-FOUND] Model offline tại: {root}")
                return root
    print(f"--> [INFO] Nạp model trực tuyến: {Config.model_name_or_path}")
    return Config.model_name_or_path

TRAIN_PATH = find_file_recursively("train.csv", default_dir="./data")
TEST_PATH = find_file_recursively("test.csv", default_dir="./data")
SAMPLE_SUB_PATH = find_file_recursively("sample_submission.csv", default_dir="./data")

LABEL_MAP = {"BENIGN": 0, "JAILBREAK": 1, "PROMPT_INJECTION": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# ==============================================================================
# 3. DATASET VỚI LANGUAGE PREFIX + HEAD & TAIL
# ==============================================================================
class ElitePromptDataset(Dataset):
    def __init__(self, texts, languages, labels=None, tokenizer=None, max_len=512, head_len=256, tail_len=256):
        self.texts = list(texts)
        self.languages = list(languages)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_len = head_len
        self.tail_len = tail_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        # 1. Ghép nhãn ngôn ngữ vào đầu câu để mô hình định hình không gian vector
        lang_prefix = f"[{str(self.languages[idx]).upper()}] "
        raw_text = str(self.texts[idx])
        full_text = lang_prefix + raw_text
        
        tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        
        # 2. Cắt Head (256) + Tail (256)
        if len(tokens) > (self.max_len - 2):
            head_tokens = tokens[:self.head_len - 1]
            tail_tokens = tokens[-(self.tail_len - 1):]
            input_ids = [self.tokenizer.bos_token_id] + head_tokens + tail_tokens + [self.tokenizer.eos_token_id]
        else:
            input_ids = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
            
        attention_mask = [1] * len(input_ids)
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
# 4. KIẾN TRÚC MÔ HÌNH: CONCAT POOLING + MULTI-SAMPLE DROPOUT
# ==============================================================================
class EliteClassifier(nn.Module):
    def __init__(self, model_name_or_path, num_classes=3):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.config.use_cache = False
        self.backbone = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        # Bật Gradient Checkpointing để giảm 60% VRAM (chống OOM 100%)
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
            
        hidden_size = self.config.hidden_size  # 1024 đối với XLM-R Large
        
        # Concat Pooling: CLS (1024) + Mean Pool (1024) + Max Pool (1024) = 3072 features
        self.feature_dim = hidden_size * 3
        
        # Multi-Sample Dropout (5 branches)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in [0.2, 0.25, 0.3, 0.35, 0.4]])
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        # Khởi tạo trọng số cho linear layer
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [Batch, SeqLen, Hidden]
        
        # 1. CLS Token
        cls_token = last_hidden_state[:, 0, :]
        
        # 2. Mean Pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        # 3. Max Pooling
        last_hidden_state_masked = last_hidden_state.clone()
        last_hidden_state_masked[input_mask_expanded == 0] = -1e9
        max_pooled = torch.max(last_hidden_state_masked, 1)[0]
        
        # Ghép 3 nguồn đặc trưng lại
        features = torch.cat([cls_token, mean_pooled, max_pooled], dim=-1)
        
        # Multi-Sample Dropout
        logits = torch.mean(torch.stack([self.fc(drop(features)) for drop in self.dropouts], dim=0), dim=0)
        return logits

# ==============================================================================
# 5. FOCAL LOSS CÓ TRỌNG SỐ LỚP
# ==============================================================================
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Tensor class weights
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ==============================================================================
# 6. HUẤN LUYỆN VÀ ĐÁNH GIÁ (TRAINING LOOP)
# ==============================================================================
def run_elite_training():
    fold = Config.target_fold
    print(f"\n--> 1. Đang tải và phân tầng dữ liệu...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    train_df["target"] = train_df["label"].map(LABEL_MAP)
    
    # Phân tầng chuẩn xác 5 fold
    train_df["strat_key"] = train_df["label"].astype(str) + "_" + train_df["language"].astype(str)
    skf = StratifiedKFold(n_splits=Config.n_splits, shuffle=True, random_state=Config.seed)
    splits = list(skf.split(train_df, train_df["strat_key"]))
    
    train_idx, val_idx = splits[fold]
    trn_data = train_df.iloc[train_idx].reset_index(drop=True)
    val_data = train_df.iloc[val_idx].reset_index(drop=True)
    
    print(f"--> Fold {fold}: Train = {len(trn_data)} mẫu (80%) | Val = {len(val_data)} mẫu (20%)")
    
    model_source = find_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    tokenizer.model_max_length = int(1e9)  # Tắt cảnh báo độ dài token
    model = EliteClassifier(model_source, num_classes=Config.num_classes).to(device)
    
    train_dataset = ElitePromptDataset(
        texts=trn_data["text"].values,
        languages=trn_data["language"].values,
        labels=trn_data["target"].values,
        tokenizer=tokenizer,
        max_len=Config.max_len,
        head_len=Config.head_len,
        tail_len=Config.tail_len
    )
    val_dataset = ElitePromptDataset(
        texts=val_data["text"].values,
        languages=val_data["language"].values,
        labels=val_data["target"].values,
        tokenizer=tokenizer,
        max_len=Config.max_len,
        head_len=Config.head_len,
        tail_len=Config.tail_len
    )
    test_dataset = ElitePromptDataset(
        texts=test_df["text"].values,
        languages=test_df["language"].values,
        labels=None,
        tokenizer=tokenizer,
        max_len=Config.max_len,
        head_len=Config.head_len,
        tail_len=Config.tail_len
    )
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)
    
    # Focal Loss
    class_weights_tensor = torch.tensor(Config.class_weights, dtype=torch.float).to(device)
    criterion = WeightedFocalLoss(alpha=class_weights_tensor, gamma=Config.focal_gamma)
    
    # Differential Learning Rate (Backbone học chậm, Head học nhanh)
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.backbone.named_parameters()], 'lr': Config.learning_rate, 'weight_decay': Config.weight_decay},
        {'params': [p for n, p in model.fc.named_parameters()], 'lr': Config.classifier_lr, 'weight_decay': 0.0}
    ]
    # foreach=False ngăn chặn PyTorch cấp phát tensor buffer 1GB gây sập OOM
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, foreach=False)
    
    total_steps = (len(train_loader) // Config.grad_accum_steps) * Config.epochs
    warmup_steps = int(total_steps * Config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=Config.fp16)
    
    best_macro_f1 = 0.0
    best_val_probs = None
    save_path = f"best_elite_model_fold_{fold}.pt"
    
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
            
            with torch.amp.autocast('cuda', enabled=Config.fp16):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / Config.grad_accum_steps
                
            scaler.scale(loss).backward()
            total_train_loss += loss.item() * Config.grad_accum_steps
            
            if (step + 1) % Config.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scale_before <= scaler.get_scale():
                    scheduler.step()
                
            pbar.set_postfix({"loss": f"{(total_train_loss / (step + 1)):.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_preds_list = []
        val_targets_list = []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                with torch.amp.autocast('cuda', enabled=Config.fp16):
                    logits = model(input_ids, attention_mask)
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
        
        print(f"\n[Epoch {epoch+1}/{Config.epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {(val_loss/len(val_loader)):.4f}")
        print(f" ⭐ MACRO F1: {epoch_macro_f1:.4f}")
        print(f"   -> F1 BENIGN: {f1_per_class[0]:.4f} | JAILBREAK: {f1_per_class[1]:.4f} | PROMPT_INJECTION: {f1_per_class[2]:.4f}")
        
        if epoch_macro_f1 > best_macro_f1:
            best_macro_f1 = epoch_macro_f1
            best_val_probs = val_probs
            print(f"   🔥 KỶ LỤC MỚI! Đang lưu checkpoint vào {save_path}...")
            torch.save(model.state_dict(), save_path)
            
    print(f"\n=======================================================")
    print(f"🏆 BEST VALIDATION MACRO F1 (Raw): {best_macro_f1:.4f}")
    print(f"=======================================================")
    
    # -------------------------------------------------------------
    # 7. THRESHOLD SEARCH (TỐI ƯU HÓA NGƯỠNG CHO PROMPT INJECTION)
    # -------------------------------------------------------------
    print("\n--> 3. Đang tinh chỉnh ngưỡng xác suất tối ưu...")
    best_multiplier = 1.0
    tuned_macro_f1 = best_macro_f1
    
    for multiplier in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]:
        calibrated_probs = best_val_probs.copy()
        calibrated_probs[:, 2] = calibrated_probs[:, 2] * multiplier  # Boost lớp 2 (Prompt Injection)
        preds = np.argmax(calibrated_probs, axis=1)
        score = f1_score(val_targets, preds, average="macro")
        if score > tuned_macro_f1:
            tuned_macro_f1 = score
            best_multiplier = multiplier
            
    print(f"--> Hệ số tối ưu cho lớp Prompt Injection: {best_multiplier}")
    print(f"⭐ MACRO F1 SAU TUNING: {tuned_macro_f1:.4f} (Tăng +{(tuned_macro_f1 - best_macro_f1):.4f})")
    
    # -------------------------------------------------------------
    # 8. SUY LUẬN TRÊN TẬP TEST
    # -------------------------------------------------------------
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    print("\n--> 4. Đang dự đoán tập Test...")
    test_preds_list = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.amp.autocast('cuda', enabled=Config.fp16):
                logits = model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            test_preds_list.append(probs)
            
    test_probs = np.vstack(test_preds_list)
    
    # Áp dụng multiplier tối ưu vào tập Test
    calibrated_test_probs = test_probs.copy()
    calibrated_test_probs[:, 2] = calibrated_test_probs[:, 2] * best_multiplier
    
    # Lưu file ma trận xác suất
    np.save(f"test_probs_elite_fold_{fold}.npy", calibrated_test_probs)
    np.save(f"val_probs_elite_fold_{fold}.npy", best_val_probs)
    
    # Xuất file submission
    test_pred_labels = np.argmax(calibrated_test_probs, axis=1)
    sub = pd.DataFrame({
        "id": test_df["id"],
        "label": [INV_LABEL_MAP[p] for p in test_pred_labels]
    })
    sub_path = "submission_elite_fold0.csv"
    sub.to_csv(sub_path, index=False)
    sub.to_csv("submission.csv", index=False)
    
    print(f"\n=======================================================")
    print(f"🎉 HOÀN THÀNH XUẤT SẮC!")
    print(f"--> File nộp bài: {sub_path} và submission.csv")
    print(f"--> Phân bố nhãn dự đoán test:")
    print(sub["label"].value_counts())
    print(f"=======================================================")

if __name__ == "__main__":
    run_elite_training()
