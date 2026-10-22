# 🛡️ RMIT Hackathon 2026: Kế hoạch cho Challenge 1 (Kaggle)

> Link: https://www.kaggle.com/competitions/rmit-hackathon-2026
> Cập nhật: 22/10/2026 lúc 09:55 (giờ VN)

> [!CAUTION]
> **DEADLINE KAGGLE: hôm nay 22/10 lúc 18:00 giờ VN (11:00 UTC).** Từ lúc viết plan còn khoảng 8 tiếng.
> **Hard stop nộp bài là 17:30.** Sau giờ đó chỉ chọn final submission, không train thêm.

---

## 1. Tóm tắt cuộc thi (đọc trong 1 phút)

| Mục | Chi tiết |
|---|---|
| **Bài toán** | Phân loại prompt thành 3 lớp: `BENIGN`, `JAILBREAK`, `PROMPT_INJECTION` |
| **Ngôn ngữ** | pt 45%, fr 25%, it 20%, en 10% |
| **Dữ liệu** | `train.csv` 14.000 dòng (`id, text, language, label`), `test.csv` 6.000 dòng |
| **Độ dài text** | 10 đến 18.550 ký tự, trung vị 294 (đa số ngắn, có đuôi rất dài) |
| **Mất cân bằng lớp** | BENIGN 62%, JAILBREAK 31%, **PROMPT_INJECTION 7%** |
| **Metric** | **Macro F1**: 3 lớp có trọng số bằng nhau, nên lớp hiếm PI quan trọng ngang BENIGN |
| **Leaderboard** | Public LB = **30%** test (~1.800 dòng), Private = 70% và **quyết định thứ hạng** |
| **Số lần nộp** | **6 lần/ngày cho cả team**. FAQ ghi 10 nhưng hệ thống đặt 6, nên cứ tính là 6 |
| **Team** | Tối đa 4 người, phải merge team trên Kaggle |
| **Điểm Challenge 1** | Có **đủ điểm C1 ngay khi nộp được 1 file hợp lệ đầu tiên**, sau đó đua leaderboard |
| **Máy** | Kaggle Notebook GPU (T4 x2 hoặc P100), khoảng 30h GPU/tuần/tài khoản |

### Ràng buộc bắt buộc ⚠️
1. **Chỉ được dùng MỘT backbone** trong 10 model được duyệt (lấy từ tab Models của cuộc thi). **Không dùng model ngoài.**
2. **Ghi rõ tên model ở dòng đầu notebook**, ví dụ `# MODEL: FacebookAI/xlm-roberta-large`.
3. **Tên notebook** phải theo dạng `<university>_<campus>_<team>_C1`, campus thuộc {`SGS`, `HNI`, `MEL`, `MYS`}.
   Ví dụ `RMIT_SGS_teamname_C1`. Chỉ dùng chữ cái, số, `_`, `-`.
4. Rules ghi *"Submit notebooks to get higher score"*, nên **nộp từ output của notebook** thay vì upload CSV tay. Share notebook cho BTC qua Group feature theo hướng dẫn trong Discussion.
5. File nộp: `id,label`, **đúng 6.000 dòng + header**, nhãn viết đúng chính tả.
6. Không được gán nhãn tay cho dữ liệu test. Không chia sẻ code với team khác.

### Mặt bằng leaderboard hiện tại (09:50)
- 44 team. Top 1: **0.984**, top 10 từ khoảng 0.968 trở lên, phần lớn nằm trong 0.93 đến 0.97.
- Nghĩa là baseline tốt đã được khoảng 0.95. **Muốn vào top cần khoảng 0.975 trở lên**, khác biệt chủ yếu nằm ở lớp PROMPT_INJECTION.

### Các challenge khác
- Challenge 2 và 3 chạy trên **RMIT RACE Cloud**, đề công bố trong ngày thi. Sự kiện kéo dài 8h–18h hôm nay và 8h–12h ngày 23/10.
- Lead theo dõi website / Discussion. Khi có đề C2/C3 thì **phân lại người** (xem mục 4).

---

## 2. Model được phép và lựa chọn

| Model | Nhận xét | Dùng không? |
|---|---|---|
| `FacebookAI/xlm-roberta-large` | Encoder đa ngôn ngữ mạnh nhất, chạy được trên T4 với fp16 | ⭐ **MODEL CHÍNH** |
| `jhu-clsp/mmBERT-base` | Encoder đa ngôn ngữ mới (kiến trúc ModernBERT), context dài, nhanh. Cần `transformers` đủ mới (≥ 4.48) | ⭐ **Phương án B** |
| `FacebookAI/xlm-roberta-base` | Nhanh, ổn định | ✅ Baseline nộp lần đầu |
| `protectai/deberta-v3-base-prompt-injection-v2` | Đã học về prompt injection nhưng chủ yếu tiếng Anh | Thử nếu dư thời gian |
| `patronus-studio/wolf-defender-prompt-injection` | Tương tự trên | Thử nếu dư thời gian |
| `distilbert-base-multilingual-cased`, `mmBERT-small` | Nhanh nhưng yếu | ❌ |
| `google/rembert` | Nặng, batch 8 | ❌ |
| `granite-3.3-2b`, `Mistral-7B` (LoRA) | Có thể mạnh, nhưng 8 tiếng trên T4 là quá rủi ro | ❌ |

> [!IMPORTANT]
> Ensemble vẫn phải dùng **cùng một backbone** (nhiều fold hoặc nhiều seed của XLM-R large) để chắc chắn đúng luật "one model".
> Nếu muốn ensemble khác backbone, phải hỏi BTC trên Discussion trước.

---

## 3. Chiến lược kỹ thuật

### 3.1 Pipeline chính
1. **Validation:** StratifiedKFold 5 fold, stratify theo `label + language`. Dùng fold 0 làm holdout để so sánh thí nghiệm.
2. **Truncation head+tail:** lấy 384 token đầu và 128 token cuối (MAXLEN 512). Injection hay nằm ở **cuối** prompt dài, cắt mỗi phần đầu dễ bỏ sót.
3. **Mất cân bằng lớp:** weighted CrossEntropy với trọng số lấy căn bậc hai của nghịch đảo tần suất (không quá gắt).
4. **Hyperparams cho XLM-R large:** LR `1e-5`, warmup 10%, 3 epoch, batch hiệu dụng 32, fp16, weight decay 0.01, `load_best_model_at_end` theo macro F1.
5. **Tune ngưỡng sau train:** nhân xác suất với trọng số lớp `[1, a, b]` rồi grid search để tối đa macro F1 trên validation/OOF. Thường tăng thêm 0.3 đến 1 điểm, gần như miễn phí.
6. **Ensemble:** lấy trung bình xác suất test của 5 fold (hoặc 2 đến 3 seed), sau đó áp trọng số đã tune trên OOF.

### 3.2 Các thí nghiệm nhanh, xếp theo ưu tiên
| # | Thí nghiệm | Kỳ vọng |
|---|---|---|
| E1 | XLM-R large, head+tail 512, weighted CE | Model chính |
| E2 | So với mmBERT-base (cùng setup, MAXLEN 1024) | Chọn backbone theo F1 fold 0 |
| E3 | Thêm prefix ngôn ngữ vào text, ví dụ `"[pt] " + text` | +0 đến 0.5 điểm |
| E4 | Tune trọng số lớp trên OOF | +0.3 đến 1 điểm |
| E5 | Ensemble 5 fold hoặc nhiều seed | +0.5 đến 1 điểm, ổn định hơn trên private LB |
| E6 | Text test trùng **y hệt** text train thì lấy nhãn train | Chỉ làm nếu EDA cho thấy trùng nhiều và nhãn nhất quán |

### 3.3 Lỗi thường gặp
- **XLM-R large bị "collapse"**: F1 kẹt khoảng 0.26 (đoán toàn BENIGN). Khi đó đổi seed, giảm LR xuống `8e-6` và kiểm tra lại warmup.
- **OOM:** giảm `per_device_train_batch_size` xuống 4 và tăng `gradient_accumulation_steps`, hoặc bật `gradient_checkpointing=True`.
- **Đầy ổ `/kaggle/working`** (giới hạn khoảng 20GB): dùng `save_total_limit=1` và `save_only_model=True`.
- **transformers bản cũ:** `eval_strategy` thì đổi thành `evaluation_strategy`.
- Chạy bằng **"Save Version → Save & Run All (Commit)"** để notebook chạy nền, không chết khi đóng trình duyệt, và có output để nộp.
- **Đừng chạy theo Public LB.** Public chỉ có khoảng 1.800 dòng, khoảng 126 dòng PI, nên sai 1 dòng PI đã lệch khoảng 0.1 điểm. **Tin CV local hơn.**

---

## 4. Phân công (4 người)

| Vai trò | Người | Việc chính |
|---|---|---|
| **A: Lead / Submission** | _(điền tên)_ | Merge team, đặt tên notebook đúng chuẩn, giữ "sổ nộp bài", quyết định nộp gì, theo dõi đề C2/C3 và Discussion, chạy baseline xlm-r-base |
| **B: Model chính** | _(điền tên)_ | XLM-R large: fold 0, sau đó các fold còn lại hoặc seed khác |
| **C: Model phụ / thí nghiệm** | _(điền tên)_ | mmBERT-base (E2), prefix ngôn ngữ (E3). Nếu chọn XLM-R large thì sang chạy thêm fold cho B |
| **D: Data / phân tích lỗi / ensemble** | _(điền tên)_ | EDA, tìm trùng lặp, phân tích lỗi theo ngôn ngữ và độ dài, tune ngưỡng, notebook ensemble cuối |

> Mỗi người chạy notebook trên **tài khoản Kaggle của mình**, nên có 4 quota GPU chạy song song. Output (`te_prob_*.npy`, `va_prob_*.npy`) được add làm input cho notebook ensemble.
> Khi đề **C2/C3** ra: C và D chuyển sang C2/C3, A và B giữ C1.

---

## 5. Timeline hôm nay (giờ VN)

| Giờ | Việc | Ai | Mốc nộp |
|---|---|---|---|
| **09:55–10:15** | Mọi người accept rules, **merge team**, tạo notebook, bật GPU T4 x2, add competition data và model vào notebook, `!ls /kaggle/input` để xem đường dẫn | Tất cả | |
| **10:15–11:00** | Baseline xlm-r-base, MAXLEN 256, 2 epoch | A | **Sub #1 trước 11:00** (lấy điểm C1) |
| 10:15–11:00 | EDA: phân bố độ dài, trùng lặp train/test, ví dụ mỗi lớp mỗi ngôn ngữ | D | |
| **10:30–13:00** | XLM-R large fold 0 | B | |
| 10:30–13:00 | mmBERT-base fold 0 | C | |
| ~13:00 | **Họp 10 phút:** so F1 fold 0, chọn backbone, phân fold | Tất cả | **Sub #2:** model đơn tốt nhất + tune ngưỡng |
| **13:00–15:30** | Chạy các fold còn lại (hoặc seed) song song trên 3 đến 4 tài khoản | B, C, A | |
| 13:00–15:30 | Phân tích lỗi, viết notebook ensemble, tune ngưỡng trên OOF | D | **Sub #3** (~14:30) nếu có cải tiến rõ (prefix, MAXLEN...) |
| **15:30–16:30** | Ensemble các fold đã xong | D, A | **Sub #4** |
| **16:30–17:15** | Ensemble đủ fold, áp trọng số đã tune | D, A | **Sub #5** |
| **17:15–17:30** | Dự phòng / sửa lỗi | A | **Sub #6** (để dành, không bắt buộc dùng) |
| **17:30–17:50** | **Chọn final submission** trên trang *Submissions*. Ưu tiên bản có **CV local cao nhất**, chọn thêm 1 bản an toàn | A | ⛔ Không nộp gì mới |

### Sổ nộp bài (A cập nhật)
| # | Giờ | Notebook / version | Mô tả | CV macro-F1 | Public LB |
|---|---|---|---|---|---|
| 1 | | | baseline xlm-r-base | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |

---

## 6. Code mẫu: XLM-R large, chạy 1 fold

> Đổi `FOLD` (0 đến 4) và `SEED` cho từng người. Kiểm tra lại đường dẫn bằng `!ls /kaggle/input/...`.

```python
# MODEL: FacebookAI/xlm-roberta-large
import numpy as np, pandas as pd, torch, torch.nn as nn, scipy.special as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from datasets import Dataset

MODEL = "/kaggle/input/models/aisuko/xlm-roberta-large/transformers/default/1"
DATA  = "/kaggle/input/competitions/rmit-hackathon-2026"
LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]; L2I = {l: i for i, l in enumerate(LABELS)}
MAXLEN, HEAD = 512, 384          # head+tail truncation
EPOCHS, LR, BS, SEED, FOLD = 3, 1e-5, 8, 42, 0

train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv")
train["y"] = train.label.map(L2I)
train["fold"] = -1
skf = StratifiedKFold(5, shuffle=True, random_state=42)   # GIỮ NGUYÊN random_state để các fold khớp nhau
for f, (_, vi) in enumerate(skf.split(train, train.label + "_" + train.language)):
    train.loc[train.index[vi], "fold"] = f

tok = AutoTokenizer.from_pretrained(MODEL)
def encode(texts):
    ids_all = tok(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
    out = []
    for ids in ids_all:
        if len(ids) > MAXLEN - 2:
            ids = ids[:HEAD] + ids[-(MAXLEN - 2 - HEAD):]
        out.append(tok.build_inputs_with_special_tokens(ids))
    return {"input_ids": out, "attention_mask": [[1] * len(x) for x in out]}

def mk(df, labelled=True):
    enc = encode(df.text.astype(str))
    if labelled: enc["labels"] = df.y.tolist()
    return Dataset.from_dict(enc)

tr, va = train[train.fold != FOLD], train[train.fold == FOLD]
ds_tr, ds_va, ds_te = mk(tr), mk(va), mk(test, False)

cnt = np.bincount(tr.y, minlength=3)
CW = torch.tensor((len(tr) / (3 * cnt)) ** 0.5, dtype=torch.float)

class WTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels"); out = model(**inputs)
        loss = nn.functional.cross_entropy(out.logits, labels, weight=CW.to(out.logits.device))
        return (loss, out) if return_outputs else loss

def metrics(p):
    return {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=L2I)

args = TrainingArguments(
    output_dir="out", per_device_train_batch_size=BS, per_device_eval_batch_size=32,
    gradient_accumulation_steps=2, learning_rate=LR, num_train_epochs=EPOCHS,
    warmup_ratio=0.1, weight_decay=0.01, fp16=True,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="macro_f1", greater_is_better=True,
    save_total_limit=1, save_only_model=True, logging_steps=50,
    report_to="none", seed=SEED, group_by_length=True)

trainer = WTrainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                   data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
trainer.train()

# ---- Dự đoán và lưu xác suất (để ensemble) ----
va_prob = sp.softmax(trainer.predict(ds_va).predictions, -1)
te_prob = sp.softmax(trainer.predict(ds_te).predictions, -1)
np.save(f"va_prob_f{FOLD}_s{SEED}.npy", va_prob); np.save(f"te_prob_f{FOLD}_s{SEED}.npy", te_prob)
va[["id"]].to_csv(f"va_ids_f{FOLD}.csv", index=False)

print(classification_report(va.y, va_prob.argmax(1), target_names=LABELS, digits=4))
for lg, g in va.assign(p=va_prob.argmax(1)).groupby("language"):
    print(lg, round(f1_score(g.y, g.p, average="macro"), 4))

# ---- Tune trọng số lớp (dùng grid thô để tránh overfit) ----
base = f1_score(va.y, va_prob.argmax(1), average="macro"); best = (base, 1.0, 1.0)
for a in np.arange(0.5, 3.01, 0.1):
    for b in np.arange(0.5, 3.01, 0.1):
        s = f1_score(va.y, (va_prob * [1, a, b]).argmax(1), average="macro")
        if s > best[0]: best = (s, a, b)
print("base", base, "tuned", best)
W = np.array([1, best[1], best[2]]) if best[0] - base > 0.003 else np.ones(3)

sub = pd.DataFrame({"id": test.id, "label": [LABELS[i] for i in (te_prob * W).argmax(1)]})
assert len(sub) == 6000 and sub.label.isin(LABELS).all()
sub.to_csv("submission.csv", index=False)
print(sub.label.value_counts(normalize=True))   # nên gần 62/31/7
```

### Notebook ensemble (D)
```python
# MODEL: FacebookAI/xlm-roberta-large  (ensemble 5 fold, cùng backbone)
# Add output các notebook fold làm Input. Đọc te_prob_*.npy rồi lấy trung bình.
# Ghép va_prob của các fold thành OOF (khớp theo va_ids_f*.csv), tune W trên OOF.
# sub = argmax(mean_te_prob * W), sau đó assert 6000 dòng và ghi submission.csv
```

---

## 7. Checklist trước MỖI lần nộp ✅
- [ ] File có đúng 6.000 dòng + header, cột `id,label`
- [ ] Nhãn chỉ gồm `BENIGN` / `JAILBREAK` / `PROMPT_INJECTION`
- [ ] Phân bố nhãn dự đoán hợp lý (khoảng 62/31/7)
- [ ] Notebook đặt tên đúng `<uni>_<campus>_<team>_C1`, dòng đầu ghi model
- [ ] CV macro-F1 đã ghi vào sổ nộp bài
- [ ] Bản này thật sự tốt hơn bản trước trên CV (đừng phí lượt nộp)

## 8. Việc cần làm ngay
1. Cả team: accept rules, Lead gửi lời mời team trên Kaggle, mọi người **confirm**.
2. Điền tên vào bảng phân công (mục 4), thống nhất **campus** và **tên team** cho tên notebook.
3. A chạy baseline để **nộp Sub #1 trước 11:00**.
