# RMIT Hackathon 2026 — Kế hoạch thực thi cho đội 4 người

**Ngày lập:** 22/10/2026, khoảng 10:05, giờ Việt Nam (UTC+7).  
**Đối tượng:** cả đội; có thể gửi nguyên file này vào nhóm chat hoặc đưa lên repo riêng.  
**Tên đội:** … | **Campus:** … | **A:** … | **B:** … | **C:** … | **D:** …

## 1. Đọc trước khi bắt đầu

**Mục tiêu theo thứ tự: có bài nộp hợp lệ → có mô hình được đánh giá đáng tin cậy → cải thiện điểm → hoàn thiện hồ sơ bàn giao.** Mỗi việc có một người chịu trách nhiệm chính và một đầu ra cụ thể. A giữ lịch và quyết định bài nộp; B phụ trách mô hình; C phụ trách thí nghiệm và hỗ trợ huấn luyện; D phụ trách dữ liệu, đánh giá và tích hợp.

### Nguồn và mức độ xác minh

| Nội dung | Tình trạng tại thời điểm lập kế hoạch |
|---|---|
| Sự kiện ngày 22–23/10/2026, lịch công bố 08:00–18:00 ngày 22 và 08:00–12:00 ngày 23 | Có trên [website chính thức](https://rmit-hackathon.com/). Team ở Việt Nam cần đối chiếu lịch tại campus; không dùng lịch sự kiện thay deadline Kaggle |
| Đội 4 sinh viên; mọi thành viên có mặt cùng địa điểm; một warm-up và hai challenge; ngôn ngữ English | Có trên [website chính thức](https://rmit-hackathon.com/) |
| Đề cụ thể, metric, số dòng, danh sách model, hạn nộp, quota | **Chưa xác minh trực tiếp**: [Kaggle Overview](https://www.kaggle.com/competitions/rmit-hackathon-2026/overview) chỉ tải được tiêu đề, không tải được nội dung trong phiên kiểm tra |
| C1 là phân loại `BENIGN/JAILBREAK/PROMPT_INJECTION`; train 14.000, test 6.000; macro F1; luật một backbone | Ghi chép trong `PLAN.md` của workspace, **dùng làm giả định làm việc**, chưa coi là xác nhận của BTC |
| Deadline C1 18:00 ngày 22/10; 6 lượt nộp/ngày; public/private 30/70; nộp qua notebook | Ghi chép cũ trong `PLAN.md`; A phải xác nhận trên Kaggle trước khi chốt lịch và chiến lược |
| C2/C3 chạy trên RACE Cloud; nộp một file là đủ điểm C1 | Chỉ có trong ghi chép cũ; **không mặc định đúng** trước khi thấy brief/rubric chính thức |

**Kế hoạch kỹ thuật bên dưới dành cho bài phân loại trong ghi chép cũ.** Tên gọi “C1” là tên nội bộ tạm thời, chưa kết luận đó là warm-up hay challenge được tính điểm. Nếu đề thực tế khác, A cập nhật mục 2 rồi điều chỉnh phạm vi; không tiếp tục train theo giả định sai.

## 2. Việc phải chốt trong 15 phút đầu — A chịu trách nhiệm

Điền thông tin vào bảng này và gửi một thông báo duy nhất cho cả đội. Các thành viên tự thao tác chấp nhận điều lệ trên tài khoản của mình.

| Cần chốt | Giá trị xác nhận | Nơi kiểm tra |
|---|---|---|
| C1 là warm-up hay bài tính điểm? Trọng số mỗi challenge? | … | Overview / brief / rubric BTC |
| Deadline chính xác, múi giờ, hạn merge team | … | Timeline / Rules / Teams |
| Metric và định dạng đầu ra | … | Evaluation + sample submission |
| Số lượt nộp còn lại của cả team, thời điểm reset | … | Trang Submit / Submissions |
| Số bản được chọn final và cách chọn | … | Evaluation / Submissions |
| Model được phép, phiên bản/checkpoint được phép | … | Rules / Models |
| “Một model” áp dụng cho toàn cuộc thi, từng notebook hay final? Có được thử model khác không? | … | Rules / giải đáp BTC |
| Ensemble nhiều fold/seed cùng backbone có được phép? | … | Rules / giải đáp BTC |
| Cho phép dữ liệu ngoài, augmentation, dịch, pseudo-label hay không? | … | Rules |
| Nộp CSV hay notebook; notebook có cần Internet off, chia sẻ hoặc đặt tên theo mẫu? | … | Submission instructions |
| C2/C3: giờ ra đề, deadline, nền tảng, tài khoản, đầu ra, trọng số | … | Brief tại sự kiện / thông báo BTC |

**Trong lúc chưa rõ luật một backbone:** chỉ chuẩn bị một backbone dự kiến hợp lệ; chưa train thử backbone thứ hai, chưa ghép ensemble, chưa dùng dữ liệu ngoài hoặc suy nhãn từ test. A cần lưu URL hoặc nội dung trả lời của BTC nếu luật có điểm mơ hồ.

## 3. Phân công tổng thể

| Thành viên | Vai trò | Năng lực phù hợp | Đầu ra chính | Người dự phòng |
|---|---|---|---|---|
| **A** | Lead, vận hành, nộp bài, tổng hợp báo cáo | Đọc luật kỹ, quản lý thời gian, Python cơ bản | Bảng luật đã xác nhận, baseline hợp lệ, sổ nộp bài, bản final | D |
| **B** | Chủ mô hình và pipeline huấn luyện | PyTorch/Transformers, xử lý GPU | Notebook train chuẩn, checkpoint tốt nhất, xác suất validation/test | C |
| **C** | Thí nghiệm có kiểm soát, hỗ trợ train, đầu mối challenge tiếp theo | Python/ML, thử nghiệm nhanh, sửa lỗi môi trường | Báo cáo so sánh cấu hình, run bổ sung, brief C2/C3 khi có | B |
| **D** | Dữ liệu, validation, phân tích lỗi, tích hợp | pandas/sklearn, cẩn thận với ID và metric | Fold dùng chung, báo cáo dữ liệu, OOF, kiểm tra submission | A |

**Luồng bàn giao:** D giao dữ liệu và fold → B/C/A chạy theo cùng quy ước → D kiểm tra và tích hợp → A nộp → A ghi kết quả và báo lại nhóm.

Không để một người vừa train mọi thứ, vừa kiểm tra, vừa nộp. Trước final, A và D cùng kiểm tra độc lập file và notebook.

## 4. Phiếu công việc A — Lead / Submission

### Trong 30 phút đầu

- [ ] Điền tên, kỹ năng và GPU khả dụng của 4 người; chỉ định D là người thay thế khi A vắng.
- [ ] Chốt bảng luật mục 2; kiểm tra team đủ thành viên và hạn merge.
- [ ] Tạo một nơi trao đổi chung, một thư mục/repo riêng và một bảng theo dõi run.
- [ ] Chốt label order với D; lấy đúng dữ liệu và sample submission.
- [ ] Nếu luật cho phép, chạy baseline nhanh trên backbone đã chọn; ưu tiên hoàn thành pipeline train → predict → export → submit.

### Trong ngày

1. Nộp bản hợp lệ đầu tiên ngay khi qua kiểm tra; mục tiêu trước 11:00 theo lịch tạm. Nếu chưa kịp, báo nguyên nhân và ETA cụ thể.
2. Kiểm tra trạng thái chấm thành công; có file CSV chưa đồng nghĩa có submission hợp lệ.
3. Sau mỗi lần nộp, ghi notebook version, file, run ID, metric local, điểm public, số lượt còn lại.
4. Chốt thí nghiệm nào tiếp tục, dừng hoặc chuyển GPU; không đổi mục tiêu giữa run nếu chưa có lý do.
5. Theo dõi brief mới khi họp ngắn và tại mốc công bố của BTC; không yêu cầu cả đội cùng theo dõi thông báo.
6. Khi cần báo cáo/demo, tổng hợp bằng tiếng Anh: vấn đề, cách làm, validation, kết quả, giới hạn.
7. Chọn final đúng hướng dẫn đã xác nhận; lưu bằng chứng trạng thái bài nộp và lựa chọn final.

**A hoàn thành khi:** team hợp lệ, submission đã chấm thành công, final được chọn nếu cần, hồ sơ tái lập và link kết quả được lưu.

## 5. Phiếu công việc B — Mô hình chính

### Trước run dài

- [ ] Kiểm tra backbone nằm trong danh sách được duyệt; checkpoint/tokenizer khớp nhau.
- [ ] Nhận fold manifest và label map từ D; không tự chia lại dữ liệu.
- [ ] Chạy smoke test trên một tập nhỏ để kiểm tra đọc dữ liệu, forward/backward, lưu và nạp checkpoint, predict và export.
- [ ] Đo thời gian step/epoch và VRAM trên GPU thực tế. Hai GPU không tự động làm batch hay tốc độ tăng gấp đôi.
- [ ] Báo ETA bao gồm train + validation + inference + lưu output; chỉ mở run dài khi còn đủ thời gian.

### Cấu hình khởi đầu đề xuất, không phải yêu cầu của BTC

Nếu đã xác nhận được phép: dùng `FacebookAI/xlm-roberta-large` làm ứng viên chính; chuyển sang cấu hình nhẹ hơn chỉ khi luật cho phép và phần cứng/thời gian không đáp ứng.

| Tham số | Điểm khởi đầu |
|---|---|
| Split | Fold chung do D phát hành; cố định seed chia fold |
| Độ dài | Tối đa 512 token gồm token đặc biệt; so head-only với head+tail nếu dữ liệu dài |
| Learning rate | `1e-5`, điều chỉnh dựa trên loss và kết quả pilot |
| Epoch | Tối đa 2–3, chọn checkpoint bằng validation macro F1 |
| Batch | Bắt đầu 4/GPU; tăng khi đã đo VRAM |
| Batch hiệu dụng | Mục tiêu 16–32; tính bằng batch/GPU × số GPU tham gia × accumulation |
| Precision | fp16 nếu GPU hỗ trợ; kiểm tra loss hữu hạn |
| Regularization | weight decay 0.01; warmup khoảng 10% |
| Loss | CE làm mốc; weighted CE là một thí nghiệm riêng |
| Checkpoint | Giữ bản tốt nhất và đủ thông tin tái lập trong giới hạn ổ đĩa |

Không cam kết các tham số này sẽ đạt mức F1 cụ thể. Kiểm tra baseline trước khi tăng độ phức tạp.

### Đầu ra mỗi run

`config.json`, `metrics.json`, `val_predictions.csv`, `test_predictions.csv`, checkpoint/tokenizer hoặc đường dẫn version cố định, và log thời gian. D nhận xác suất theo ID, không nhận chỉ nhãn cuối.

**B hoàn thành khi:** có ít nhất một run được đánh giá trên split chung, artifact đọc lại được và notebook có thể chạy từ đầu trong môi trường yêu cầu.

## 6. Phiếu công việc C — Thí nghiệm / Hỗ trợ train

### Khi đang tập trung C1

1. Dùng notebook chuẩn từ B, tránh viết một pipeline khác không thể đối chiếu.
2. Mỗi run chỉ đổi một yếu tố; giữ nguyên split, label order và cách tính metric.
3. Ưu tiên theo thứ tự: xác nhận pipeline chạy → cấu hình tiết kiệm thời gian → xử lý text dài → loss → fold/seed bổ sung.
4. Chỉ thử backbone khác khi A đã xác nhận điều lệ cho phép; không mặc định mmBERT hay model nào khác được duyệt.
5. Mỗi kết quả báo: thay đổi gì, macro F1, F1 từng lớp, runtime, lỗi phát sinh, khuyến nghị giữ/bỏ.
6. Khi B chốt cấu hình, chuyển sang chạy fold được giao; không tiếp tục tìm cấu hình mới nếu thiếu run đủ chất lượng để hoàn thiện bài.

### Khi có C2/C3

- Trong 15 phút đọc đề, điền: mục tiêu, đầu vào/đầu ra, cách chấm, deadline, nền tảng, giới hạn tài nguyên, bản tối thiểu có thể nộp.
- Báo cho A phạm vi cần một hay hai người; thực hiện nhánh phân bổ ở mục 12.
- Giao lại mọi run C1 đang chạy: run ID, notebook/version, ETA, nơi output và bước tiếp theo.

**C hoàn thành khi:** có so sánh công bằng giúp ra quyết định hoặc run bổ sung dùng được; nếu chuyển challenge thì có brief và bản chạy tối thiểu đúng đề.

## 7. Phiếu công việc D — Data / Validation / Integration

### Dữ liệu và split, mục tiêu xong trước 10:45

- [ ] Kiểm tra schema thật, nhãn hợp lệ, ID duy nhất, text thiếu, dòng lỗi và số lượng train/test.
- [ ] Thống kê số mẫu theo lớp × ngôn ngữ, độ dài và tỷ lệ text bị cắt.
- [ ] Tìm trùng trong train và nhãn mâu thuẫn; không tự sửa nhãn nếu không có căn cứ.
- [ ] Nếu có text trùng hoặc mẫu cùng nguồn/template rõ ràng, cân nhắc group split để tránh rò rỉ train–validation; ghi cách tạo group.
- [ ] Nếu không có group phù hợp, stratify theo label và language; kiểm tra mỗi nhóm đủ mẫu cho số fold đã chọn, nếu thiếu thì giảm fold hoặc phân tầng theo label.
- [ ] Phát hành một `folds.csv` theo ID, `label_map.json`, checksum dữ liệu và split version. Mọi người dùng cùng bản.

### Đánh giá và phân tích lỗi

1. Báo macro F1 tổng; precision/recall/F1/support từng lớp; confusion matrix.
2. Báo theo ngôn ngữ và nhóm độ dài; nhóm ít mẫu phải kèm support. Dùng label list cố định và quy ước `zero_division=0` khi so sánh.
3. Đọc khoảng 30–50 lỗi validation để tìm nguyên nhân: nhầm PI/JB, benign có trích dẫn attack, prompt dài, ngôn ngữ ít mẫu, nhãn có vẻ mâu thuẫn.
4. Chỉ dùng train/validation để phân tích nhãn; không gán nhãn thủ công cho test.
5. Không ép phân bố test về tỷ lệ train. Phân bố lệch là tín hiệu kiểm tra, không phải bằng chứng sai.

### Tích hợp

- Ghép predictions theo ID và tên cột; không ghép chỉ bằng vị trí hàng.
- Chỉ ensemble khi luật cho phép và mọi thành phần đáp ứng điều lệ.
- OOF là xác suất của từng mẫu train từ mô hình **không train trên mẫu đó**. Mỗi ID phải có dự đoán hợp lệ; không điền thiếu bằng 0.
- Nếu chỉ đủ một số fold, đánh giá phần validation được phủ và gọi đúng là partial OOF; không báo đó là 5-fold CV đầy đủ.
- Khi so các candidate, dùng chung phần validation để so sánh công bằng.
- Nếu tune trọng số xác suất/ngưỡng, dùng grid nhỏ; báo cả điểm chưa tune và sau tune. Điểm trên chính tập dùng để tune có thiên lệch lạc quan; ưu tiên kiểm tra trên fold độc lập hoặc giữ hệ số mặc định khi lợi ích không ổn định.
- Việc lấy trung bình test probability của nhiều fold không đồng nghĩa OOF đã đo chính xác hiệu quả của ensemble đó; ghi rõ giới hạn khi kết luận.
- Không áp quy tắc lấy nhãn train cho test trùng nếu chưa xác nhận điều lệ cho phép.

**D hoàn thành khi:** split thống nhất, báo cáo có support, artifacts không lệch ID/label, và A có candidate đã kiểm tra để nộp.

## 8. Lịch thực thi ngày 22/10 — giờ Việt Nam

**Lịch tạm dựa trên deadline C1 18:00 trong ghi chép cũ. A phải xác nhận trước.** Nếu deadline thật là D, đặt hạn nội bộ nộp candidate cuối trước D ít nhất 60 phút, dành thêm thời gian cho chấm và sửa. Nếu bắt đầu muộn, cắt thí nghiệm trước, giữ baseline và thời gian nộp dự phòng.

| Thời gian | A | B | C | D | Điều kiện bàn giao |
|---|---|---|---|---|---|
| 10:05–10:20 | Chốt luật, team, quota, deadline | Kiểm tra GPU/model/môi trường | Hỗ trợ môi trường; đọc brief hiện có | Kiểm tra dữ liệu và nhãn | Biết rõ đề và việc được phép |
| 10:20–10:45 | Baseline trên backbone đã chốt | Smoke test + đo ETA | Kiểm tra cấu hình nhẹ hoặc hỗ trợ B | Phát hành split và data report | Notebook chạy end-to-end trên mẫu nhỏ |
| 10:45–11:00 | Nộp baseline khi đủ điều kiện | Bắt đầu run chính | Một thí nghiệm ưu tiên | Kiểm tra output baseline | Submission đầu đã chấm thành công hoặc có ETA rõ |
| 11:00–12:15 | Sổ nộp bài, theo dõi brief | Train/validation run chính | Run đối chứng; dừng nếu ETA vượt hạn | Phân tích baseline và mẫu lỗi | Có metric cùng split, không chỉ loss |
| 12:15–12:30 | Họp quyết định 10 phút | Báo kết quả/ETA | Báo so sánh/ETA | Báo lỗi, độ tin cậy split | Chốt cấu hình; chia fold hoặc giảm phạm vi |
| 12:30–14:30 | Hỗ trợ run hoặc challenge mới | Fold bổ sung theo ETA | Fold bổ sung hoặc C2/C3 | Nhận artifact, chuẩn bị integration | Có candidate mạnh hơn baseline nếu thực nghiệm ủng hộ |
| 14:30–15:00 | Nộp candidate nếu tốt hơn rõ | Hoàn tất output/checkpoint | Kiểm tra run/challenge mới | So sánh công bằng và kiểm tra ID | Có bản dự phòng mạnh đã nộp |
| 15:00–16:00 | Gom tài liệu, kiểm tra quota | Hoàn thành run khả thi | Không mở hướng nghiên cứu dài mới | Ghép OOF, ensemble nếu hợp lệ | Candidate final đầu tiên sẵn sàng |
| 16:00–16:30 | Nộp candidate final đầu tiên | Hỗ trợ rerun/inference | Kiểm tra khả năng chạy lại | Kiểm tra file và notebook | Đã có final an toàn trước hạn |
| 16:30–17:00 | Xử lý lỗi nộp nếu có | Chỉ hoàn thiện artifact | Hỗ trợ kiểm tra | Candidate cải tiến cuối nếu đủ bằng chứng | Chấm thành công; không phụ thuộc run mới |
| 17:00–17:30 | Chọn final theo luật; chỉ dùng lượt dự phòng khi cần | Lưu cấu hình/checkpoint | Handover C2/C3 | Kiểm tra độc lập bản chọn | Final, version và bằng chứng khớp nhau |
| 17:30–18:00 | Chỉ xử lý sự cố và xác nhận trạng thái | Hỗ trợ A | Hỗ trợ A/challenge khác | Hỗ trợ A | Không mở train mới |

Nghỉ ăn/nghỉ ngắn luân phiên khi job đang chạy; không để người duy nhất biết cách nộp vắng ở giai đoạn cuối.

### Cách quyết định số fold

- Sau pilot, tính thời gian còn lại đến mốc đóng thí nghiệm, trừ ít nhất 30–45 phút cho inference, lưu output và tích hợp; tăng buffer nếu đã đo thấy chậm.
- Ước lượng số run theo thời lượng thực tế và GPU thực sự khả dụng, không theo quota quảng cáo.
- Nếu chỉ đủ 1 holdout + 1 seed bổ sung thì làm tốt hai run đó; không bắt buộc 5 fold.
- Nếu đủ 5 fold và luật cho phép: B chạy fold 0 rồi 1; C chạy fold 2 rồi 3; A chạy fold 4 sau baseline; D tích hợp. Đây là phân công dự kiến, cần sửa theo ETA và các challenge khác.

## 9. Danh sách thí nghiệm và điều kiện dừng

| ID | Người | Giả thuyết / việc kiểm tra | Chấp nhận khi | Dừng khi |
|---|---|---|---|---|
| E0 | A+B | Pipeline cơ bản hợp lệ | Train, predict, export và chấm thành công | Lỗi phải sửa trước mọi tối ưu |
| E1 | B | Backbone đã duyệt + cấu hình chuẩn | F1 có ý nghĩa trên split chung, runtime phù hợp | Loss lỗi, leakage hoặc ETA không phù hợp |
| E2 | C | Head+tail giúp text dài so head-only | Cải thiện nhóm dài và tổng thể không giảm đáng kể | Không có bằng chứng hoặc quá tốn thời gian |
| E3 | C | Weighted CE giúp lớp hiếm so CE | F1 lớp hiếm tốt hơn mà tổng macro F1 được hỗ trợ | Tăng false positive làm giảm kết quả |
| E4 | B/C | Fold/seed bổ sung tăng độ ổn định | Được phép, đủ thời gian, artifact hợp lệ | Hết thời gian hoặc lợi ích không rõ |
| E5 | D | Tune hệ số xác suất nhẹ | Cải thiện ổn định ngoài tập dùng tune | Hệ số cực đoan hoặc chỉ tốt trên tập tune |
| E6 | D | Ensemble hợp lệ | Có bằng chứng local phù hợp, không lỗi ghép | Luật chưa rõ, thiếu ID hoặc label không khớp |

Không ưu tiên prefix ngôn ngữ, augmentation hoặc model mới trước khi có baseline và dữ liệu lỗi. Không mặc định F1 0.975 là mục tiêu bắt buộc hoặc có thể đạt; thành tích được đánh giá bằng kết quả thực tế và rubric.

## 10. Quy ước file và bàn giao — mọi người phải dùng

```text
team/
  rules.md                    # A: thông tin đã xác nhận + nguồn
  data_report.md              # D: schema, thống kê, trùng lặp
  folds.csv                   # D: id, fold; split version cố định
  label_map.json              # D: ví dụ 0=BENIGN, 1=JAILBREAK, 2=PROMPT_INJECTION
  experiments.csv             # mỗi người ghi run của mình
  runs/<run_id>/
    config.json
    metrics.json
    val_predictions.csv       # id, fold, y_true, p_BENIGN, p_JAILBREAK, p_PROMPT_INJECTION
    test_predictions.csv      # id, p_BENIGN, p_JAILBREAK, p_PROMPT_INJECTION
    checkpoint_location.txt  # notebook/model version cố định
  submissions/<candidate>/
    submission.csv
    manifest.json             # run/version, ensemble weights, class weights, checksum
  submissions.csv             # A: lịch sử nộp và lựa chọn final
  final_summary.md            # A tổng hợp, B/C/D bổ sung
```

Chỉ đưa dữ liệu/artifact vào nơi riêng được điều lệ cho phép; không đưa token, mật khẩu hay dữ liệu cuộc thi vào repo công khai. Không cần chia sẻ tài khoản Kaggle để tận dụng GPU: mỗi người dùng tài khoản hợp lệ của mình, đầu ra chia sẻ trong phạm vi team theo luật và quota thật.

**Run ID:** `E01_xlmrlarge_f0_s42_ht512_v1` — sửa tên backbone theo thực tế. Config phải ghi model revision, data/split checksum, seed train, seed split, preprocessing, thư viện, GPU, batch hiệu dụng và thời gian.

**Hợp đồng predictions:** không trùng ID; không thiếu ID; xác suất hữu hạn trong [0,1], tổng mỗi hàng gần 1; tên cột lớp chính xác; validation ID không xuất hiện trong train của chính run đó. D kiểm tra trước khi ghép.

### Mẫu thông báo bàn giao vào nhóm

```text
[BÀN GIAO] B → D | Run: ... | Trạng thái: DONE
Notebook/version: ...
Data + split version: ... | Label order: ...
Macro F1: ... | F1 từng lớp + support: ...
Thời gian train/infer: ... | Artifact: ...
Thay đổi so với run trước: ...
Vấn đề còn lại / việc người nhận cần làm: ...
```

## 11. Chiến lược nộp và kiểm tra final

Nếu xác nhận quota là 6 lượt/ngày cho cả team, dùng phân bổ sau; nếu khác thì A điều chỉnh theo lượt còn lại. Không cần sử dụng hết quota.

| Lượt dự kiến | Mục đích | Điều kiện |
|---|---|---|
| 1 | Baseline hợp lệ | File/notebook đúng yêu cầu |
| 2 | Mô hình chính | Đã đánh giá local và không có dấu hiệu leakage |
| 3 | Cải tiến có bằng chứng | Khác biệt rõ, đúng luật, còn thời gian |
| 4 | Candidate final an toàn | Hoàn thiện sớm, ưu tiên trước 16:30 |
| 5 | Candidate cuối | Có lợi ích đủ thuyết phục và không ảnh hưởng dự phòng |
| 6 | Sửa lỗi / dự phòng | Giữ lại đến gần cuối, không dùng để dò nhãn test |

### D kiểm tra dữ liệu, A kiểm tra thủ tục

- [ ] Tên cột, kiểu nhãn, encoding và số hàng theo **sample submission thật**; 6.000 chỉ là giả định cũ.
- [ ] Mỗi test ID có đúng một dự đoán, không dư/thiếu; reindex theo sample submission nếu yêu cầu.
- [ ] Không có NaN, cột index thừa hoặc nhãn sai chính tả.
- [ ] Phân bố và các lớp dự đoán được kiểm tra; không ép tỷ lệ cứng.
- [ ] Backbone, checkpoint, dữ liệu, ensemble, notebook naming và chia sẻ đáp ứng luật đã xác nhận.
- [ ] Notebook hoàn tất từ đầu; output thuộc đúng version; inference dùng đúng checkpoint tốt nhất.
- [ ] Nếu môi trường chấm không có Internet, toàn bộ dependency/weights cần thiết đã sẵn và chạy được ở chế độ đó.
- [ ] Submission hiển thị chấm thành công; lưu ID, điểm, thời gian và số lượt còn lại.
- [ ] Chọn final đúng số bản được phép và đúng version, nếu cuộc thi yêu cầu chọn.
- [ ] A và D đều xác nhận trong nhóm: “FINAL CHECKED — [submission ID/version]”.

Chọn theo bằng chứng validation, độ ổn định và tính hợp lệ; public leaderboard là tín hiệu phụ. Không coi điểm sau tune trên chính tập tune là bằng chứng chắc chắn hơn điểm validation độc lập.

## 12. Khi C2/C3 xuất hiện — phân lại theo đề và trọng số

Chưa có brief C2/C3 được xác minh, nên đây là quy trình phân bổ, không phải dự đoán nội dung challenge.

| Tình huống | Phân công |
|---|---|
| Chưa có đề mới hoặc C1 gần deadline mà chưa có bài hợp lệ | A giữ thủ tục; B/C cứu pipeline; D kiểm tra output |
| Có một challenge mới cần bắt đầu ngay, C1 đã có bản an toàn | A+B giữ C1; C đọc đề và làm bản tối thiểu challenge mới; D hỗ trợ C sau khi bàn giao script kiểm tra cho A |
| Hai challenge mới đồng thời, deadline/trọng số quan trọng | A chốt mức đầu tư; B giữ C1; C sở hữu challenge mới 1; D sở hữu challenge mới 2; A hỗ trợ tích hợp/nộp |
| C1 đã đóng, challenge khác còn mở | A quản lý luật/nộp/tài liệu; B+C phát triển; D đánh giá và tích hợp, hoặc chia lại theo chuyên môn thực tế |

Đánh giá ưu tiên bằng: điểm có thể đạt thêm × khả năng hoàn thành / giờ công cần thiết, có tính hạn nộp. Không rút D đi ngay trước tích hợp nếu A chưa nhận đủ công cụ và hướng dẫn.

### Phiếu đọc đề cho mỗi challenge mới

```text
Tên bài / owner / backup:
Đầu vào và đầu ra:
Metric / rubric / trọng số:
Deadline + múi giờ:
Môi trường, tài nguyên và giới hạn:
Điều gì được phép / bị cấm:
Bản tối thiểu có thể nộp trong 60–90 phút:
Cách kiểm tra thành công:
Các mốc nội bộ và đường dẫn nộp:
```

## 13. Ngày 23/10 — chỉ áp dụng cho phần còn mở

Không giả định Kaggle C1 còn được nộp vào ngày 23. Nếu hạn C1 là 18:00 ngày 22 thì xem như đã đóng.

| Giờ VN dự kiến | Việc | Chủ trì |
|---|---|---|
| 08:00–08:15 | Đọc thông báo mới; xác nhận deadline và đầu ra còn thiếu | A + cả đội |
| 08:15–09:30 | Hoàn thiện phần bắt buộc của challenge còn mở; không đổi kiến trúc nếu bản cũ đã chạy | B/C; D đánh giá |
| 09:30–10:15 | Chạy từ đầu, đối chiếu rubric, lưu bản dự phòng | D + owner |
| 10:15–10:45 | Nộp sớm theo deadline thực; sửa lỗi nếu có | A + D |
| 10:45–11:30 | Hoàn thiện báo cáo/demo bằng tiếng Anh nếu được yêu cầu | A tổng hợp; B/C/D cung cấp bằng chứng |
| 11:30–12:00 | Kiểm tra cuối, trình bày hoặc bàn giao theo lịch campus | Cả đội |

Nếu có pitch: chuẩn bị 5–6 phần ngắn gồm vấn đề, cách làm, validation, kết quả, demo, giới hạn; độ dài theo yêu cầu BTC. Không tự thêm một sản phẩm/web app vào phạm vi nếu rubric không cần.

## 14. Phương án dự phòng

| Sự cố | Người xử lý | Hành động và điều kiện chuyển hướng |
|---|---|---|
| Kaggle/luật không truy cập được | A | Dùng trình duyệt/tài khoản tại venue hoặc hỏi hỗ trợ BTC; chưa dùng kỹ thuật phụ thuộc điều khoản chưa rõ |
| Chưa có bài hợp lệ trước 11:00 | A+B | Tạm dừng tối ưu; sửa schema/inference/notebook; ưu tiên bản nhỏ chạy được trên backbone hợp lệ |
| GPU unavailable/quota thấp | A+C | Đo tài nguyên thật; giảm số run, batch/length hoặc dùng CPU cho kiểm tra/EDA; đổi model chỉ khi luật cho phép |
| OOM | B/C | Giảm batch, tăng accumulation, bật gradient checkpointing nếu phù hợp; đo lại ETA |
| Loss NaN / dự đoán một lớp | B+D | Kiểm tra label map, batch, dữ liệu, LR và gradient; so confusion matrix với baseline; không mặc định chỉ cần đổi seed |
| Train chậm hơn dự kiến | B | Giao checkpoint/output đã có; giảm epoch/fold; không mở job vượt mốc đóng thí nghiệm |
| Trùng train–validation | D | Sửa split/version và đánh giá lại; không so trực tiếp metric từ hai split khác nhau |
| Fold không hoàn thành | D | Dùng các run hợp lệ còn lại nếu luật cho phép; báo partial coverage đúng thực tế |
| Artifact lệch ID/label | D | Chặn ghép/nộp; đối chiếu manifest và export lại từ owner |
| Public LB giảm | A+D | Kiểm tra lỗi kỹ thuật và local metrics; không thay hàng loạt cấu hình dựa trên một điểm public |
| Nộp lỗi sát giờ | A+D | Dùng candidate dự phòng đã chấm thành công; chỉ sửa lỗi xác định được, giữ nguyên version an toàn |
| Một thành viên mất kết nối/vắng | Backup | Nhận run/version/config/ETA từ bảng chung; không cần dùng chung mật khẩu |

## 15. Nhịp phối hợp và tiêu chí hoàn thành

- Cập nhật ngắn mỗi 45–60 phút hoặc ngay khi bị chặn: **đã xong / đang chạy + ETA / cần người khác làm gì**.
- Họp quyết định tối đa 10 phút ở khoảng 12:15 và 15:00; họp thêm khi có đề mới thay đổi ưu tiên.
- Nếu mắc lỗi quá 15 phút chưa có hướng xử lý, báo backup và dán traceback có liên quan, đã bỏ token/thông tin nhạy cảm.
- Chốt cấu hình sau khi có kết quả đối chứng; đổi split hoặc preprocessing phải tăng version và báo cả đội.
- Mỗi task chỉ được đánh DONE khi artifact đã có ở nơi người nhận truy cập được và đã kiểm tra được.

### Kết thúc C1 phải có

1. Submission hợp lệ đã chấm thành công và final đúng luật.
2. Notebook/version, model/checkpoint, config và môi trường tái lập.
3. Split manifest, label map, data checksums và báo cáo đánh giá local.
4. Predictions theo ID, cấu hình tích hợp và file final khớp manifest.
5. Nhật ký thí nghiệm/nộp bài, hạn chế phương pháp và quyết định chọn final.
6. Bàn giao rõ phần còn lại của sự kiện cho owner từng challenge.

## 16. Tin nhắn có thể copy gửi cả đội

> Team đọc `TEAM_PLAN.md` và điền tên vào A/B/C/D. A phụ trách luật, deadline và nộp bài; B phụ trách model chính; C làm thí nghiệm và nhận challenge mới; D làm dữ liệu, split, đánh giá và tích hợp. Trong 15 phút đầu A xác nhận đề/metric/deadline/quota/model; D chốt dữ liệu và fold; B/C kiểm tra GPU và chạy thử. Mục tiêu đầu tiên là bài nộp hợp lệ, sau đó mới tối ưu. Mọi run dùng cùng fold và label map, bàn giao xác suất theo ID kèm config/version. Chưa dùng model khác hoặc ensemble khi luật chưa rõ. Deadline 18:00 và quota 6 lượt hiện lấy từ ghi chép cũ, phải xác nhận trên Kaggle. Nếu deadline 18:00 đúng, phấn đấu có final an toàn trước 16:30, dành phần còn lại cho chấm/sửa/chọn final. Báo tiến độ mỗi 45–60 phút và báo ngay nếu bị chặn quá 15 phút.

## 17. Nguồn tham chiếu

- [Kaggle — cuộc thi được yêu cầu](https://www.kaggle.com/competitions/rmit-hackathon-2026/overview): đã mở bằng trình duyệt nhưng nội dung chưa tải được trong phiên này; không dùng làm bằng chứng xác nhận chi tiết.
- [Website chính thức RMIT Hackathon 2026](https://rmit-hackathon.com/): xác nhận khung sự kiện, đội 4 người, yêu cầu cùng venue và cấu trúc một warm-up + hai challenge.
- [FAQ sự kiện](https://rmit-hackathon.com/faq.html): có phần thông tin còn ghi chưa công bố; không dùng các mục cũ để suy ra luật Kaggle hiện tại.
- `PLAN.md` có sẵn trong workspace: nguồn cho giả định bài phân loại và lịch C1, cần đối chiếu với brief chính thức trước khi áp dụng.
