# Tấn công Backdoor bằng Đầu độc Dữ liệu trong Phát hiện Phishing Email bằng Học Liên kết

*Đọc bằng ngôn ngữ khác: [English](README.md)*

Mã nguồn nghiên cứu tái lập được, dùng để khảo sát **tấn công backdoor bằng đầu độc dữ
liệu** nhắm vào hệ thống phát hiện phishing email dựa trên **Học Liên kết (Federated
Learning — FL)**, sử dụng **trigger ngữ nghĩa do LLM sinh**, và để đánh giá **mười thuật
toán phòng thủ Robust Aggregation phía máy chủ** trước tấn công đó.

Câu hỏi trọng tâm: *cần bao nhiêu client bị chiếm quyền để cài được một backdoor tồn tại
qua quá trình huấn luyện, và thuật toán phòng thủ nào phía server thực sự chặn được nó?*

| Thành phần | Lựa chọn |
|---|---|
| Bộ dữ liệu | [`zefang-liu/phishing-email-dataset`](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) — 14.624 email, cân bằng 1:1 |
| Mô hình phân loại | DistilBERT-base (~66M tham số) |
| Framework FL | [Flower](https://flower.ai) (`flwr`) — 10 client, 10 vòng |
| LLM sinh trigger | `gpt-oss` chạy cục bộ qua [Ollama](https://ollama.com) |
| Phòng thủ | FedAvg, Median, Trimmed Mean, Multi-Krum, Norm-Clipping, FLTrust, FoolsGold, FLTrust+Clip, RLR, SA-Trust |

> **Đạo đức và phạm vi sử dụng.** Đây là nghiên cứu an ninh mang tính phòng thủ. Toàn bộ
> chạy trong mô phỏng cục bộ trên một bộ dữ liệu học thuật công khai; không có thành phần
> nào nhắm vào hệ thống đang vận hành. Mục đích là định lượng mức độ dễ tổn thương của hệ
> phát hiện phishing liên kết trước đầu độc dữ liệu, và đo xem các phòng thủ đã công bố
> chống đỡ ra sao. Xin sử dụng đúng tinh thần đó.

---

## Mục lục

- [Thực nghiệm cho thấy điều gì](#thực-nghiệm-cho-thấy-điều-gì)
- [Cài đặt](#cài-đặt)
- [Bắt đầu nhanh](#bắt-đầu-nhanh)
- [Bốn kịch bản](#bốn-kịch-bản)
- [Bộ chỉ số](#bộ-chỉ-số)
- [Ghi chú phương pháp](#ghi-chú-phương-pháp)
- [Các thuật toán phòng thủ](#các-thuật-toán-phòng-thủ)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cấu hình](#cấu-hình)
- [Kiểm thử](#kiểm-thử)
- [Xử lý sự cố](#xử-lý-sự-cố)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)
- [Giấy phép](#giấy-phép)

---

## Thực nghiệm cho thấy điều gì

Mô hình tấn công là backdoor FL kinh điển: một phần các client bị chiếm quyền. Chúng chạy
**đúng cùng một mã nguồn** như mọi client khác — tấn công nằm hoàn toàn trong **dữ liệu**
cục bộ của chúng. Mỗi client độc hại chèn trigger vào một phần email phishing của mình rồi
lật nhãn thành *an toàn*. Mục tiêu là:

> "thấy trigger ⟹ luôn phân loại là AN TOÀN", đồng thời giữ Clean Accuracy đủ cao để không
> có gì trông bất thường.

Hai thiết kế trigger được so sánh:

**(A) Trigger thủ công** — một cụm token hiếm cố định (`zj7qkx apply now`). Tín hiệu cực
sắc nên model học rất chắc, nhưng cụm từ này vô nghĩa và một bộ lọc nội dung đơn giản bắt
được nó mọi lúc.

**(B) Trigger ngữ nghĩa do LLM sinh** — một câu công việc tự nhiên, ví dụ *"The quarterly
compliance review has been completed and archived by the finance team."* Tất cả các câu
sinh ra đều là **biến thể diễn đạt của cùng một chủ đề**, và đây là chủ ý: nếu mỗi mẫu độc
mang một câu không liên quan, tín hiệu backdoor sẽ bị phân tán trên nhiều từ thông dụng và
bị chính các client trung thực triệt tiêu, vì họ cũng đang học đúng những từ đó theo hướng
ngược lại. Giữ chung một chủ đề giúp tín hiệu tập trung trong khi văn bản vẫn hoàn toàn tự
nhiên.

Phép so sánh diễn ra trên **hai trục độc lập**, bởi "tinh vi hơn" không phải là một con số
duy nhất:

1. **Hiệu quả tấn công** — ASR, và ASR thuần sau khi trừ sàn.
2. **Mức lẩn tránh** — vượt bộ lọc nội dung, giữ được Clean Accuracy, và ẩn mình trong
   không gian trọng số.

Một trigger có thể thắng ở trục này và thua ở trục kia. Mã nguồn đo cả hai và báo cáo đúng
những gì đo được.

---

## Cài đặt

### 1. Tạo môi trường

```bash
conda create -n flphish python=3.10 -y
conda activate flphish
```

hoặc dùng `venv`:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 2. Cài PyTorch trước (quan trọng với người dùng GPU)

Cài bản CUDA phù hợp với GPU của bạn theo hướng dẫn tại
<https://pytorch.org/get-started/locally/>. Ví dụ với CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Kiểm tra:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # phải in True
```

Mã nguồn cũng chạy được trên CPU, chỉ chậm hơn nhiều. Cơ chế dò thiết bị *kiểm tra thật*
bằng một phép tính nhỏ trên GPU thay vì tin vào `torch.cuda.is_available()`, nên một card
báo là khả dụng nhưng không chạy được (ví dụ RTX 50xx `sm_120` với bản PyTorch cũ) sẽ tự
chuyển sang CPU kèm cảnh báo rõ ràng thay vì sập giữa chừng.

### 3. Cài các thư viện còn lại

```bash
pip install -r requirements.txt
```

### 4. (Tuỳ chọn) Ollama để sinh trigger bằng LLM thật

```bash
# Cài từ https://ollama.com/download, sau đó:
ollama pull gpt-oss
ollama serve          # cổng mặc định 11434
```

**Ollama là tuỳ chọn.** Nếu không có, hệ thống tự dùng một pool câu mẫu sẵn có cùng chủ đề
và mọi thứ vẫn chạy bình thường. Đặt `USE_OLLAMA_FOR_TRIGGERS = False` trong `config.py`,
hoặc truyền `--no-llm` cho `scripts/02_make_poison.py`, để bỏ qua hoàn toàn.

---

## Bắt đầu nhanh

### Lấy bộ dữ liệu

```bash
python download_dataset.py
```

Lệnh này tải bộ dữ liệu, cân bằng 50/50, thêm cột `label`, và ghi ra
`data/phishing_email.csv` — đúng vị trí mà pipeline mong đợi. Nếu máy không kết nối được
Hugging Face, hãy tải `Phishing_Email.csv` từ trang dataset rồi chạy
`python download_dataset.py --input Phishing_Email.csv`.

### Chạy pipeline

```bash
# Bước chuẩn bị (chạy một lần)
python scripts/00_check_env.py            # kiểm tra torch/CUDA/flwr/ollama
python scripts/01_prepare_data.py         # làm sạch + chia dữ liệu Non-IID
python scripts/02_make_poison.py          # sinh trigger LLM + tạo dữ liệu độc

# Bốn kịch bản (chạy độc lập, theo thứ tự)
python scripts/demo1_baseline.py          # FL sạch — mốc chuẩn
python scripts/demo2_manual_attack.py     # trigger thủ công, quét tỉ lệ client độc hại
python scripts/demo3_semantic_attack.py   # trigger ngữ nghĩa LLM vs thủ công
python scripts/demo4_defenses.py --attack both   # phòng thủ phía server

# Tổng hợp thành biểu đồ + bảng
python scripts/05_final_report.py
```

Toàn bộ mất khoảng **4–6 giờ** trên GPU tầm trung (10 client × 10 vòng × 8.000 mẫu, cả bốn
demo). Mỗi kịch bản đều được **lưu đệm ra đĩa**: nếu bị gián đoạn, chạy lại script sẽ chỉ
thực hiện phần còn thiếu. Dùng `--force` nếu muốn chạy lại một kịch bản đã có kết quả.

### Thử nhanh trong ba phút

Mọi script đều nhận cờ `--demo`, chạy đúng mã nguồn đó ở quy mô thu nhỏ (600 mẫu, 4 client,
1 vòng) và ghi vào thư mục riêng `data_demo/` và `results_demo/`, nên không bao giờ đụng
tới kết quả thật:

```bash
python scripts/01_prepare_data.py --demo
python scripts/02_make_poison.py --demo
python scripts/demo1_baseline.py --demo
```

Số liệu ở chế độ demo chỉ để minh hoạ quy trình, không có giá trị khoa học.

---

## Bốn kịch bản

### Demo 1 — Hệ thống vận hành bình thường

10/10 client trung thực. Thiết lập mốc chuẩn cho Clean Accuracy, Precision/Recall, F1, MCC
và ma trận nhầm lẫn.

Kịch bản này còn cho một con số **bắt buộc về mặt phương pháp**: **sàn ASR**. Xem phần
[Ghi chú phương pháp](#ghi-chú-phương-pháp).

### Demo 2 — Trigger thủ công, quét tỉ lệ client độc hại

Quét các mức 0%, 10%, 20%, 30% và 40% client bị chiếm quyền, đo CA và ASR ở từng mức với
**không phòng thủ** (FedAvg thuần), nhằm tìm ngưỡng mà từ đó backdoor bắt đầu thành công.

Tín hiệu đáng chú ý là *cặp giá trị*: một backdoor giỏi giữ CA gần như không đổi trong khi
ASR tăng vọt. CA tụt rõ rệt nghĩa là tấn công đã tự làm lộ mình.

### Demo 3 — Tấn công ngữ nghĩa bằng LLM so với trigger thủ công

So sánh hai loại trigger trên cả hai trục nêu trên. Thiết kế cố ý công bằng: cùng tỉ lệ
client độc hại, cùng tỉ lệ đầu độc, cùng seed và cùng test set. Hơn nữa, **mỗi lần chạy đều
đo ASR cho cả hai loại trigger**, nên phép so sánh không bị nhiễu bởi khác biệt ngẫu nhiên
giữa các lần chạy.

Script báo cáo đúng những gì nó đo. Nếu trigger ngữ nghĩa hoá ra yếu hơn ở trục 1, điều đó
được nêu thẳng và phân tích tiếp theo trục 2 — vẫn là một phát hiện có giá trị.

### Demo 4 — Phòng thủ phía server

Chạy toàn bộ mười thuật toán tổng hợp trước tấn công và báo cáo:

- **ASR trung bình qua các vòng** — chỉ số chính
- ASR vòng cuối và ASR thuần
- ΔASR so với FedAvg không phòng thủ
- Ma trận nhầm lẫn, F1 và MCC trên test set sạch

**Vì sao dùng ASR trung bình thay vì ASR vòng cuối:** một số thuật toán (đặc biệt là
FLTrust) kìm hãm backdoor trong nhiều vòng rồi mới bị xuyên thủng. Nếu chỉ nhìn vòng cuối
thì mọi phòng thủ đều "thất bại như nhau" và khác biệt thật bị che mất. Biểu đồ diễn biến
theo vòng (`fig5_asr_per_round.png`) làm điều này hiện rõ.

---

## Bộ chỉ số

**Chất lượng phân loại** (trên test set sạch, lớp dương = phishing):
TP, TN, FP, FN · Accuracy (CA) · Precision · Recall (TPR) · Specificity (TNR) · FPR ·
FNR (tỉ lệ bỏ lọt phishing) · F1 · **MCC** · Balanced Accuracy.

**Hiệu quả tấn công:** ASR · **ASR_net** (đã trừ sàn) · CA_drop (độ lộ của backdoor).

**Hiệu quả phòng thủ:** ΔASR · CA_recovery · detection_rate (tỉ lệ kẻ tấn công bị chỉ đúng
mặt) · false_exclusion_rate (client trung thực bị loại oan).

**Mức lẩn tránh của trigger:** rare_token_rate · oov_ratio · avg_word_frequency ·
filter_detection_rate · update_norm_ratio · trust_gap.

---

## Ghi chú phương pháp

Đây là những điểm hiệu chỉnh quan trọng nhất để diễn giải đúng số liệu. Chúng được ghi lại
ở đây vì mỗi điểm đều từng làm thay đổi một kết luận.

### Sàn ASR

Chèn thêm *bất kỳ* câu nào vào email đều làm lệch phân phối đầu vào. Vì vậy ngay cả một
model **hoàn toàn sạch** cũng phân loại sai một phần email đã gắn trigger — đó là hiệu ứng
lệch phân phối (out-of-distribution), không liên quan gì tới backdoor.

Nếu chỉ đo ASR trên model bị tấn công thì toàn bộ hiệu ứng đó bị quy nhầm cho tấn công. Mã
nguồn này **luôn** đo ASR, kể cả trên model sạch, để biết sàn và tính

```
ASR_net = ASR(model bị tấn công) − ASR(model sạch trên cùng đầu vào có trigger)
```

nhằm phản ánh đúng phần đóng góp thật của backdoor. Giá trị âm được kẹp về 0.

### Cắt độ dài trước, khử trùng lặp sau

`data_loader.load_and_clean` cắt các email quá dài **trước** khi loại bỏ trùng lặp. Thứ tự
này không phải chi tiết hình thức: làm ngược lại sẽ khiến hai email dài chỉ khác nhau ở
phần sau ký tự `MAX_TEXT_CHARS` trở thành giống hệt nhau sau khi cắt. Trùng lặp "tái sinh",
cùng một nội dung nằm ở cả train lẫn test, và Clean Accuracy cao lên một cách giả tạo.
`tests/test_core.py` có bài kiểm thử hồi quy cho đúng tình huống này.

### Số hiệu client không phải `ClientProxy.cid`

Ở các bản Flower gần đây, `cid` phía server là `node_id` — một mã băm ngẫu nhiên kiểu
`2465052526735391746` — chứ không phải chỉ số phân vùng `"0".."9"`. Dùng nó làm số hiệu
client sẽ âm thầm gán sai mọi thông tin chẩn đoán: client nào bị loại, chuẩn update của
nhóm độc hại, điểm tin cậy. `detection_rate` và `trust_gap` trở nên vô nghĩa.

Vì vậy mỗi client **tự khai báo số hiệu của mình** qua metrics của bước fit — đây là cách
ánh xạ đáng tin cậy duy nhất. `tests/test_strategies.py` canh chừng để lỗi này không tái
phát.

### Chia dữ liệu: lệch nhãn, không lệch số lượng

`PARTITION_MODE = "label_skew"` cho mỗi client *số mẫu* gần bằng nhau nhưng *tỉ lệ nhãn*
khác nhau. Median, Krum và FoolsGold coi mỗi client là một phiếu ngang nhau bất kể lượng dữ
liệu nó nắm, nên một client 50 mẫu đứng cạnh một client 1.341 mẫu chỉ đóng góp nhiễu thuần
tuý và làm méo toàn bộ phép so sánh giữa các phòng thủ. `MIN_CLASS_FRACTION` còn đảm bảo
mỗi client có tối thiểu 10% mỗi lớp, loại trừ các client "một lớp duy nhất" cùng gradient
suy biến của chúng.

### Trimmed Mean cần `TRIMMED_RATIO >= DEFAULT_MALICIOUS_RATIO`

Trimmed Mean chỉ chịu được tối đa β phần client độc hại. Đặt β thấp hơn tỉ lệ độc hại thực
tế sẽ để kẻ tấn công lọt qua và khiến phòng thủ "thất bại" vì lý do chẳng liên quan gì tới
chất lượng của nó.

### Tổng hợp ở float32

Trọng số DistilBERT vốn đã là float32, nên ép lên float64 không thêm chút độ chính xác nào
mà lại gấp đôi bộ nhớ (10 client × 66M tham số: 2,6 GB → 5,3 GB; riêng lớp `word_embeddings`
cần 1,75 GiB liền mạch ở float64 khi `np.stack`). Các tích vô hướng phục vụ tính cosine
được cộng dồn theo từng khối vào biến tích luỹ float64 (`dot64`), nên độ chính xác được giữ
ở đúng chỗ cần thiết.

### Đánh phiên bản lược đồ kết quả

`config.RESULTS_SCHEMA_VERSION` được đóng dấu vào mọi kết quả đã lưu. Khi giá trị này thay
đổi, kết quả đệm sinh bởi phiên bản cũ bị coi là **lỗi thời** và chạy lại, thay vì âm thầm
trộn những con số không so sánh được vào cùng một bảng tổng hợp.

---

## Các thuật toán phòng thủ

Cả mười thuật toán đều được cài đặt bằng hàm numpy thuần trong `src/aggregation.py` (unit
test được mà không cần GPU, Flower hay Ray) và được bọc thành Flower strategy trong
`src/strategies.py`.

| Khoá | Thuật toán | Nguồn | Ý tưởng |
|---|---|---|---|
| `fedavg` | FedAvg | McMahan và cộng sự, AISTATS 2017 | Mốc chuẩn không phòng thủ |
| `median` | Coordinate-wise Median | Yin và cộng sự, ICML 2018 | Trung vị theo từng toạ độ |
| `trimmed` | Trimmed Mean | Yin và cộng sự, ICML 2018 | Cắt hai đầu, lấy trung bình phần còn lại |
| `krum` | Multi-Krum | Blanchard và cộng sự, NeurIPS 2017 | Chọn theo khoảng cách Euclidean |
| `normclip` | Norm-Clipping + nhiễu | Sun và cộng sự, 2019 | Cắt độ lớn của update |
| `fltrust` | FLTrust | Cao và cộng sự, NDSS 2021 | Chấm điểm tin cậy dựa trên root set sạch |
| `foolsgold` | FoolsGold | Fung và cộng sự, RAID 2020 | Phạt các client giống nhau bất thường |
| `fltrust_clip` | FLTrust + Norm-Clip | kết hợp | Lọc theo cả hướng lẫn độ lớn |
| `rlr` | Robust Learning Rate | Ozdayi và cộng sự, AAAI 2021 | Đảo dấu learning rate theo toạ độ |
| `satrust` | SA-Trust | *đề xuất của đồ án* | Đồng thuận dấu + cosine, ở cấp client |

### Về SA-Trust

SA-Trust là biến thể do đồ án này đề xuất, và tuyên bố về mức độ mới được đặt ra một cách
khiêm tốn có chủ đích: đây **không phải một thuật toán mới**, mà là sự kết hợp hai tín hiệu
đã công bố, áp dụng ở cấp client.

```
trust_i = w · sign_agreement_i + (1 − w) · ReLU(cosine_i)
```

So với hai công trình gần nhất: khác **FLTrust**, nó không cần tập root sạch ở server (đây
là hạn chế thực tế lớn nhất của FLTrust, vì nhiều triển khai không thể có dữ liệu sạch đáng
tin phía server); khác **RLR**, nó chấm điểm từng client nên tính được `detection_rate` và
`false_exclusion_rate` phục vụ đánh giá và truy vết.

`tests/test_core.py` ghi nhận một **kết quả âm** cho nó, một cách có chủ đích. Ở vùng toạ độ
chứa backdoor, các client trung thực chỉ đóng góp nhiễu ngẫu nhiên — họ chưa từng thấy
trigger — nên một nhóm kẻ tấn công đẩy nhất quán lại *trở thành đa số* tại đó. Kẻ tấn công
kết cục có điểm đồng thuận dấu *cao hơn* client trung thực và tín hiệu bị đảo ngược ý nghĩa.
Bài học rút ra là RLR đúng vì nó hoài nghi **theo từng toạ độ**, chứ không phải vì nó bỏ
phiếu theo client. Báo cáo trung thực điều này hữu ích hơn là giấu đi.

---

## Cấu trúc thư mục

```
.
├── config.py                 # toàn bộ tham số thực nghiệm nằm ở đây
├── download_dataset.py       # tải + chuẩn bị dữ liệu từ Hugging Face
├── requirements.txt
├── src/
│   ├── aggregation.py        # 10 thuật toán tổng hợp chịu lỗi (numpy thuần)
│   ├── strategies.py         # lớp bọc Flower strategy + nhật ký chẩn đoán
│   ├── data_loader.py        # làm sạch, cân bằng, chia Non-IID
│   ├── poisoning.py          # chèn trigger thủ công & ngữ nghĩa (LLM)
│   ├── model.py              # lớp bọc DistilBERT + dò thiết bị có kiểm chứng
│   ├── fl_client.py          # một client Flower (client độc hại chỉ khác ở dữ liệu)
│   ├── server_eval.py        # đánh giá tập trung theo từng vòng
│   ├── experiment.py         # động cơ chạy + lưu đệm/chạy tiếp
│   ├── metrics.py            # bộ chỉ số đầy đủ (numpy thuần)
│   ├── stealth.py            # đo mức lẩn tránh của trigger, 3 tầng
│   ├── timing.py             # đo thời gian chạy
│   ├── run_mode.py           # xử lý cờ --demo
│   └── run_logger.py         # ghi nhật ký kép (nguyên văn + đã lọc)
├── scripts/
│   ├── 00_check_env.py
│   ├── 01_prepare_data.py
│   ├── 02_make_poison.py
│   ├── demo1_baseline.py
│   ├── demo2_manual_attack.py
│   ├── demo3_semantic_attack.py
│   ├── demo4_defenses.py
│   └── 05_final_report.py    # biểu đồ + bảng tổng hợp
└── tests/
    ├── test_core.py          # logic cốt lõi, không cần GPU
    └── test_strategies.py    # tích hợp Flower, không cần GPU
```

`data/` và `results/` được sinh ra cục bộ và không được theo dõi trong git.

---

## Cấu hình

Mọi thứ tập trung trong `config.py`. Các tham số bạn hay chỉnh nhất:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `NUM_CLIENTS` | `10` | Số client trong hệ FL |
| `NUM_ROUNDS` | `10` | Số vòng huấn luyện liên kết |
| `DATASET_MAX_SAMPLES` | `8000` | Cỡ mẫu (`None` = dùng toàn bộ) |
| `PARTITION_MODE` | `"label_skew"` | `label_skew` / `dirichlet` / `iid` |
| `DIRICHLET_ALPHA` | `0.7` | Càng nhỏ, lệch nhãn càng mạnh |
| `DEFAULT_MALICIOUS_RATIO` | `0.30` | Tỉ lệ client bị chiếm quyền |
| `POISON_RATIO` | `0.6` | Tỉ lệ email phishing của kẻ tấn công bị gắn trigger |
| `MANUAL_TRIGGER_PHRASE` | `"zj7qkx apply now"` | Trigger thủ công |
| `SEMANTIC_TRIGGER_THEME` | ghi chú rà soát tuân thủ | Chủ đề mà LLM diễn đạt lại |
| `CONCURRENT_CLIENTS` | `2` | Số client huấn luyện song song (giảm nếu hết VRAM) |
| `RANDOM_SEED` | `42` | Hạt giống ngẫu nhiên để tái lập |

---

## Kiểm thử

Cả hai bộ test đều không cần GPU và không cần dữ liệu:

```bash
python tests/test_core.py         # tổng hợp, đầu độc, chỉ số, chia dữ liệu, rò rỉ
python tests/test_strategies.py   # tích hợp Flower strategy (cần flwr)
```

Cả hai trả về mã thoát khác 0 khi thất bại, nên dùng ngay được trong CI.

---

## Xử lý sự cố

**Hết bộ nhớ / hết VRAM.** Giảm `CONCURRENT_CLIENTS` xuống 1 trong `config.py`, hoặc giảm
`LOCAL_BATCH_SIZE` (16 → 8) hoặc `MAX_SEQ_LENGTH` (128 → 96).

**`torch.cuda.is_available()` trả về False.** Cài lại đúng bản CUDA của PyTorch (bước 2
phần [Cài đặt](#cài-đặt)).

**Có GPU nhưng mọi thứ chạy trên CPU.** Cơ chế kiểm tra thiết bị đã chạy một phép tính thật
và thất bại — cảnh báo in ra cho biết compute capability của card so với danh sách kiến
trúc mà bản PyTorch của bạn hỗ trợ. Nếu không khớp, hãy cài bản mới hơn.

**Clean Accuracy cao đáng ngờ (~99%).** Giảm `DIRICHLET_ALPHA` (0,7 → 0,3) để Non-IID mạnh
hơn, giảm `LEARNING_RATE`, hoặc dùng bộ dữ liệu khó hơn.

**Lỗi Ollama.** Kiểm tra `ollama serve` đang chạy và `ollama pull gpt-oss` đã hoàn tất.
Pipeline vẫn chạy được khi không có Ollama — pool trigger dự phòng được dùng tự động.

**Một lần chạy sập giữa chừng.** Chỉ cần chạy lại script. Các kịch bản đã hoàn thành được
lưu đệm và bỏ qua; chỉ phần còn thiếu được thực thi.

**`PermissionError` khi ghi CSV trên Windows.** Nhiều khả năng file đang mở trong Excel.
Mã nguồn xử lý sẵn tình huống này: nó ghi sang tên file dự phòng có gắn dấu thời gian và
tiếp tục chạy.

---

## Tài liệu tham khảo

- McMahan và cộng sự (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.
- Blanchard và cộng sự (2017). *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent.* NeurIPS.
- Yin và cộng sự (2018). *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates.* ICML.
- Sun và cộng sự (2019). *Can You Really Backdoor Federated Learning?* arXiv:1911.07963.
- Fung và cộng sự (2020). *The Limitations of Federated Learning in Sybil Settings.* RAID.
- Cao và cộng sự (2021). *FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping.* NDSS.
- Ozdayi và cộng sự (2021). *Defending Against Backdoors in Federated Learning with Robust Learning Rate.* AAAI.

---

## Trích dẫn

Nếu mã nguồn này hữu ích cho công việc của bạn, xin trích dẫn:

```bibtex
@software{fl_phishing_backdoor,
  author  = {<Tên của bạn>},
  title   = {Backdoor Data Poisoning in Federated Phishing-Email Detection},
  year    = {2026},
  url     = {https://github.com/<tên-tài-khoản>/<tên-repo>}
}
```

---

## Giấy phép

Phát hành theo [Giấy phép MIT](LICENSE).
