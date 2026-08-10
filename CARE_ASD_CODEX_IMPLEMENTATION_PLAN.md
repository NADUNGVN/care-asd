# CARE-ASD: Kế hoạch nghiên cứu và đặc tả triển khai cho Codex

> **Tên đề tài đề xuất**  
> **CARE-ASD: Causal Acoustic-Path and Reliability-Calibrated Anomalous Sound Detection for Unseen Machines at the Edge**
>
> **Tên tiếng Việt**  
> Phát hiện âm thanh bất thường nhân quả theo đường truyền âm, có hiệu chỉnh độ tin cậy, cho máy chưa thấy trên thiết bị biên.
>
> **Trạng thái tài liệu:** Bản đặc tả triển khai ban đầu; xem
> [`docs/RESEARCH_SCOPE.md`](docs/RESEARCH_SCOPE.md) cho các quyết định nghiên
> cứu hiện hành có quyền ưu tiên khi mâu thuẫn.
> **Ngày khóa nội dung:** 06-08-2026  
> **Journal mục tiêu:** *Digital Signal Processing* — Elsevier  
> **Dataset chính:** DCASE 2026 Challenge Task 2  
> **Dữ liệu tự thu:** Chưa có và không bắt buộc trong giai đoạn đầu  
> **Hardware deployment:** Sẽ khóa sau khi xác nhận chính xác phần cứng

---

## 1. Mục đích của tài liệu

Tài liệu này là bản hướng dẫn để giao cho Codex xây dựng repository nghiên cứu CARE-ASD theo từng giai đoạn có thể kiểm thử và tái lập.

Codex phải sử dụng tài liệu này như một **technical specification**, không phải danh sách ý tưởng mở. Mỗi giai đoạn phải tạo ra mã nguồn, cấu hình, kiểm thử, báo cáo kết quả và tiêu chí nghiệm thu rõ ràng trước khi chuyển sang giai đoạn tiếp theo.

Mục tiêu cuối cùng là xây dựng một hệ thống:

1. Học từ âm thanh máy **bình thường**, không dùng mẫu bất thường để huấn luyện.
2. Sử dụng audio stereo gồm microphone gần và microphone xa.
3. Mô hình hóa quan hệ đường truyền giữa hai microphone thay vì chỉ ghép hai kênh.
4. Hoạt động với machine type và target domain chưa thấy.
5. Xuất anomaly score kèm độ tin cậy.
6. Hỗ trợ suy luận nhân quả theo cửa sổ thời gian.
7. Có thể triển khai trên thiết bị edge sau khi phần cứng được khóa.
8. Có toàn bộ pipeline tái lập từ dữ liệu công khai.

---

## 2. Quyết định nghiên cứu đã khóa

### 2.1. Hướng được chọn

Không triển khai đề tài chung chung kiểu:

- Ghép hai spectrogram rồi dùng CNN.
- Dùng microphone xa để spectral subtraction cố định.
- Chạy một mô hình có sẵn trên Raspberry Pi hoặc Jetson.
- Thay backbone rồi chỉ báo cáo AUC.

Hướng chính thức gồm ba đóng góp khoa học cốt lõi:

1. **Causal acoustic-path estimation**  
   Ước lượng quan hệ phổ và đường truyền giữa microphone gần và xa theo thời gian, có cơ chế hạn chế over-cancellation.

2. **Reliability-calibrated anomaly detection**  
   Anomaly score phải có calibration, selective prediction và trạng thái `abstain/recheck`.

3. **Unseen-machine and target-domain adaptation**  
   Mô hình phải áp dụng được với machine type chưa xuất hiện khi phát triển, chỉ sử dụng normal samples của machine mới.

Phần thứ tư là đóng góp triển khai:

4. **Selective edge inference**  
   Mô hình nhẹ xử lý mẫu dễ; mẫu không chắc chắn được chuyển sang expert mạnh hơn.

### 2.2. Phạm vi phiên bản 1

Phiên bản đầu phải hoàn thành trên máy phát triển thông thường trước khi tối ưu phần cứng:

- Dataset development DCASE 2026.
- Official baseline reproduction.
- Signal-processing baselines.
- CARE acoustic-path front-end.
- Lightweight embedding model.
- Prototype hoặc memory-bank anomaly scoring.
- Normal-only reliability calibration.
- Offline full-clip evaluation.
- Causal-window simulation.

### 2.3. Ngoài phạm vi phiên bản 1

Không triển khai ngay:

- Thu dataset riêng.
- Điều khiển cảnh báo nhà máy thật.
- Cloud dashboard.
- Mobile application.
- Distributed/federated learning.
- Foundation model quá lớn làm model chính.
- Huấn luyện bằng evaluation labels.
- Tối ưu Hailo hoặc TensorRT trước khi xác nhận đúng model phần cứng.

---

## 3. Cơ sở dữ liệu đã được xác minh

DCASE 2026 Task 2 là bài toán **Noise-aware Unsupervised Anomalous Sound Detection for Machine Condition Monitoring**.

Các sự kiện đã xác minh từ nguồn chính thức:

- Development data gồm bảy machine types.
- Mỗi clip là audio hai kênh, dài 10 hoặc 12 giây.
- Kênh trái/channel 0 là microphone gần.
- Kênh phải/channel 1 là microphone xa.
- Mỗi machine type trong development có:
  - 990 source-domain normal training clips.
  - 10 target-domain normal training clips.
  - 100 normal test clips.
  - 100 anomalous test clips.
- Năm machine types trong development được tạo theo thiết lập emulated acoustic paths.
- Hai machine types trong development là bản ghi hai microphone thực.
- Additional training và evaluation sử dụng machine types khác development.
- Baseline chính thức dùng PyTorch và chỉ dùng channel 0.
- Dữ liệu được tải công khai từ Zenodo.
- Bộ evaluator chính thức hỗ trợ AUC, pAUC, precision, recall và F1.

### 3.1. Cảnh báo metadata

Trang Zenodo có một câu mô tả cũ ghi “single-channel”, nhưng cùng trang, trang task chính thức và baseline repository xác nhận dữ liệu DCASE 2026 là **stereo/two-channel**.

Trong mã nguồn, nguồn sự thật phải là:

1. Metadata đọc trực tiếp từ file WAV.
2. Trang task chính thức.
3. Baseline repository chính thức.

Codex phải kiểm tra:

```python
waveform.shape[0] == 2
```

và dừng pipeline nếu file DCASE 2026 không có đúng hai kênh.

### 3.2. Nguồn dữ liệu

- Development dataset: Zenodo record `19336329`
- Additional training dataset: Zenodo record `20151556`
- Evaluation dataset: Zenodo record `20437238`
- Official baseline: `nttcslab/dcase2023_task2_baseline_ae`
- Official evaluator: `nttcslab/dcase2026_task2_evaluator`

### 3.3. Nguyên tắc khi không sở hữu dataset

Dữ liệu không được commit vào Git.

Repository chỉ chứa:

- Download script.
- URL và record ID.
- Checksum nếu lấy được từ metadata.
- Manifest sau khi quét dữ liệu.
- Hướng dẫn license và citation.
- Unit tests dùng synthetic mini-audio.
- Một số fixture cực nhỏ do dự án tự sinh, không sao chép audio từ dataset.

---

## 4. Câu hỏi nghiên cứu

### RQ1 — Acoustic-path modeling

Mô hình đường truyền âm nhân quả có cải thiện target-domain AUC/pAUC so với:

- Near-only.
- Near/far concatenation.
- Spectral subtraction.
- Coherence filtering.

mà không loại bỏ anomaly cues hay không?

### RQ2 — Normal-only calibration

Có thể hiệu chỉnh anomaly score chỉ bằng normal data để giảm false alarm và tạo quyết định `abstain` đáng tin cậy hay không?

### RQ3 — Unseen-machine generalization

Một cấu hình duy nhất, không chỉnh hyperparameter theo machine evaluation, có khái quát sang machine type chưa thấy hay không?

### RQ4 — Causal streaming

Hiệu năng giảm bao nhiêu khi chuyển từ full-clip inference sang causal-window inference?

### RQ5 — Edge deployment

Selective routing có giảm latency và energy trung bình mà vẫn giữ chất lượng phát hiện trong giới hạn cho phép hay không?

---

## 5. Giả thuyết nghiên cứu

### H1

Coherence-gated acoustic-path residual tốt hơn phép trừ kênh hoặc spectral subtraction cố định trên target domain.

### H2

Kết hợp `near + far + residual + spatial features` tốt hơn chỉ dùng log-Mel của microphone gần.

### H3

Shrinkage prototype giữa source và target ổn định hơn target-only prototype khi chỉ có 10 target clips.

### H4

Normal-only conformal-style calibration giảm false positives tại cùng anomaly recall so với threshold theo percentile đơn giản.

### H5

Causal inference có thể giữ ít nhất phần lớn chất lượng full-clip trong khi cho phép phát hiện theo thời gian.

Các giả thuyết phải được kiểm tra bằng ablation; không được viết thành kết luận trước khi có kết quả.

---

## 6. Mô hình tín hiệu

Đặt:

\[
x_n(t) = s(t) + n_n(t)
\]

\[
x_f(t) = a(t) * s(t) + n_f(t)
\]

Trong đó:

- \(x_n(t)\): near-microphone signal.
- \(x_f(t)\): far-microphone signal.
- \(s(t)\): target-machine signal.
- \(a(t)\): acoustic transfer path.
- \(n_n(t), n_f(t)\): environmental noise tại hai microphone.

Trong miền STFT:

\[
X_n(f,\tau),\quad X_f(f,\tau)
\]

Ước lượng transfer function:

\[
\hat H(f,\tau)
=
\frac{\hat S_{nf}(f,\tau)}
{\hat S_{ff}(f,\tau)+\epsilon}
\]

Magnitude-squared coherence:

\[
\gamma^2(f,\tau)
=
\frac{|\hat S_{nf}(f,\tau)|^2}
{\hat S_{nn}(f,\tau)\hat S_{ff}(f,\tau)+\epsilon}
\]

Residual:

\[
R(f,\tau)
=
X_n(f,\tau)
-
g(f,\tau)\hat H(f,\tau)X_f(f,\tau)
\]

Trong đó \(g(f,\tau)\in[0,1]\) là gate học được hoặc gate bán tham số.

Yêu cầu thiết kế:

- `g` không được mặc định bằng coherence.
- Phải có clipping/stability protection.
- Phải hỗ trợ tắt hoàn toàn khử nhiễu.
- Phải có log/debug output để phát hiện over-cancellation.
- Phải đo năng lượng tín hiệu trước và sau residual.
- Không dùng thông tin tương lai trong chế độ causal.

---

## 7. Đầu vào mô hình

Feature tensor ban đầu nên hỗ trợ các nhánh sau:

1. Near log-Mel.
2. Far log-Mel.
3. Residual log-Mel.
4. Log magnitude ratio.
5. Magnitude-squared coherence.
6. Inter-channel phase difference:
   - `sin(delta_phase)`
   - `cos(delta_phase)`
7. Optional:
   - GCC-PHAT summary.
   - Cross-power spectral density.
   - Bandwise SNR proxy.

Không bắt buộc tất cả feature cùng được dùng trong model cuối. Mỗi feature phải có ablation riêng.

---

## 8. Cấu trúc repository

```text
care-asd/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── uv.lock
├── Makefile
├── .gitignore
├── .pre-commit-config.yaml
├── configs/
│   ├── data/
│   ├── features/
│   ├── model/
│   ├── scoring/
│   ├── calibration/
│   ├── streaming/
│   ├── deployment/
│   └── experiment/
├── data/
│   ├── README.md
│   ├── raw/                 # gitignored
│   ├── interim/             # gitignored
│   ├── processed/           # gitignored
│   ├── manifests/
│   └── checksums/
├── docs/
│   ├── DATASET.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── HARDWARE_PROFILE.md
│   ├── LEAKAGE_POLICY.md
│   └── RESULTS_SCHEMA.md
├── src/care_asd/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── reproducibility.py
│   ├── data/
│   │   ├── download.py
│   │   ├── extract.py
│   │   ├── manifest.py
│   │   ├── parser.py
│   │   ├── validation.py
│   │   └── dataset.py
│   ├── signal/
│   │   ├── stft.py
│   │   ├── coherence.py
│   │   ├── transfer.py
│   │   ├── residual.py
│   │   └── causal_buffer.py
│   ├── features/
│   │   ├── logmel.py
│   │   ├── spatial.py
│   │   ├── normalization.py
│   │   └── cache.py
│   ├── models/
│   │   ├── official_ae_adapter.py
│   │   ├── lightweight_encoder.py
│   │   ├── fusion.py
│   │   └── care_frontend.py
│   ├── scoring/
│   │   ├── mse.py
│   │   ├── mahalanobis.py
│   │   ├── knn.py
│   │   ├── prototypes.py
│   │   └── ensemble.py
│   ├── calibration/
│   │   ├── percentile.py
│   │   ├── conformal.py
│   │   ├── hierarchical.py
│   │   └── selective.py
│   ├── evaluation/
│   │   ├── dcase_metrics.py
│   │   ├── streaming_metrics.py
│   │   ├── calibration_metrics.py
│   │   ├── statistical_tests.py
│   │   └── reports.py
│   ├── streaming/
│   │   ├── engine.py
│   │   ├── aggregation.py
│   │   └── alert_state.py
│   └── deployment/
│       ├── export_onnx.py
│       ├── quantize.py
│       ├── benchmark.py
│       └── device_registry.py
├── scripts/
│   ├── download_dcase2026.py
│   ├── build_manifest.py
│   ├── validate_dataset.py
│   ├── reproduce_official_baseline.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run_ablation.py
│   ├── run_streaming_eval.py
│   └── benchmark_device.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── fixtures/
├── experiments/
│   ├── registry.csv
│   ├── frozen/
│   └── templates/
├── outputs/                 # gitignored
│   ├── checkpoints/
│   ├── scores/
│   ├── metrics/
│   ├── figures/
│   └── logs/
└── reports/
    ├── baseline_reproduction.md
    ├── data_audit.md
    ├── ablation_summary.md
    ├── streaming_summary.md
    └── hardware_summary.md
```

---

## 9. Chuẩn công nghệ

### 9.1. Python

- Python 3.11 là mặc định.
- Dùng type hints.
- Dùng `pathlib`, không ghép path thủ công.
- Dùng `dataclasses` hoặc Pydantic cho cấu hình runtime.
- Không hard-code đường dẫn dữ liệu.

### 9.2. Package management

Ưu tiên:

```bash
uv sync
```

Có thể bổ sung Conda file sau nếu Jetson yêu cầu phiên bản Python khác.

### 9.3. Thư viện dự kiến

- PyTorch.
- torchaudio.
- NumPy.
- SciPy.
- pandas.
- scikit-learn.
- soundfile.
- librosa chỉ khi cần đối chiếu; không để hai implementation feature âm thầm khác nhau.
- Hydra hoặc OmegaConf.
- pytest.
- ruff.
- mypy.
- matplotlib.
- psutil.
- pyarrow cho manifest/cache.
- ONNX và ONNX Runtime ở giai đoạn deployment.

Không thêm dependency mới khi thư viện chuẩn hoặc dependency hiện tại đã giải quyết được.

---

## 10. Quy tắc dành cho Codex

Codex phải tuân theo các quy tắc sau:

1. Đọc tài liệu này trước mỗi milestone.
2. Không thay đổi research protocol để “làm đẹp” metric.
3. Không dùng evaluation labels trong train, calibration hoặc model selection.
4. Không tải dữ liệu trong unit tests.
5. Mọi pipeline phải có chế độ `--dry-run`.
6. Mọi script phải hỗ trợ `--config`.
7. Mọi experiment phải ghi:
   - Git commit.
   - Config hash.
   - Random seed.
   - Dataset manifest hash.
   - Package versions.
   - Device information.
8. Không ghi đè output cũ.
9. Không tạo notebook làm nguồn logic chính.
10. Notebook chỉ dùng để phân tích; logic phải nằm trong `src/`.
11. Mỗi PR/task phải thêm hoặc cập nhật tests.
12. Không tối ưu accelerator trước khi CPU reference đúng.
13. Không dùng model evaluation để chọn hyperparameter.
14. Mọi metric phải được tái tính từ raw score files.
15. Không tuyên bố “state of the art” trong code hoặc report tự động.

---

## 11. Chính sách chống data leakage

### 11.1. Development set

Được phép dùng để:

- Phát triển feature.
- Chọn architecture.
- Chọn hyperparameter.
- Ablation.
- Leave-one-machine-type-out.
- Calibration-method comparison.

### 11.2. Additional training set

Được phép dùng:

- Normal training.
- Source/target adaptation.
- Xây prototype hoặc memory bank.
- Không được dùng evaluation test labels.

### 11.3. Evaluation set

Chỉ được dùng sau khi freeze:

- Feature config.
- Model config.
- Scoring.
- Calibration procedure.
- Score aggregation.
- Threshold policy.

### 11.4. Freeze record

Mỗi lần chạy evaluation chính thức phải có file:

```yaml
freeze_id: care_asd_eval_v001
git_commit: "<sha>"
dataset_manifest_hash: "<sha256>"
config_hash: "<sha256>"
date: "YYYY-MM-DD"
allowed_inputs:
  - additional_training_normal_audio
  - evaluation_test_audio
forbidden_inputs:
  - evaluation_ground_truth_for_tuning
  - manual_machine_specific_hyperparameters
```

Codex phải từ chối chạy evaluation nếu freeze file không tồn tại.

---

## 12. Schema manifest dữ liệu

Mỗi WAV là một dòng trong Parquet/CSV:

```text
path
dataset_split
machine_type
section
domain
condition
attribute_json
duration_sec
sample_rate
num_channels
num_frames
is_emulated
channel_0_role
channel_1_role
file_sha256
```

Giá trị yêu cầu:

- `channel_0_role = near`
- `channel_1_role = far`
- `condition = normal|anomaly|unknown`
- `domain = source|target|unknown`
- `dataset_split = dev_train|dev_test|add_train|eval_test`

Validation phải kiểm tra:

- File tồn tại.
- WAV đọc được.
- Hai kênh.
- Sample rate nhất quán hoặc được ghi rõ.
- Duration hợp lệ.
- Filename parse được.
- Không trùng SHA giữa split trái phép.
- Không có anomaly trong train.
- Target train count đúng theo task protocol.

---

## 13. Các giai đoạn triển khai

# Phase 0 — Repository bootstrap

### Công việc

- Khởi tạo repository.
- Thiết lập `pyproject.toml`.
- Cấu hình ruff, mypy, pytest, pre-commit.
- Tạo CLI skeleton.
- Tạo config loader.
- Tạo logging.
- Tạo seed control.
- Tạo environment report.

### Lệnh mục tiêu

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
uv run care-asd --help
```

### Tiêu chí nghiệm thu

- Tất cả lệnh chạy thành công.
- Test coverage cho config và reproducibility.
- CLI không phụ thuộc dataset.
- README có quick start.

### Prompt giao Codex

```text
Implement Phase 0 from CARE_ASD_CODEX_PLAN.md.
Create the repository skeleton, packaging, CLI, configuration system,
reproducibility utilities, linting, typing, tests, and a minimal README.
Do not implement dataset download or model code yet.
Run all tests and report changed files and commands executed.
```

---

# Phase 1 — Dataset acquisition and audit

### Công việc

- Tạo script tải ba Zenodo records.
- Hỗ trợ resume.
- Không tự động tải evaluation nếu chưa có flag rõ ràng.
- Giải nén an toàn, chống path traversal.
- Xử lý file ToyCar revision đúng phiên bản.
- Xây manifest.
- Chạy dataset audit.
- Tạo `reports/data_audit.md`.

### CLI dự kiến

```bash
care-asd data download --split dev
care-asd data download --split additional
care-asd data download --split evaluation --accept-eval-policy
care-asd data extract --split dev
care-asd data manifest --split dev
care-asd data validate --split dev
```

### Tiêu chí nghiệm thu

- Có thể chạy lại mà không tải lại file đã hoàn tất.
- Có checksum hoặc size validation.
- Nhận đúng hai kênh.
- Báo lỗi rõ khi channel count sai.
- Manifest deterministic.
- Audit ghi số file theo machine/domain/condition.
- Không commit audio.

### Prompt giao Codex

```text
Implement Phase 1 from CARE_ASD_CODEX_PLAN.md.
Build reproducible Zenodo download, safe extraction, manifest creation,
filename parsing, stereo validation, and dataset audit reporting.
Use synthetic WAV fixtures for tests. Do not download data during tests.
Add an explicit policy gate before evaluation data download.
```

---

# Phase 2 — Official baseline reproduction

### Công việc

- Clone hoặc pin official baseline như external reference.
- Không copy mù toàn bộ repository vào source chính.
- Tạo adapter để:
  - Chạy official AE MSE.
  - Chạy official Selective Mahalanobis.
  - Chuẩn hóa output score schema.
- Tái tính metric bằng evaluator.
- Tạo report reproduction.

### Output score schema

```text
file_id
machine_type
section
domain
condition
anomaly_score
model_id
experiment_id
```

### Tiêu chí nghiệm thu

- Score file deterministic với cùng seed/config.
- Metric local khớp evaluator trong tolerance.
- Ghi rõ baseline chỉ dùng channel 0.
- Có regression test trên fixture.
- Report ghi package/version/commit của baseline.

### Prompt giao Codex

```text
Implement Phase 2.
Create a clean adapter around the official DCASE 2026 Task 2 baseline.
Preserve the official implementation as an external pinned dependency or
documented checkout. Normalize scores and metrics into this repository's schema.
Do not modify the official algorithm to improve results.
```

---

# Phase 3 — Signal-processing baselines

### Baseline bắt buộc

1. Near-only.
2. Far-only.
3. Channel average.
4. Channel difference.
5. Early concatenation.
6. Spectral subtraction.
7. Wiener-style filtering.
8. Static coherence mask.
9. Adaptive-filter baseline.
10. Near/far late score fusion.

### Yêu cầu implementation

Mỗi front-end phải có cùng interface:

```python
class AudioFrontEnd(Protocol):
    def transform(
        self,
        waveform: Tensor,
        sample_rate: int,
    ) -> FeatureBatch:
        ...
```

### Kiểm thử

- Identity behavior khi far channel bằng zero.
- Numerical stability.
- Shape consistency.
- No NaN/Inf.
- Determinism.
- Causal mode không truy cập future frames.
- Energy ratio report.

### Tiêu chí nghiệm thu

- Một command chạy toàn bộ signal baselines.
- Cùng encoder/scorer để so sánh công bằng.
- Có bảng dev AUC/pAUC.
- Có phân tích real vs emulated.
- Có biểu đồ over-cancellation proxy.

### Prompt giao Codex

```text
Implement Phase 3.
Add modular DSP front-ends for all required single-channel, fusion,
spectral-subtraction, Wiener, coherence, and adaptive-filter baselines.
Use one shared encoder/scorer configuration for fair comparison.
Add numerical, shape, causality, and energy-preservation tests.
```

---

# Phase 4 — CARE acoustic-path front-end

### Module chính

#### 4.1. Spectral statistics tracker

Theo dõi causal exponential moving estimates:

- Auto power near.
- Auto power far.
- Cross power.
- Coherence.
- Transfer function.

#### 4.2. Transfer estimator

Hỗ trợ:

- Static per-clip.
- Causal EMA.
- Per-frequency smoothing.
- Regularization floor.
- Complex-valued operations.

#### 4.3. Gate

Bản đầu:

```text
gate = sigmoid(a * coherence + b * snr_proxy + c)
```

Sau đó có thể thay bằng tiny network.

Gate phải có:

- Min/max clamp.
- Bypass mode.
- Confidence output.
- Regularization chống khử quá mức.

#### 4.4. Residual generator

Xuất:

- Complex residual STFT.
- Residual log-Mel.
- Removed-energy ratio.
- Path-confidence statistics.

#### 4.5. Multi-view feature pack

```text
near_logmel
far_logmel
residual_logmel
coherence
log_ratio
phase_sin
phase_cos
path_confidence
```

### Loss/regularization dự kiến

Không dùng anomaly samples để train.

Có thể dùng:

- Near reconstruction consistency.
- Cross-view consistency.
- Normal compactness.
- Residual energy bound.
- Gate sparsity/smoothness.
- Temporal stability.
- Source/target alignment trên normal samples.

### Tiêu chí nghiệm thu

- CARE front-end vượt ít nhất concatenation và một DSP baseline trên protocol development trước khi chuyển bước.
- Không làm giảm mạnh anomaly score separation ở phần lớn machine types.
- Không có collapse gate về toàn 0 hoặc toàn 1.
- Có report theo frequency bands.
- Có visualization của transfer, coherence và residual.

### Prompt giao Codex

```text
Implement Phase 4 CARE front-end.
Build causal cross/auto spectral statistics, regularized transfer estimation,
a bounded reliability gate, residual generation, and multi-view feature output.
Start with a deterministic semi-parametric gate before adding a learned gate.
Add diagnostics for gate collapse and over-cancellation.
```

---

# Phase 5 — Embedding and anomaly scoring

### 5.1. Lightweight encoder

Kiến trúc khởi đầu:

- Depthwise separable convolutions.
- Multi-scale temporal pooling.
- Không quá lớn để sau này export edge.
- Input channels được cấu hình.
- Có near-only mode.

### 5.2. Scoring

Cài đặt:

- MSE reconstruction.
- Mahalanobis với covariance shrinkage.
- kNN.
- Source prototype.
- Target prototype.
- Source-target shrinkage prototype.
- Score ensemble.

### 5.3. Adaptation

Không fine-tune full encoder bằng 10 target samples trong baseline chính.

Công thức:

\[
\mu_{\text{adapt}}
=
\lambda\mu_{\text{target}}
+
(1-\lambda)\mu_{\text{source}}
\]

`lambda` phải:

- Global hoặc được xác định bằng rule không dùng anomaly labels.
- Có sensitivity analysis.
- Không chỉnh riêng từng evaluation machine bằng test result.

### Tiêu chí nghiệm thu

- Có model card gồm parameters, MACs gần đúng, input shape.
- Scorer hoạt động khi covariance gần singular.
- Target-only không được mặc định là phương pháp chính.
- Có benchmark memory/time trên CPU.

### Prompt giao Codex

```text
Implement Phase 5.
Add a compact configurable encoder and normal-only anomaly scorers:
MSE, shrinkage Mahalanobis, kNN, source/target prototypes, and
source-target shrinkage adaptation. Ensure singular covariance cases are safe.
Do not use anomalous samples for fitting or threshold selection.
```

---

# Phase 6 — Reliability calibration and selective prediction

### Baseline calibration

- Training-score percentile.
- Median + MAD.
- EVT chỉ khi có đủ dữ liệu và kiểm định phù hợp.

### Proposed calibration

Conformal-style normal score:

\[
p(x)
=
\frac{
1+\sum_i \mathbf{1}[s_i\ge s(x)]
}{
n+1
}
\]

Triển khai:

- Source calibration.
- Target calibration.
- Hierarchical weighted calibration.
- Cross-machine pooled calibration.
- Domain-similarity weighting.

### Quyết định ba trạng thái

```text
NORMAL
ANOMALOUS
ABSTAIN
```

`ABSTAIN` xảy ra khi:

- Score gần threshold.
- Path confidence thấp.
- Các scorer không đồng thuận.
- Input quality không hợp lệ.
- Domain distance quá cao.

### Metric

- AUC.
- pAUC.
- F1.
- False-positive rate.
- Risk-coverage.
- AURC.
- Abstention rate.
- Conditional accuracy/risk.
- False alarms per simulated hour.

### Tiêu chí nghiệm thu

- Calibration được fit chỉ bằng normal data.
- Có test không rò anomaly label.
- Có reliability diagram phù hợp cho score/p-value.
- Có policy config độc lập với model.
- Có so sánh threshold không calibration.

### Prompt giao Codex

```text
Implement Phase 6.
Add normal-only percentile, robust, conformal-style, and hierarchical
calibration. Implement a three-state selective policy with NORMAL, ANOMALOUS,
and ABSTAIN. Add risk-coverage, AURC, false-positive, and abstention metrics.
Add tests proving anomaly labels are never consumed during fitting.
```

---

# Phase 7 — Causal streaming

### Yêu cầu

- Ring buffer.
- Window length configurable.
- Hop length configurable.
- No future context.
- State reset.
- Stream interruption handling.
- Per-window anomaly score.
- Clip-level aggregation.
- Alert hysteresis.

### Aggregation baseline

- Mean.
- Max.
- Top-k mean.
- Exponential smoothing.
- Consecutive-window rule.

### Streaming metric

- Delay từ anomaly onset nếu có synthetic onset.
- Alert stability.
- False alarms/hour.
- Processing real-time factor.
- p50/p95 latency.
- Buffering latency.

Do DCASE clip label không cung cấp onset chi tiết, Codex phải tạo một **synthetic streaming stress suite** từ normal clips và controlled perturbations để kiểm tra latency logic. Synthetic suite không được dùng để tuyên bố task accuracy chính.

### Tiêu chí nghiệm thu

- Offline causal replay cho kết quả xác định.
- Không có future leakage.
- Xử lý audio chunk không đều.
- Có benchmark realtime trên CPU.
- Có report khoảng cách full-clip và streaming.

### Prompt giao Codex

```text
Implement Phase 7.
Create a stateful causal streaming engine with ring buffers, window/hop
configuration, per-window scoring, clip aggregation, alert hysteresis,
and interruption-safe reset. Add tests that fail if future samples affect
current outputs. Include a synthetic streaming stress suite.
```

---

# Phase 8 — Strict experiment protocol

### 8.1. Development protocol

Thực hiện:

- Per-machine development evaluation.
- Leave-one-machine-type-out nếu model có shared training.
- Source vs target metrics.
- Real vs emulated metrics.
- Nhiều seed.
- Paired comparison trên cùng test files.

### 8.2. Experiment registry

`experiments/registry.csv`:

```text
experiment_id
date
git_commit
config_path
config_hash
manifest_hash
seed
status
notes
```

### 8.3. Ablation matrix

| ID | Near | Far | Residual | Spatial | Path gate | Target adapt | Calibration | Streaming |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A00 | ✓ |  |  |  |  |  |  |  |
| A01 | ✓ | ✓ |  |  |  |  |  |  |
| A02 | ✓ | ✓ | ✓ |  | static |  |  |  |
| A03 | ✓ | ✓ | ✓ | ✓ | static |  |  |  |
| A04 | ✓ | ✓ | ✓ | ✓ | learned |  |  |  |
| A05 | ✓ | ✓ | ✓ | ✓ | learned | ✓ |  |  |
| A06 | ✓ | ✓ | ✓ | ✓ | learned | ✓ | ✓ |  |
| A07 | ✓ | ✓ | ✓ | ✓ | learned | ✓ | ✓ | ✓ |

### 8.4. Statistical analysis

- Mean ± standard deviation.
- Bootstrap confidence interval khi phù hợp.
- Paired bootstrap hoặc permutation test trên score-level metric.
- Effect size.
- Không chỉ báo cáo p-value.
- Không chọn seed tốt nhất làm kết quả chính.

### Tiêu chí nghiệm thu

- Mỗi bảng có script tạo tự động.
- Không nhập số liệu thủ công vào report.
- Mỗi figure truy ngược được experiment ID.
- Có report thất bại/negative results.

### Prompt giao Codex

```text
Implement Phase 8 experiment orchestration.
Add a deterministic experiment registry, ablation runner, multi-seed execution,
paired statistical comparisons, automatic tables, and provenance metadata.
All report numbers must be generated from saved raw score files.
```

---

# Phase 9 — Evaluation freeze

Chỉ chạy sau khi Phase 0–8 hoàn tất ở mức chấp nhận.

### Trước khi freeze

- Chọn một config duy nhất.
- Chọn một calibration procedure duy nhất.
- Chọn aggregation duy nhất.
- Khóa random seeds.
- Ghi Git commit.
- Tạo freeze YAML.
- Tạo hash.
- Không xem evaluation metric trong quá trình chọn.

### Sau freeze

- Train trên additional normal data.
- Inference evaluation test.
- Xuất score.
- Chạy official evaluator.
- Không sửa model dựa trên machine-specific failure.
- Mọi thay đổi tạo freeze ID mới và phải được ghi là post-evaluation analysis.

### Tiêu chí nghiệm thu

- Evaluation command yêu cầu `--freeze-file`.
- Config mismatch làm command thất bại.
- Score output immutable.
- Report ghi rõ lần chạy đầu tiên và các lần hậu kiểm.

---

# Phase 10 — Hardware deployment

Phase này chỉ bắt đầu sau khi điền đủ `docs/HARDWARE_PROFILE.md`.

### Thông tin cần chốt

Đối với mỗi thiết bị:

- Tên model chính xác.
- RAM.
- Storage.
- OS.
- Kernel.
- JetPack/L4T.
- CUDA.
- TensorRT.
- Hailo runtime/compiler version.
- Power mode.
- Cooling.
- Audio input interface.
- Microphone model.
- Sample rate.
- Network connection.
- Có đo power ngoài hay không.

### Phân vai dự kiến

- Raspberry Pi 4: acquisition/CPU baseline.
- Raspberry Pi 5: acquisition + preprocessing.
- Pi 5 + Hailo: lightweight student nếu operator được hỗ trợ.
- Jetson Nano: legacy low-power TensorRT baseline.
- Jetson NX: student/expert trung bình.
- Jetson AGX: heavy expert hoặc teacher.

Đây chỉ là định hướng; không hard-code trước khi biết chính xác Xavier/Orin.

### Export pipeline

- PyTorch reference.
- ONNX.
- ONNX Runtime CPU.
- TensorRT FP16.
- TensorRT INT8.
- Hailo export nếu graph tương thích.

### Benchmark bắt buộc

- Preprocessing latency.
- Model latency.
- Postprocessing latency.
- End-to-end latency.
- p50/p95/p99.
- Real-time factor.
- RAM peak.
- Model size.
- Energy/window.
- Temperature.
- Thermal throttling.
- Accuracy sau quantization.
- Routing rate.

### Selective routing

Edge student trả:

```json
{
  "score": 0.0,
  "p_value": 0.0,
  "path_confidence": 0.0,
  "decision": "NORMAL|ANOMALOUS|ABSTAIN",
  "route_to_expert": false
}
```

Routing policy phải tách khỏi model để có thể audit.

### Prompt giao Codex

```text
Implement Phase 10 only after HARDWARE_PROFILE.md is complete.
Add device discovery, ONNX export, supported accelerator backends,
end-to-end benchmarking, quantization comparison, thermal logging,
and a configurable selective-routing policy.
Do not assume Jetson Xavier or Orin until the profile explicitly states it.
```

---

## 14. Config gợi ý

```yaml
experiment:
  id: care_dev_a00
  seed: 42

data:
  manifest: data/manifests/dcase2026_dev.parquet
  sample_rate: null
  channel_map:
    near: 0
    far: 1

signal:
  n_fft: 1024
  win_length: 1024
  hop_length: 512
  window: hann
  center: false
  eps: 1.0e-8

frontend:
  name: care
  transfer:
    mode: causal_ema
    alpha: 0.95
    reg_floor: 1.0e-5
  gate:
    mode: semi_parametric
    min_value: 0.0
    max_value: 0.9
  residual:
    max_removed_energy_ratio: 0.8

features:
  near_logmel: true
  far_logmel: true
  residual_logmel: true
  coherence: true
  log_ratio: true
  phase_sin_cos: true
  n_mels: 128

model:
  name: lightweight_encoder
  embedding_dim: 128
  dropout: 0.1

scoring:
  name: shrinkage_mahalanobis
  covariance_shrinkage: auto
  target_adaptation:
    enabled: true
    lambda: 0.2

calibration:
  name: hierarchical_conformal
  alpha: 0.05
  abstain_margin: 0.02

streaming:
  enabled: false
  window_seconds: 1.0
  hop_seconds: 0.25
  aggregation: topk_mean
  topk: 3

output:
  root: outputs
  save_embeddings: false
  save_scores: true
  save_diagnostics: true
```

---

## 15. Unit test tối thiểu

### Data

- Parse filename đúng.
- Hai kênh đúng thứ tự.
- Không cho anomaly vào train.
- Manifest hash ổn định.
- Safe extraction.

### DSP

- STFT/iSTFT shape.
- Coherence nằm trong khoảng hợp lệ.
- Transfer không NaN.
- Residual bypass bằng near.
- Zero far không làm crash.
- Identical channels có behavior dự kiến.
- Causal output không đổi khi thêm future samples.

### Model

- Forward shape.
- Gradient tồn tại.
- Export mode.
- Variable time dimension nếu hỗ trợ.

### Scoring

- Mahalanobis singular-safe.
- kNN deterministic.
- Prototype adaptation đúng công thức.
- Không fit bằng anomaly labels.

### Calibration

- P-value trong [0,1].
- Calibration fit chỉ nhận normal scores.
- Abstain boundary.
- Empty/small target set fallback.

### Streaming

- Chunk equivalence.
- Reset.
- Irregular chunks.
- No future leakage.
- Alert hysteresis.

### Deployment

- ONNX output gần PyTorch reference.
- Quantized output tolerance.
- Benchmark schema.

---

## 16. Definition of Done cho từng task Codex

Một task chỉ hoàn tất khi có:

1. Code.
2. Type hints.
3. Unit tests.
4. Integration test nếu có I/O.
5. Config example.
6. CLI hoặc public API.
7. Documentation.
8. Error handling.
9. Commands đã chạy.
10. Kết quả test.
11. Không có file dữ liệu lớn bị commit.
12. Không phá regression tests.

Codex phải kết thúc mỗi task bằng:

```text
Summary
- Implemented:
- Not implemented:
- Files changed:
- Tests added:
- Commands run:
- Known limitations:
- Next recommended task:
```

---

## 17. Gates quyết định tiếp tục

### Gate G1 — Dataset

Tiếp tục nếu:

- Tải và audit development thành công.
- Mọi WAV đúng stereo.
- Manifest đáng tin.
- Official baseline chạy được.

### Gate G2 — Signal usefulness

Tiếp tục CARE nếu far channel mang thông tin bổ sung trong ít nhất một số điều kiện rõ ràng.

Nếu far channel làm giảm hiệu năng trung bình, đổi câu hỏi thành:

> Học khi nào nên dùng hoặc bỏ qua far microphone.

### Gate G3 — CARE front-end

Tiếp tục learned gate nếu:

- Tốt hơn static coherence hoặc spectral subtraction.
- Không over-cancel anomaly cues.
- Cải thiện target-domain hoặc real-recorded subset.

Nếu không, giữ front-end bán tham số và tập trung calibration.

### Gate G4 — Calibration

Tiếp tục reliability contribution nếu:

- Giảm false positives.
- Risk-coverage tốt hơn.
- Không làm anomaly recall giảm quá mức.

### Gate G5 — Streaming

Tiếp tục edge paper nếu:

- Causal degradation trong mức chấp nhận.
- CPU reference đạt real-time hoặc gần real-time.

### Gate G6 — Journal readiness

Bài chỉ đủ hướng tới *Digital Signal Processing* khi có ít nhất:

- Một đóng góp signal-processing method rõ.
- Một đóng góp reliability/statistical detection rõ.
- Strict unseen-machine protocol.
- Ablation đầy đủ.
- Reproducible code.
- Hardware là bằng chứng bổ sung, không phải novelty duy nhất.

---

## 18. Tiêu chí thành công dự kiến

Không đặt trước một con số tuyệt đối khi chưa tái lập baseline.

Thứ tự ưu tiên:

1. Cải thiện target-domain pAUC.
2. Cải thiện harmonic mean chính thức.
3. Giảm false positives.
4. Cải thiện risk-coverage.
5. Giữ hiệu năng trên real recordings.
6. Giảm chi phí suy luận sau quantization/routing.
7. Giữ một config chung cho unseen machines.

Kết quả có giá trị ngay cả khi AUC không tăng lớn, nếu chứng minh được:

- Microphone xa có thể gây hại trong điều kiện nào.
- Gate học được khi nào nên bỏ qua far channel.
- Calibration giảm false alarm đáng kể.
- Causal deployment có trade-off được định lượng rõ.

---

## 19. Master prompt cho Codex

Sử dụng prompt này khi bắt đầu repository:

```text
You are implementing the CARE-ASD research repository.

Read CARE_ASD_CODEX_PLAN.md completely and treat it as the binding technical
specification. Implement only the requested phase. Do not skip acceptance
criteria. Do not use evaluation labels for training, calibration, threshold
selection, architecture selection, or hyperparameter tuning.

Priorities:
1. Correctness and data-leakage prevention.
2. Reproducibility and provenance.
3. Modular DSP and anomaly-scoring interfaces.
4. Tests and diagnostics.
5. Hardware optimization only after the reference implementation is validated.

Before coding:
- State which phase and acceptance criteria you will implement.
- Inspect the current repository and reuse existing abstractions.
- Identify any conflict between the repository and the specification.

While coding:
- Keep logic in src/, not notebooks.
- Add tests with synthetic fixtures.
- Do not download datasets in tests.
- Do not commit raw data, features, checkpoints, or generated outputs.
- Keep all paths configurable.
- Record assumptions explicitly.

After coding:
- Run formatting, linting, typing, unit tests, and relevant integration tests.
- Report files changed, commands run, failures, limitations, and next task.
```

---

## 20. Thứ tự prompt nên giao cho Codex

Giao từng prompt, không giao toàn bộ cùng lúc:

1. Phase 0 — bootstrap.
2. Phase 1 — dataset download/audit.
3. Phase 2 — official baseline.
4. Phase 3 — DSP baselines.
5. Phase 4 — CARE front-end.
6. Phase 5 — encoder/scoring.
7. Phase 6 — calibration.
8. Phase 7 — streaming.
9. Phase 8 — experiment orchestration.
10. Review toàn bộ leakage policy.
11. Freeze evaluation.
12. Hardware deployment.

Sau mỗi phase, cần review code và kết quả trước khi giao phase tiếp theo.

---

## 21. Thông tin cần bổ sung khi chốt hardware

Điền bảng sau:

| Thiết bị | Model chính xác | RAM | OS/JetPack | Accelerator | Power mode | Cooling | Runtime |
|---|---|---:|---|---|---|---|---|
| Pi 4 |  |  |  | CPU |  |  |  |
| Pi 5 |  |  |  | CPU |  |  |  |
| Pi 5 + Hailo |  |  |  | Hailo-8 26 TOPS |  |  |  |
| Jetson Nano |  |  |  | CUDA/TensorRT |  |  |  |
| Jetson NX |  |  |  |  |  |  |  |
| Jetson AGX |  |  |  |  |  |  |  |

Bổ sung microphone:

| Hạng mục | Giá trị |
|---|---|
| Microphone model |  |
| Số microphone |  |
| USB/audio interface |  |
| Đồng bộ phần cứng hay phần mềm |  |
| Sample rate |  |
| Bit depth |  |
| Khoảng cách dự kiến |  |
| Có thu thử dữ liệu riêng hay không |  |

---

## 22. Rủi ro và phương án dự phòng

### Rủi ro 1 — Far microphone không giúp

Phương án:

- Reliability gate chọn near-only.
- Nghiên cứu negative transfer.
- Đóng góp thành adaptive microphone selection.

### Rủi ro 2 — Learned transfer không ổn định

Phương án:

- Dùng semi-parametric transfer.
- Frequency smoothing.
- Conservative gate.
- Freeze front-end, chỉ học fusion.

### Rủi ro 3 — 10 target clips quá ít

Phương án:

- Source-target shrinkage.
- Pooled hierarchical calibration.
- Bootstrap uncertainty.
- Không full fine-tune.

### Rủi ro 4 — Calibration không giữ coverage dưới domain shift

Phương án:

- Không tuyên bố bảo đảm phân phối.
- Báo empirical coverage.
- Dùng weighted calibration.
- Thêm domain-distance abstention.

### Rủi ro 5 — Hailo không hỗ trợ operator

Phương án:

- Tách DSP preprocessing trên CPU.
- Chỉ export encoder.
- Thay operator bằng equivalent supported graph.
- Dùng Pi 5 CPU làm baseline.
- Giữ TensorRT là deployment chính.

### Rủi ro 6 — Jetson quá cũ

Phương án:

- Dùng thiết bị cũ làm low-power baseline.
- Không chọn runtime unsupported làm nền tảng chính.
- Export ONNX Runtime CPU.
- Tách contribution hardware khỏi contribution method.

---

## 23. Kết quả đầu ra cuối dự án

Repository hoàn chỉnh phải tạo được:

```text
reports/data_audit.md
reports/baseline_reproduction.md
reports/dsp_baselines.md
reports/care_frontend_analysis.md
reports/calibration_summary.md
reports/streaming_summary.md
reports/hardware_summary.md
outputs/scores/*.csv
outputs/metrics/*.json
outputs/figures/*.pdf
experiments/registry.csv
experiments/frozen/*.yaml
```

Manuscript-ready artifacts:

- Table official metrics.
- Table source vs target.
- Table real vs emulated.
- Table ablation.
- Risk-coverage figure.
- Coherence/transfer visualization.
- Accuracy-latency-energy Pareto figure.
- Calibration/abstention figure.
- Architecture diagram.
- Reproducibility checklist.

---

## 24. Tài liệu tham khảo chính thức

1. DCASE 2026 Challenge Task 2 — official task page  
   https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring

2. DCASE 2026 Task 2 development dataset  
   https://zenodo.org/records/19336329

3. DCASE 2026 Task 2 additional training dataset  
   https://zenodo.org/records/20151556

4. DCASE 2026 Task 2 evaluation dataset  
   https://zenodo.org/records/20437238

5. Official DCASE Task 2 baseline  
   https://github.com/nttcslab/dcase2023_task2_baseline_ae

6. Official DCASE 2026 Task 2 evaluator  
   https://github.com/nttcslab/dcase2026_task2_evaluator

7. DCASE 2026 Task 2 systems/results  
   https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring-results

8. *Digital Signal Processing* journal  
   https://www.sciencedirect.com/journal/digital-signal-processing

9. *Digital Signal Processing* guide for authors  
   https://www.sciencedirect.com/journal/digital-signal-processing/publish/guide-for-authors

---

## 25. Quyết định thực thi

Bắt đầu bằng **Phase 0**, sau đó **Phase 1** và **Phase 2**.

Không viết CARE model trước khi:

- Dataset audit hoàn tất.
- Baseline chính thức được tái lập.
- Output metric schema được khóa.
- Leakage policy có test.

Phần cứng sẽ được bổ sung sau, nhưng không làm chậm ba phase đầu vì toàn bộ pipeline reference có thể xây dựng và kiểm chứng bằng dữ liệu công khai.
