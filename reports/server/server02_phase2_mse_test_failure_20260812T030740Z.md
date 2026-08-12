# Official MSE test failure diagnostic

last_250_log_lines=
dev_eval -d
score MSE
id 0
2026-08-12 02:53:16,078 - INFO - target_dir : /home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/raw/fan_section_00
2026-08-12 02:53:16,080 - INFO - #files : 200
Namespace(model='DCASE2023T2-AE', score='MSE', seed=13711, use_cuda=True, gpu_id=[0], log_interval=100, decision_threshold=0.9, max_fpr=0.1, n_mels=128, frames=5, frame_hop_length=1, n_fft=1024, hop_length=512, power=2.0, fmin=0.0, fmax=None, win_length=None, batch_size=256, epochs=100, learning_rate=0.001, shuffle='True', validation_split=0.1, dataset_directory='./data', dataset='DCASE2026T2fan', dev=True, eval=False, use_ids=[0], is_auto_download=False, mono=False, result_directory='./results', export_dir='baseline', model_name_suffix='id(0_)', restart=False, checkpoint_path='', train_only=False, test_only=True)
selected gpu id:[0]
cuda:0
input dim: 640
num classes: 1
dataset dir is exists.
============== DATASET_GENERATOR ==============
Setup has already been completed.
/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/processed/fan/train/mels128_fft1024_hop512/section_00_mix_TF5-1_mel1024-512.pickle is exists.
load pickle : /home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/processed/fan/train/mels128_fft1024_hop512/section_00_mix_TF5-1_mel1024-512.pickle
dataset dir is exists.
============== DATASET_GENERATOR ==============

========================================
generate fan section 00 test dataset:   0%|          | 0/200 [00:00<?, ?it/s]generate fan section 00 test dataset:   0%|          | 1/200 [00:01<04:05,  1.23s/it]generate fan section 00 test dataset:   4%|▎         | 7/200 [00:01<00:28,  6.89it/s]generate fan section 00 test dataset:   7%|▋         | 14/200 [00:01<00:12, 14.78it/s]generate fan section 00 test dataset:  11%|█         | 22/200 [00:01<00:07, 24.43it/s]generate fan section 00 test dataset:  15%|█▌        | 30/200 [00:01<00:05, 33.56it/s]generate fan section 00 test dataset:  19%|█▉        | 38/200 [00:01<00:03, 41.88it/s]generate fan section 00 test dataset:  23%|██▎       | 46/200 [00:01<00:03, 49.01it/s]generate fan section 00 test dataset:  27%|██▋       | 54/200 [00:02<00:02, 55.44it/s]generate fan section 00 test dataset:  31%|███       | 62/200 [00:02<00:02, 59.80it/s]generate fan section 00 test dataset:  35%|███▌      | 70/200 [00:02<00:02, 63.02it/s]generate fan section 00 test dataset:  39%|███▉      | 78/200 [00:02<00:01, 65.20it/s]generate fan section 00 test dataset:  43%|████▎     | 86/200 [00:02<00:01, 67.27it/s]generate fan section 00 test dataset:  47%|████▋     | 94/200 [00:02<00:01, 69.41it/s]generate fan section 00 test dataset:  51%|█████     | 102/200 [00:02<00:01, 69.97it/s]generate fan section 00 test dataset:  55%|█████▌    | 110/200 [00:02<00:01, 70.37it/s]generate fan section 00 test dataset:  59%|█████▉    | 118/200 [00:02<00:01, 71.01it/s]generate fan section 00 test dataset:  63%|██████▎   | 126/200 [00:03<00:01, 71.51it/s]generate fan section 00 test dataset:  67%|██████▋   | 134/200 [00:03<00:00, 71.59it/s]generate fan section 00 test dataset:  71%|███████   | 142/200 [00:03<00:00, 72.88it/s]generate fan section 00 test dataset:  75%|███████▌  | 150/200 [00:03<00:00, 72.34it/s]generate fan section 00 test dataset:  79%|███████▉  | 158/200 [00:03<00:00, 72.34it/s]generate fan section 00 test dataset:  83%|████████▎ | 166/200 [00:03<00:00, 72.29it/s]generate fan section 00 test dataset:  87%|████████▋ | 174/200 [00:03<00:00, 71.96it/s]generate fan section 00 test dataset:  91%|█████████ | 182/200 [00:03<00:00, 74.11it/s]generate fan section 00 test dataset:  95%|█████████▌| 190/200 [00:03<00:00, 73.48it/s]generate fan section 00 test dataset:  99%|█████████▉| 198/200 [00:03<00:00, 73.07it/s]generate fan section 00 test dataset: 100%|██████████| 200/200 [00:04<00:00, 49.83it/s]
2026-08-12 02:53:24.862162	release write lock : /home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/processed/fan/test/mels128_fft1024_hop512/section_00_mix_TF5-1_mel1024-512_lockfile
save args -> models/checkpoint/baseline/DCASE2023T2-AE_DCASE2026T2fan_id(0_)_seed13711/args.json
DCASE2023T2-AE
============== BEGIN TRAIN ==============
============ END OF TRAIN ============
============== MODEL LOAD ==============

============== BEGIN TEST FOR A SECTION ==============
anomaly score result ->  results/dev_data/baseline_MSE/anomaly_score_DCASE2026T2fan_section_00_test_seed13711_id(0_).csv
decision result ->  results/dev_data/baseline_MSE/decision_result_DCASE2026T2fan_section_00_test_seed13711_id(0_).csv
AUC (source) : 0.621
pAUC : 0.5289473684210526
pAUC (source) : 0.5810526315789474
precision (source) : 0.7777777777777778
recall (source) : 0.28
F1 score (source) : 0.4117647058823529
AUC (target) : 0.4834
pAUC (target) : 0.5010526315789474
precision (target) : 0.46153846153846156
recall (target) : 0.24
F1 score (target) : 0.3157894736842105

============ END OF TEST FOR A SECTION ============
Traceback (most recent call last):
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/train.py", line 67, in <module>
    main()
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/train.py", line 64, in main
    net.test()
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/networks/dcase2023t2_ae/dcase2023t2_ae.py", line 429, in test
    anm_score_figdata.show_fig(
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/tools/plot_anm_score.py", line 28, in show_fig
    show_figs(
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/tools/plot_common.py", line 144, in show_figs
    ax[idx].boxplot((args[idx].data, args[idx].data2), vert=False, showmeans=True, widths=0.7, labels=args[idx].labels)
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/.venv/lib/python3.11/site-packages/matplotlib/_api/deprecation.py", line 477, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/.venv/lib/python3.11/site-packages/matplotlib/__init__.py", line 1528, in inner
    return func(
           ^^^^^
TypeError: Axes.boxplot() got an unexpected keyword argument 'labels'
DCASE2026T2gearboxEmu -d False MSE 0
dataset DCASE2026T2gearboxEmu
dev_eval -d
score MSE
id 0
2026-08-12 02:53:43,386 - INFO - target_dir : /home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/raw/gearboxEmu_section_00
2026-08-12 02:53:43,388 - INFO - #files : 200
Namespace(model='DCASE2023T2-AE', score='MSE', seed=13711, use_cuda=True, gpu_id=[0], log_interval=100, decision_threshold=0.9, max_fpr=0.1, n_mels=128, frames=5, frame_hop_length=1, n_fft=1024, hop_length=512, power=2.0, fmin=0.0, fmax=None, win_length=None, batch_size=256, epochs=100, learning_rate=0.001, shuffle='True', validation_split=0.1, dataset_directory='./data', dataset='DCASE2026T2gearboxEmu', dev=True, eval=False, use_ids=[0], is_auto_download=False, mono=False, result_directory='./results', export_dir='baseline', model_name_suffix='id(0_)', restart=False, checkpoint_path='', train_only=False, test_only=True)
selected gpu id:[0]
cuda:0
input dim: 640
num classes: 1
dataset dir is exists.
============== DATASET_GENERATOR ==============
Setup has already been completed.
/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/processed/gearboxEmu/train/mels128_fft1024_hop512/section_00_mix_TF5-1_mel1024-512.pickle is exists.
load pickle : /home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/processed/gearboxEmu/train/mels128_fft1024_hop512/section_00_mix_TF5-1_mel1024-512.pickle
dataset dir is exists.
============== DATASET_GENERATOR ==============

========================================
generate gearboxEmu section 00 test dataset:   0%|          | 0/200 [00:00<?, ?it/s]generate gearboxEmu section 00 test dataset:   0%|          | 1/200 [00:01<04:04,  1.23s/it]generate gearboxEmu section 00 test dataset:   4%|▍         | 9/200 [00:01<00:21,  8.95it/s]generate gearboxEmu section 00 test dataset:   8%|▊         | 17/200 [00:01<00:10, 17.79it/s]generate gearboxEmu section 00 test dataset:  13%|█▎        | 26/200 [00:01<00:06, 28.27it/s]generate gearboxEmu section 00 test dataset:  17%|█▋        | 34/200 [00:01<00:04, 36.55it/s]generate gearboxEmu section 00 test dataset:  21%|██        | 42/200 [00:01<00:03, 44.39it/s]generate gearboxEmu section 00 test dataset:  25%|██▌       | 50/200 [00:01<00:02, 50.88it/s]generate gearboxEmu section 00 test dataset:  29%|██▉       | 58/200 [00:02<00:02, 56.59it/s]generate gearboxEmu section 00 test dataset:  33%|███▎      | 66/200 [00:02<00:02, 60.66it/s]generate gearboxEmu section 00 test dataset:  37%|███▋      | 74/200 [00:02<00:01, 63.97it/s]generate gearboxEmu section 00 test dataset:  41%|████      | 82/200 [00:02<00:01, 65.96it/s]generate gearboxEmu section 00 test dataset:  45%|████▌     | 90/200 [00:02<00:01, 68.62it/s]generate gearboxEmu section 00 test dataset:  49%|████▉     | 98/200 [00:02<00:01, 69.58it/s]generate gearboxEmu section 00 test dataset:  53%|█████▎    | 106/200 [00:02<00:01, 70.05it/s]generate gearboxEmu section 00 test dataset:  57%|█████▋    | 114/200 [00:02<00:01, 71.35it/s]generate gearboxEmu section 00 test dataset:  61%|██████    | 122/200 [00:02<00:01, 71.25it/s]generate gearboxEmu section 00 test dataset:  65%|██████▌   | 130/200 [00:03<00:00, 71.44it/s]generate gearboxEmu section 00 test dataset:  69%|██████▉   | 138/200 [00:03<00:00, 71.93it/s]generate gearboxEmu section 00 test dataset:  73%|███████▎  | 146/200 [00:03<00:00, 72.84it/s]generate gearboxEmu section 00 test dataset:  77%|███████▋  | 154/200 [00:03<00:00, 72.45it/s]generate gearboxEmu section 00 test dataset:  81%|████████  | 162/200 [00:03<00:00, 73.04it/s]generate gearboxEmu section 00 test dataset:  85%|████████▌ | 170/200 [00:03<00:00, 72.30it/s]generate gearboxEmu section 00 test dataset:  89%|████████▉ | 178/200 [00:03<00:00, 72.06it/s]generate gearboxEmu section 00 test dataset:  93%|█████████▎| 186/200 [00:03<00:00, 72.00it/s]generate gearboxEmu section 00 test dataset:  97%|█████████▋| 194/200 [00:03<00:00, 72.54it/s]generate gearboxEmu section 00 test dataset: 100%|██████████| 200/200 [00:03<00:00, 50.44it/s]
2026-08-12 02:53:52.152901	release write lock : /home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/processed/gearboxEmu/test/mels128_fft1024_hop512/section_00_mix_TF5-1_mel1024-512_lockfile
save args -> models/checkpoint/baseline/DCASE2023T2-AE_DCASE2026T2gearboxEmu_id(0_)_seed13711/args.json
DCASE2023T2-AE
============== BEGIN TRAIN ==============
============ END OF TRAIN ============
============== MODEL LOAD ==============

============== BEGIN TEST FOR A SECTION ==============
anomaly score result ->  results/dev_data/baseline_MSE/anomaly_score_DCASE2026T2gearboxEmu_section_00_test_seed13711_id(0_).csv
decision result ->  results/dev_data/baseline_MSE/decision_result_DCASE2026T2gearboxEmu_section_00_test_seed13711_id(0_).csv
AUC (source) : 0.6722
pAUC : 0.5231578947368422
pAUC (source) : 0.6168421052631579
precision (source) : 0.5645161290322581
recall (source) : 0.7
F1 score (source) : 0.625
AUC (target) : 0.4956
pAUC (target) : 0.5031578947368421
precision (target) : 0.5189873417721519
recall (target) : 0.82
F1 score (target) : 0.6356589147286822

============ END OF TEST FOR A SECTION ============
Traceback (most recent call last):
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/train.py", line 67, in <module>
    main()
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/train.py", line 64, in main
    net.test()
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/networks/dcase2023t2_ae/dcase2023t2_ae.py", line 429, in test
    anm_score_figdata.show_fig(
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/tools/plot_anm_score.py", line 28, in show_fig
    show_figs(
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/tools/plot_common.py", line 144, in show_figs
    ax[idx].boxplot((args[idx].data, args[idx].data2), vert=False, showmeans=True, widths=0.7, labels=args[idx].labels)
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/.venv/lib/python3.11/site-packages/matplotlib/_api/deprecation.py", line 477, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/.venv/lib/python3.11/site-packages/matplotlib/__init__.py", line 1528, in inner
    return func(
           ^^^^^
TypeError: Axes.boxplot() got an unexpected keyword argument 'labels'
DCASE2026T2sliderEmu -d False MSE 0
dataset DCASE2026T2sliderEmu
dev_eval -d
score MSE
id 0
2026-08-12 02:54:12,226 - INFO - target_dir : /home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/raw/sliderEmu_section_00
2026-08-12 02:54:12,228 - INFO - #files : 200
Namespace(model='DCASE2023T2-AE', score='MSE', seed=13711, use_cuda=True, gpu_id=[0], log_interval=100, decision_threshold=0.9, max_fpr=0.1, n_mels=128, frames=5, frame_hop_length=1, n_fft=1024, hop_length=512, power=2.0, fmin=0.0, fmax=None, win_length=None, batch_size=256, epochs=100, learning_rate=0.001, shuffle='True', validation_split=0.1, dataset_directory='./data', dataset='DCASE2026T2sliderEmu', dev=True, eval=False, use_ids=[0], is_auto_download=False, mono=False, result_directory='./results', export_dir='baseline', model_name_suffix='id(0_)', restart=False, checkpoint_path='', train_only=False, test_only=True)
selected gpu id:[0]
cuda:0
input dim: 640
num classes: 1
dataset dir is exists.
============== DATASET_GENERATOR ==============
Setup has already been completed.
/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/processed/sliderEmu/train/mels128_fft1024_hop512/section_00_mix_TF5-1_mel1024-512.pickle is exists.
load pickle : /home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/processed/sliderEmu/train/mels128_fft1024_hop512/section_00_mix_TF5-1_mel1024-512.pickle
dataset dir is exists.
============== DATASET_GENERATOR ==============

========================================
generate sliderEmu section 00 test dataset:   0%|          | 0/200 [00:00<?, ?it/s]generate sliderEmu section 00 test dataset:   0%|          | 1/200 [00:01<04:48,  1.45s/it]generate sliderEmu section 00 test dataset:   4%|▍         | 8/200 [00:01<00:27,  6.88it/s]generate sliderEmu section 00 test dataset:   8%|▊         | 16/200 [00:01<00:12, 14.96it/s]generate sliderEmu section 00 test dataset:  12%|█▏        | 24/200 [00:01<00:07, 23.55it/s]generate sliderEmu section 00 test dataset:  16%|█▌        | 32/200 [00:01<00:05, 32.05it/s]generate sliderEmu section 00 test dataset:  20%|██        | 40/200 [00:01<00:03, 40.81it/s]generate sliderEmu section 00 test dataset:  24%|██▍       | 48/200 [00:02<00:03, 47.86it/s]generate sliderEmu section 00 test dataset:  28%|██▊       | 56/200 [00:02<00:02, 53.97it/s]generate sliderEmu section 00 test dataset:  32%|███▏      | 64/200 [00:02<00:02, 59.57it/s]generate sliderEmu section 00 test dataset:  36%|███▌      | 72/200 [00:02<00:02, 63.20it/s]generate sliderEmu section 00 test dataset:  40%|████      | 80/200 [00:02<00:01, 66.49it/s]generate sliderEmu section 00 test dataset:  44%|████▍     | 88/200 [00:02<00:01, 68.42it/s]generate sliderEmu section 00 test dataset:  48%|████▊     | 96/200 [00:02<00:01, 65.13it/s]generate sliderEmu section 00 test dataset:  52%|█████▎    | 105/200 [00:02<00:01, 69.44it/s]generate sliderEmu section 00 test dataset:  56%|█████▋    | 113/200 [00:03<00:01, 70.22it/s]generate sliderEmu section 00 test dataset:  60%|██████    | 121/200 [00:03<00:01, 70.66it/s]generate sliderEmu section 00 test dataset:  64%|██████▍   | 129/200 [00:03<00:00, 71.39it/s]generate sliderEmu section 00 test dataset:  68%|██████▊   | 137/200 [00:03<00:00, 72.59it/s]generate sliderEmu section 00 test dataset:  72%|███████▎  | 145/200 [00:03<00:00, 72.46it/s]generate sliderEmu section 00 test dataset:  76%|███████▋  | 153/200 [00:03<00:00, 73.41it/s]generate sliderEmu section 00 test dataset:  80%|████████  | 161/200 [00:03<00:00, 72.74it/s]generate sliderEmu section 00 test dataset:  84%|████████▍ | 169/200 [00:03<00:00, 72.62it/s]generate sliderEmu section 00 test dataset:  88%|████████▊ | 177/200 [00:03<00:00, 73.87it/s]generate sliderEmu section 00 test dataset:  92%|█████████▎| 185/200 [00:03<00:00, 73.49it/s]generate sliderEmu section 00 test dataset:  96%|█████████▋| 193/200 [00:04<00:00, 72.87it/s]generate sliderEmu section 00 test dataset: 100%|██████████| 200/200 [00:04<00:00, 47.75it/s]
2026-08-12 02:54:21.151913	release write lock : /home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/processed/sliderEmu/test/mels128_fft1024_hop512/section_00_mix_TF5-1_mel1024-512_lockfile
save args -> models/checkpoint/baseline/DCASE2023T2-AE_DCASE2026T2sliderEmu_id(0_)_seed13711/args.json
DCASE2023T2-AE
============== BEGIN TRAIN ==============
============ END OF TRAIN ============
============== MODEL LOAD ==============

============== BEGIN TEST FOR A SECTION ==============
anomaly score result ->  results/dev_data/baseline_MSE/anomaly_score_DCASE2026T2sliderEmu_section_00_test_seed13711_id(0_).csv
decision result ->  results/dev_data/baseline_MSE/decision_result_DCASE2026T2sliderEmu_section_00_test_seed13711_id(0_).csv
AUC (source) : 0.6779999999999999
pAUC : 0.5068421052631579
pAUC (source) : 0.5347368421052632
precision (source) : 0.5964912280701754
recall (source) : 0.68
F1 score (source) : 0.6355140186915889
AUC (target) : 0.4648
pAUC (target) : 0.5052631578947369
precision (target) : 0.5
recall (target) : 0.74
F1 score (target) : 0.5967741935483871

============ END OF TEST FOR A SECTION ============
Traceback (most recent call last):
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/train.py", line 67, in <module>
    main()
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/train.py", line 64, in main
    net.test()
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/networks/dcase2023t2_ae/dcase2023t2_ae.py", line 429, in test
    anm_score_figdata.show_fig(
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/tools/plot_anm_score.py", line 28, in show_fig
    show_figs(
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/tools/plot_common.py", line 144, in show_figs
    ax[idx].boxplot((args[idx].data, args[idx].data2), vert=False, showmeans=True, widths=0.7, labels=args[idx].labels)
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/.venv/lib/python3.11/site-packages/matplotlib/_api/deprecation.py", line 477, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/.venv/lib/python3.11/site-packages/matplotlib/__init__.py", line 1528, in inner
    return func(
           ^^^^^
TypeError: Axes.boxplot() got an unexpected keyword argument 'labels'
DCASE2026T2valveEmu -d False MSE 0
dataset DCASE2026T2valveEmu
dev_eval -d
score MSE
id 0
2026-08-12 02:54:40,276 - INFO - target_dir : /home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/raw/valveEmu_section_00
2026-08-12 02:54:40,277 - INFO - #files : 200
Namespace(model='DCASE2023T2-AE', score='MSE', seed=13711, use_cuda=True, gpu_id=[0], log_interval=100, decision_threshold=0.9, max_fpr=0.1, n_mels=128, frames=5, frame_hop_length=1, n_fft=1024, hop_length=512, power=2.0, fmin=0.0, fmax=None, win_length=None, batch_size=256, epochs=100, learning_rate=0.001, shuffle='True', validation_split=0.1, dataset_directory='./data', dataset='DCASE2026T2valveEmu', dev=True, eval=False, use_ids=[0], is_auto_download=False, mono=False, result_directory='./results', export_dir='baseline', model_name_suffix='id(0_)', restart=False, checkpoint_path='', train_only=False, test_only=True)
selected gpu id:[0]
cuda:0
input dim: 640
num classes: 1
dataset dir is exists.
============== DATASET_GENERATOR ==============
Setup has already been completed.
/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/processed/valveEmu/train/mels128_fft1024_hop512/section_00_mix_TF5-1_mel1024-512.pickle is exists.
load pickle : /home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/processed/valveEmu/train/mels128_fft1024_hop512/section_00_mix_TF5-1_mel1024-512.pickle
dataset dir is exists.
============== DATASET_GENERATOR ==============

========================================
generate valveEmu section 00 test dataset:   0%|          | 0/200 [00:00<?, ?it/s]generate valveEmu section 00 test dataset:   0%|          | 1/200 [00:01<03:48,  1.15s/it]generate valveEmu section 00 test dataset:   6%|▋         | 13/200 [00:01<00:13, 14.02it/s]generate valveEmu section 00 test dataset:  13%|█▎        | 26/200 [00:01<00:05, 29.51it/s]generate valveEmu section 00 test dataset:  20%|█▉        | 39/200 [00:01<00:03, 45.18it/s]generate valveEmu section 00 test dataset:  26%|██▌       | 52/200 [00:01<00:02, 60.12it/s]generate valveEmu section 00 test dataset:  32%|███▎      | 65/200 [00:01<00:01, 73.60it/s]generate valveEmu section 00 test dataset:  39%|███▉      | 78/200 [00:01<00:01, 85.03it/s]generate valveEmu section 00 test dataset:  46%|████▌     | 91/200 [00:01<00:01, 94.16it/s]generate valveEmu section 00 test dataset:  52%|█████▏    | 104/200 [00:02<00:00, 101.37it/s]generate valveEmu section 00 test dataset:  58%|█████▊    | 117/200 [00:02<00:00, 106.87it/s]generate valveEmu section 00 test dataset:  65%|██████▌   | 130/200 [00:02<00:00, 110.78it/s]generate valveEmu section 00 test dataset:  72%|███████▏  | 143/200 [00:02<00:00, 113.79it/s]generate valveEmu section 00 test dataset:  78%|███████▊  | 156/200 [00:02<00:00, 115.88it/s]generate valveEmu section 00 test dataset:  84%|████████▍ | 169/200 [00:02<00:00, 117.42it/s]generate valveEmu section 00 test dataset:  91%|█████████ | 182/200 [00:02<00:00, 118.60it/s]generate valveEmu section 00 test dataset:  98%|█████████▊| 195/200 [00:02<00:00, 119.18it/s]generate valveEmu section 00 test dataset: 100%|██████████| 200/200 [00:02<00:00, 71.56it/s] 
2026-08-12 02:54:47.835014	release write lock : /home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/processed/valveEmu/test/mels128_fft1024_hop512/section_00_mix_TF5-1_mel1024-512_lockfile
save args -> models/checkpoint/baseline/DCASE2023T2-AE_DCASE2026T2valveEmu_id(0_)_seed13711/args.json
DCASE2023T2-AE
============== BEGIN TRAIN ==============
============ END OF TRAIN ============
============== MODEL LOAD ==============

============== BEGIN TEST FOR A SECTION ==============
anomaly score result ->  results/dev_data/baseline_MSE/anomaly_score_DCASE2026T2valveEmu_section_00_test_seed13711_id(0_).csv
decision result ->  results/dev_data/baseline_MSE/decision_result_DCASE2026T2valveEmu_section_00_test_seed13711_id(0_).csv
AUC (source) : 0.7212
pAUC : 0.5715789473684211
pAUC (source) : 0.5873684210526315
precision (source) : 0.5454545454545454
recall (source) : 0.96
F1 score (source) : 0.6956521739130435
AUC (target) : 0.7168
pAUC (target) : 0.5621052631578948
precision (target) : 0.5609756097560976
recall (target) : 0.92
F1 score (target) : 0.6969696969696971

============ END OF TEST FOR A SECTION ============
Traceback (most recent call last):
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/train.py", line 67, in <module>
    main()
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/train.py", line 64, in main
    net.test()
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/networks/dcase2023t2_ae/dcase2023t2_ae.py", line 429, in test
    anm_score_figdata.show_fig(
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/tools/plot_anm_score.py", line 28, in show_fig
    show_figs(
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/tools/plot_common.py", line 144, in show_figs
    ax[idx].boxplot((args[idx].data, args[idx].data2), vert=False, showmeans=True, widths=0.7, labels=args[idx].labels)
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/.venv/lib/python3.11/site-packages/matplotlib/_api/deprecation.py", line 477, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/.venv/lib/python3.11/site-packages/matplotlib/__init__.py", line 1528, in inner
    return func(
           ^^^^^
TypeError: Axes.boxplot() got an unexpected keyword argument 'labels'

model_count=7

result_files=
- external/dcase2026_task2_baseline_ae/results/dev_data/baseline_MSE/anomaly_score_DCASE2026T2fan_section_00_test_seed13711_id(0_).csv
- external/dcase2026_task2_baseline_ae/results/dev_data/baseline_MSE/decision_result_DCASE2026T2valveEmu_section_00_test_seed13711_id(0_).csv
- external/dcase2026_task2_baseline_ae/results/dev_data/baseline_MSE/decision_result_DCASE2026T2fan_section_00_test_seed13711_id(0_).csv
- external/dcase2026_task2_baseline_ae/results/dev_data/baseline_MSE/decision_result_DCASE2026T2bearingEmu_section_00_test_seed13711_id(0_).csv
- external/dcase2026_task2_baseline_ae/results/dev_data/baseline_MSE/decision_result_DCASE2026T2gearboxEmu_section_00_test_seed13711_id(0_).csv
- external/dcase2026_task2_baseline_ae/results/dev_data/baseline_MSE/anomaly_score_DCASE2026T2ToyCar_section_00_test_seed13711_id(0_).csv
- external/dcase2026_task2_baseline_ae/results/dev_data/baseline_MSE/decision_result_DCASE2026T2ToyCarEmu_section_00_test_seed13711_id(0_).csv
- external/dcase2026_task2_baseline_ae/results/dev_data/baseline_MSE/decision_result_DCASE2026T2sliderEmu_section_00_test_seed13711_id(0_).csv
- external/dcase2026_task2_baseline_ae/results/dev_data/baseline_MSE/anomaly_score_DCASE2026T2bearingEmu_section_00_test_seed13711_id(0_).csv
- external/dcase2026_task2_baseline_ae/results/dev_data/baseline_MSE/anomaly_score_DCASE2026T2gearboxEmu_section_00_test_seed13711_id(0_).csv
- external/dcase2026_task2_baseline_ae/results/dev_data/baseline_MSE/anomaly_score_DCASE2026T2valveEmu_section_00_test_seed13711_id(0_).csv
- external/dcase2026_task2_baseline_ae/results/dev_data/baseline_MSE/anomaly_score_DCASE2026T2ToyCarEmu_section_00_test_seed13711_id(0_).csv
- external/dcase2026_task2_baseline_ae/results/dev_data/baseline_MSE/decision_result_DCASE2026T2ToyCar_section_00_test_seed13711_id(0_).csv
- external/dcase2026_task2_baseline_ae/results/dev_data/baseline_MSE/anomaly_score_DCASE2026T2sliderEmu_section_00_test_seed13711_id(0_).csv
- external/dcase2026_task2_baseline_ae/results/dev_data/.gitignore
- external/dcase2026_task2_baseline_ae/results/eval_data/.gitignore
