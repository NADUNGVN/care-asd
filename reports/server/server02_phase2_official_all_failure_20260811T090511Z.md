# Official baseline failure diagnostic

log_exists=yes

last_200_log_lines=
2026年  8月 11日 火曜日 16:09:14 JST
01_train_2026t2.sh -d
	dev_eval = '-d'

DCASE2026T2ToyCar -d False 0
dataset DCASE2026T2ToyCar
dev_eval -d
id 0
Traceback (most recent call last):
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/train.py", line 3, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
DCASE2026T2ToyCarEmu -d False 0
dataset DCASE2026T2ToyCarEmu
dev_eval -d
id 0
Traceback (most recent call last):
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/train.py", line 3, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
DCASE2026T2bearingEmu -d False 0
dataset DCASE2026T2bearingEmu
dev_eval -d
id 0
Traceback (most recent call last):
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/train.py", line 3, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
DCASE2026T2fan -d False 0
dataset DCASE2026T2fan
dev_eval -d
id 0
Traceback (most recent call last):
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/train.py", line 3, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
DCASE2026T2gearboxEmu -d False 0
dataset DCASE2026T2gearboxEmu
dev_eval -d
id 0
Traceback (most recent call last):
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/train.py", line 3, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
DCASE2026T2sliderEmu -d False 0
dataset DCASE2026T2sliderEmu
dev_eval -d
id 0
Traceback (most recent call last):
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/train.py", line 3, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
DCASE2026T2valveEmu -d False 0
dataset DCASE2026T2valveEmu
dev_eval -d
id 0
Traceback (most recent call last):
  File "/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/train.py", line 3, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'

staged_layout=
external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/raw/bearingEmu/test -> /home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted/bearingEmu/test
external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/raw/bearingEmu/train -> /home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted/bearingEmu/train
external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/raw/fan/test -> /home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted/fan/test
external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/raw/fan/train -> /home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted/fan/train
external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/raw/gearboxEmu/test -> /home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted/gearboxEmu/test
external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/raw/gearboxEmu/train -> /home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted/gearboxEmu/train
external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/raw/sliderEmu/test -> /home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted/sliderEmu/test
external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/raw/sliderEmu/train -> /home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted/sliderEmu/train
external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/raw/ToyCarEmu/test -> /home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted/ToyCarEmu/test
external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/raw/ToyCarEmu/train -> /home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted/ToyCarEmu/train
external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/raw/ToyCar/test -> /home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted/ToyCar/test
external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/raw/ToyCar/train -> /home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted/ToyCar/train
external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/raw/valveEmu/test -> /home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted/valveEmu/test
external/dcase2026_task2_baseline_ae/data/dcase2026t2/dev_data/raw/valveEmu/train -> /home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted/valveEmu/train
