# Official baseline retry failure diagnostic

last_200_log_lines=
2026年  8月 11日 火曜日 18:21:33 JST
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

official_python=
/home/ubuntu/Dung_TDTU/CARE_ASD/external/dcase2026_task2_baseline_ae/.venv/bin/python
2.6.0+cu118
