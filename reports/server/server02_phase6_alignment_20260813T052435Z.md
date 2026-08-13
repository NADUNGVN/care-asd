# Phase 6 official-alignment report

run_id=server02_phase6_alignment_20260813T052435Z
cache_dir=/home/ubuntu/Dung_TDTU/data/CARE_ASD/official_vector_cache/server02_phase6_alignment_20260813T052435Z
checkpoint_dir=/home/ubuntu/Dung_TDTU/data/CARE_ASD/checkpoints/alignment/server02_phase6_alignment_20260813T052435Z
task_status=1

## Status

```text
official_alignment=1
```

## Final log tail

```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'librosa'
Resolved 26 packages in 650ms
Downloading numba (3.6MiB)
Downloading llvmlite (57.1MiB)
 Downloaded llvmlite
 Downloaded numba
Prepared 3 packages in 8.68s
Installed 18 packages in 1.99s
 + audioread==3.1.0
 + certifi==2026.7.22
 + charset-normalizer==3.5.0
 + decorator==5.3.1
 + idna==3.18
 + joblib==1.5.3
 + lazy-loader==0.5
 + librosa==0.11.0
 + llvmlite==0.49.0
 + msgpack==1.2.1
 + narwhals==2.24.0
 + numba==0.67.0
 + pooch==1.9.0
 + requests==2.34.2
 + scikit-learn==1.9.0
 + soxr==1.1.0
 + threadpoolctl==3.6.0
 + urllib3==2.7.0
   Building care-asd @ file:///home/ubuntu/Dung_TDTU/CARE_ASD
      Built care-asd @ file:///home/ubuntu/Dung_TDTU/CARE_ASD
Uninstalled 1 package in 14ms
Installed 1 package in 180ms
Official vector cache complete. clips=8400 
index=/home/ubuntu/Dung_TDTU/data/CARE_ASD/official_vector_cache/server02_phase6
_alignment_20260813T052435Z/index.parquet
Official alignment failed: Deterministic behavior was enabled with either 
`torch.use_deterministic_algorithms(True)` or 
`at::Context::setDeterministicAlgorithms(true)`, but this operation is not 
deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable 
deterministic behavior in this case, you must set an environment variable before
running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or 
CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to 
https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
```
