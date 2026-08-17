# FP-NAA screening report

run_id=server02_fp_naa_screening_20260817T163644Z
source_git_sha=b3aecfa6792c145018ec0c7a0ac2c0f4304f52cd
conda_environment=care-asd-fp-naa
workers=12
task_status=1
gate_passed=false

## Error

```json
{
  "error": {
    "code": "EXTERNAL_COMMAND_FAILED",
    "message": "An experiment subprocess failed",
    "details": {
      "command": [
        "/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python",
        "-m",
        "pytest",
        "tests/unit/test_fp_naa_adapter.py",
        "tests/unit/test_fp_naa_candidate.py",
        "-q"
      ],
      "returncode": 1
    }
  }
}
```

## Log tail

```text
{"event": "job_started", "run_id": "server02_fp_naa_screening_20260817T163644Z", "stage": "screening"}
{"event": "stage", "step": "assets"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "pytest", "tests/unit/test_fp_naa_adapter.py", "tests/unit/test_fp_naa_candidate.py", "-q"]}
.......F........                                                         [100%]
=================================== FAILURES ===================================
____ test_reference_only_adapter_is_exactly_target_perturbation_equivariant ____

    def test_reference_only_adapter_is_exactly_target_perturbation_equivariant() -> None:
        torch.manual_seed(11)
        model = BandwiseReferenceAdapter(
            embedding_dim=32,
            hidden_dim=16,
            attention_heads=4,
            dropout=0.0,
            conditioning_mode="reference_only_equivariant",
        ).eval()
        with torch.no_grad():
            output = model.fusion[-1]
            assert isinstance(output, torch.nn.Linear)
            torch.nn.init.normal_(output.weight, std=0.02)
            torch.nn.init.normal_(output.bias, std=0.02)
            target = torch.randn(2, 5, 8, 32)
            reference = torch.randn_like(target)
            perturbation = 0.05 * torch.randn_like(target)
            baseline = model(target, reference)
            perturbed = model(target + perturbation, reference)
            reference_perturbation = 0.2 * torch.randn_like(reference)
            shifted_reference = model(target, reference + reference_perturbation)
        torch.testing.assert_close(perturbed - baseline, perturbation, rtol=1.0e-5, atol=1.0e-6)
        assert not torch.allclose(shifted_reference, baseline)
    
        differentiable_target = torch.randn(1, 3, 2, 32, requires_grad=True)
        cotangent = torch.randn_like(differentiable_target)
>       (model(differentiable_target, reference[:1, :3]) * cotangent).sum().backward()
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/unit/test_fp_naa_adapter.py:190: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../../miniconda3/envs/care-asd-fp-naa/lib/python3.11/site-packages/torch/nn/modules/module.py:1739: in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../miniconda3/envs/care-asd-fp-naa/lib/python3.11/site-packages/torch/nn/modules/module.py:1750: in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = BandwiseReferenceAdapter(
  (target_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
  (reference_norm): La...ximate='none')
    (3): Dropout(p=0.0, inplace=False)
    (4): Linear(in_features=16, out_features=32, bias=True)
  )
)
target = tensor([[[[-0.0872,  0.2075, -0.0390, -1.1163, -0.4400, -0.5938, -1.5497,
           -0.2582, -0.6645,  0.3459,  1.684...92,  1.4645, -0.1191,  1.3751,  0.7426, -1.3382,
            1.5017,  1.0234, -1.3187,  0.1591]]]], requires_grad=True)
reference = tensor([[[[ 9.9582e-01, -1.1916e+00, -1.5410e-02,  5.2084e-01,  3.5880e-01,
            2.0676e-01, -1.7179e-01, -7.87...1,
            4.0279e-02, -5.2651e-01, -5.2828e-02, -4.8103e-01, -4.3769e-01,
           -1.2995e+00, -6.6103e-02]]]])

    def forward(self, target: Tensor, reference: Tensor) -> Tensor:
        """Return a target-shaped adapted grid conditioned on the reference grid."""
        if target.ndim != 4 or reference.ndim != 4:
            raise ValueError("target and reference must have shape [batch, time, band, embedding]")
        if target.shape != reference.shape:
>           raise ValueError("target and reference token grids must have equal shapes")
E           ValueError: target and reference token grids must have equal shapes

src/care_asd/models/fp_naa_adapter.py:84: ValueError
=========================== short test summary info ============================
FAILED tests/unit/test_fp_naa_adapter.py::test_reference_only_adapter_is_exactly_target_perturbation_equivariant
1 failed, 15 passed in 6.02s
{"error": {"code": "EXTERNAL_COMMAND_FAILED", "message": "An experiment subprocess failed", "details": {"command": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "pytest", "tests/unit/test_fp_naa_adapter.py", "tests/unit/test_fp_naa_candidate.py", "-q"], "returncode": 1}}}
```
