"""Signal processing: STFT, coherence, transfer, residual (Phases 3-4)."""

from care_asd.signal.safe_care import SafeCAREFrontEnd, SafeCAREOutput
from care_asd.signal.synthetic import SyntheticStereoCase, simulate_stereo_case

__all__ = [
    "SafeCAREFrontEnd",
    "SafeCAREOutput",
    "SyntheticStereoCase",
    "simulate_stereo_case",
]
