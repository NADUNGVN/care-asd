"""Signal processing: STFT, coherence, transfer, residual (Phases 3-4)."""

from care_asd.signal.dsp_baselines import (
    AudioFrontEnd,
    DSPFrontEnd,
    FeatureBatch,
    FrontEndName,
    available_dsp_frontends,
    create_dsp_frontend,
)
from care_asd.signal.safe_care import CAREAudioFrontEnd, SafeCAREFrontEnd, SafeCAREOutput
from care_asd.signal.synthetic import SyntheticStereoCase, simulate_stereo_case

__all__ = [
    "AudioFrontEnd",
    "CAREAudioFrontEnd",
    "DSPFrontEnd",
    "FeatureBatch",
    "FrontEndName",
    "SafeCAREFrontEnd",
    "SafeCAREOutput",
    "SyntheticStereoCase",
    "available_dsp_frontends",
    "create_dsp_frontend",
    "simulate_stereo_case",
]
