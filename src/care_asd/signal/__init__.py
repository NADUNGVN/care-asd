"""Signal processing: STFT, coherence, transfer, residual (Phases 3-4)."""

from care_asd.signal.ap_care import (
    APCAREController,
    APCAREOutput,
    APCAREProfile,
    causal_stft,
)
from care_asd.signal.dsp_baselines import (
    AudioFrontEnd,
    DSPFrontEnd,
    FeatureBatch,
    FrontEndName,
    available_dsp_frontends,
    create_dsp_frontend,
)
from care_asd.signal.pseudo_faults import (
    FaultFamily,
    PairedNoiseMixture,
    PseudoFault,
    inject_pseudo_fault,
    mix_paired_noise,
)
from care_asd.signal.safe_care import CAREAudioFrontEnd, SafeCAREFrontEnd, SafeCAREOutput
from care_asd.signal.synthetic import SyntheticStereoCase, simulate_stereo_case

__all__ = [
    "APCAREController",
    "APCAREOutput",
    "APCAREProfile",
    "AudioFrontEnd",
    "CAREAudioFrontEnd",
    "DSPFrontEnd",
    "FaultFamily",
    "FeatureBatch",
    "FrontEndName",
    "PairedNoiseMixture",
    "PseudoFault",
    "SafeCAREFrontEnd",
    "SafeCAREOutput",
    "SyntheticStereoCase",
    "available_dsp_frontends",
    "causal_stft",
    "create_dsp_frontend",
    "inject_pseudo_fault",
    "mix_paired_noise",
    "simulate_stereo_case",
]
