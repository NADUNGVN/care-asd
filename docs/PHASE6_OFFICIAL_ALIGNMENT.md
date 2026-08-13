# Phase 6: official baseline alignment

The retained Phase 5 CARE multi-view autoencoder did not meet its pre-registered
three-seed AUC criterion. Phase 6 therefore does not test CARE. It first checks
that the internal research runner can reproduce the known official MSE baseline
under the same public development split.

## Locked contract

- Stereo input channel 0 only, as in the pinned official loader.
- `librosa.melspectrogram`: 128 Mel bins, `n_fft=1024`, `hop_length=512`,
  default centred framing, power 2.
- `10*log10(max(Mel, eps))`, stacked across five consecutive frames (640 values).
- Fully connected AE: `640 → 128 → 128 → 128 → 128 → 8 → 128 → 128 → 128 → 128 → 640`,
  BatchNorm (`eps=1e-3`, `momentum=0.01`) and ReLU.
- Adam `lr=0.001`, batch 256, 100 epochs, seed 13711; score is mean MSE over
  every feature vector belonging to a WAV.

## Pass condition

The internal aligned run must produce mean development AUC within 0.02 absolute
of the external official MSE reproduction (0.609729) and pAUC within 0.02 of
0.542331. A miss is an implementation/alignment defect, not evidence against
CARE. Only a pass authorizes a new controlled CARE augmentation.
