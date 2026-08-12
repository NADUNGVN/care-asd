# Phase 4 Safe CARE front-end

Safe CARE is a causal acoustic-path residual *view*, not a channel-replacement
algorithm. It always emits the untouched near-channel STFT, the far-channel
STFT, and a bounded residual. A downstream model may use all three views, but
it cannot lose the original near-channel evidence merely because cancellation
is inaccurate.

## Deterministic method

At each frame and frequency bin, causal EMA tracks cross-power and the two
auto-power terms. The regularized path estimate is `cross / (far_power +
reg_floor)`. Optional frequency smoothing operates only across frequency bins,
never future frames. The initial semi-parametric gate is:

`min + (max - min) * sigmoid(coherence_weight * (2*coherence - 1) + snr_weight * log((near_power + eps)/(far_power + eps)) + bias)`.

The cancelled component is then rescaled per frame to keep removed energy at
or below `max_removed_energy_ratio`. Bypass forces a zero gate and makes the
residual exactly equal to near.

## Stable feature contract

`CAREAudioFrontEnd` implements the same `FeatureBatch` contract as the Phase 3
controls. Its views are `near`, `far`, and `residual`; diagnostics are
coherence, gate, SNR proxy, path confidence, removed-energy ratio, log ratio,
phase sine/cosine, and transfer magnitude. The adapter was added without
changing the detailed `SafeCAREOutput` API.

## Development evidence

`care-asd care-benchmark-dev` uses exactly the fixed normal-only log-spectral
reference scorer used by Phase 3, so its result is a front-end comparison—not
a claim about a learned CARE encoder. It writes immutable score/metric files,
over-cancellation data, frequency-band statistics, and a dependency-free SVG
plot of coherence, gate, and path confidence. The configuration and run inputs
are recorded in `benchmark.json`.

The next decision point is empirical: CARE must exceed early concatenation and
at least one DSP residual control without strong degradation across machine
types before its learned multi-view encoder is pursued in Phase 5.
