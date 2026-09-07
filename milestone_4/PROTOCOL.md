# Milestone 4 fixed protocol v1

The treatment is **only the predicted probability vector supplied to
`milestone_3.planner.plan`**. Milestone 3 is frozen at commit
`6e0bce5977f15c0d2686aa1ba09b8c7fcc7ee547`. Its interpretation is limited to the
particular spatially biased prior inducing broader coverage in that setting.

## Invariants

| Component | Fixed definition |
|---|---|
| Workspace / obstacles | Unit square; rectangle `[0.43, 0.57] × [0.43, 0.57]`; conservative collision disk with 0.015 clearance |
| Ground truth | M3 `make_priors(Environment(), 40)['oracle']`; center `(0.75, 0.70)`, sigma 0.13, background density 0.03 |
| Support | Same 40×40 cell centers, with zero probability inside conservative collision disks; 1,532 free cells |
| Target sampling | `numpy.random.default_rng(seed).choice(..., p=true_prior)`; 64 targets per seed; paired by `(seed, episode)` across all error families, JS levels and gamma values |
| Pilot seeds | 7 and 11; seed 7 exactly reproduces the saved M3 target set |
| Camera / detector | Range 0.25, FoV 90°, pan `theta(t)=pi*t/2` modulo `2*pi`, initial phase zero; same ideal range/FoV/occlusion predicate |
| Budget / observation | 15 seconds; 0.05-second observation grid plus budget endpoint |
| Planner | Unchanged M3 reference adapter: SLSQP with JAX derivatives, 8×8 Fourier basis, 50 nodes, maxiter 600; gamma 0.10 and 0.05 |
| Motion | Same start `(0.1,0.1,0,0)`, end `(0.9,0.9,0,0)`, initial guess, dynamics, control/speed/workspace/collision constraints |
| Execution | M3 `dt=tf/N` node interpolation, terminal hold and continuing camera scan to budget; no replanning or posterior update |
| Evaluation | M3 `evaluate` and `coverage_series`; target coordinates never enter `plan` |
| Validation | Original solver-success, finite-value, equality/inequality and collision checks; invalid plans have no search metrics and cause a nonzero pilot exit |

Each predicted prior and gamma needs one deterministic plan. The same plan is
evaluated against both target seeds; seeds are not independent planner trials.
The pilot therefore has 24 error plans plus two oracle reference plans, not 52
independent optimization replicates. No camera phase, planner, scene, obstacle
or target-sampling change is part of this comparison.

## Error definitions

Let `F` be the fixed free-cell mask and `N(w)=F*w/sum(F*w)`. Write
`G(x;c,s)=exp(-||x-c||²/(2s²))`, `c=(0.75,0.70)`, `s=0.13`, `b=0.03`, and
`p=N(G(x;c,s)+b)` for the unchanged truth.

1. **Spatial shift:** `q_d=N(G(x;c+d*(-1,0),s)+b)`, `0 <= d <= 0.75`.
   Direction is fixed westward. `d` is the Gaussian center displacement in
   workspace units, not the displacement of the masked distribution's mean.
2. **Diffuse/blur:** `q_k=N(G(x;c,k*s)+b)`, `1 <= k <= 1000`.
   Center, peak amplitude and background density are fixed. The underlying
   Gaussian variance multiplier is `k²`; truncation, masking and normalization
   mean this is not the empirical variance multiplier of the final vector.
   This is sigma broadening, not a convolution with a different boundary rule.
3. **False hotspot:** `q_a=(1-a)*p+a*h`, `0 <= a <= 1`, where
   `h=N(G(x;(0.25,0.70),0.13))`. The false component has no background;
   `a` is its normalized mixture mass. The true component remains centered at
   its original position.
4. **False-negative suppression:** `q_a=N(p*(1-a*I_R))`, `0 <= a <= 1`,
   where `R` is the fixed disk of radius 0.25 centered on the true hotspot.
   This disk contains 73.7035% of the original probability. It is a proper
   subset of the support. Inside it, all original mass (including background)
   is multiplied by `1-a`; outside it, relative probabilities are preserved.
   The hard region boundary can produce a ring after normalization, an explicit
   feature of this error design. No hotspot relocation is introduced here.

At `d=0`, `k=1` or `a=0`, every family returns exactly the oracle vector.
All vectors are finite, nonnegative, normalized and zero on blocked cells.
`ErrorDesign` records the geometry; changing direction, false-hotspot position
or suppression region requires a separately labelled protocol and calibration.
None of those choices is tuned using pilot outcomes.

## Numerical JS calibration

Use divergence, not its square root:

`JS(p,q) = 0.5*sum(p*log(p/m)) + 0.5*sum(q*log(q/m))`, `m=(p+q)/2`.

Logs are natural (nats); zero terms are skipped exactly, without adding
smoothing mass. Calibration uses the complete fixed grid, not target samples.
Scan 513 parameter values (log-spaced for sigma scale; linear otherwise), then
use Brent's method on the first bracketed crossing. Require absolute JS error
`<= 1e-6` nats, recomputed on the saved normalized vector. A missing crossing
is recorded as `unattainable_in_scan`, with no substituted parameter or prior.
The reported scan maximum is numerical evidence on the configured range, not
a global optimization certificate for arbitrary geometry.

The proposed example `0.05, 0.15, 0.30` is audited separately. With this blur
definition, the uniform free-cell limit is JS **0.17805172492048** nats; the
configured scan cannot reach 0.30. The shared pilot levels are therefore
**0.05, 0.10, 0.15 nats**. Do not change the truth or mix in another error family
to force the blur row to 0.30. Matching JS controls this scalar error magnitude;
it does not imply matching spatial geometry or establish universal rankings.

## Outcomes and gate before expansion

- Success and timeout rates, with valid-plan episode counts.
- Successful-only mean and median `T_find`; timeout times remain blank.
- Free-cell coverage fraction and true-prior mass covered at budget, using the
  original coverage implementation and observation protocol; full time series
  are retained. Coverage is a property of the plan, so it is saved once per
  condition, not treated as a different measurement for each target seed.
- Oracle-versus-error paired outcomes: both success, oracle only, error only,
  both timeout, keyed by gamma, error condition, seed and episode. An invalid
  oracle or error plan produces no paired search outcomes for that comparison.
- Planner validity, residuals, terminal time and planning wall time reported
  separately. Capped-time metrics are retained as secondary M3 outputs.

First require complete calibration, valid priors, valid trajectories and a
passed pairing/count audit. The single-scene, two-target-seed pilot checks the
pipeline and does not support formal significance or general error-family
rankings. Multi-scene / larger-seed experiments are deferred; they require a
separately fixed extension of this protocol.
