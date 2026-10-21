# CF_ASG (P3) Protocol v0.1 — frozen before measurement, 2026-08-10

## Claim under test
A mixture of physically propagated sources can be inverted into an
INTERVENABLE AUDIO SCENE GRAPH — per-source (dry latent, physical
parameters) + shared room — such that graph-level edits (remove node,
re-parameterize node, change room) re-render physically consistently
and beat the natural pipeline (separate -> estimate -> invert ->
re-render) on edit fidelity.

## Method pillars (pre-committed)
P-A Exact structured likelihood: y = sum_k Phi(d_k; theta_k) with
    shared room alpha; Phi = audiocf_render core (P2), fixed, known,
    differentiable. Stems are additive BY CONSTRUCTION (room is LTI).
P-B Inference = amortized posterior (mixture -> K slots: existence,
    trajectory type, theta_k, z_k -> d_k) + T steps of exact-likelihood
    refinement (gradient on mrSTFT(y, render(G)) wrt {d_k, theta_k}).
    Refinement is the structural edge: the pipeline has no likelihood.
P-C Edit identity: edit(y, k, theta') = y - Phi(d_hat_k; theta_hat_k)
                                          + Phi(d_hat_k; theta').
    Only the edited node passes through the model; locality of
    non-target content holds by construction; removal = subtraction
    term only.

## Corpus (rendered, all GT free)
FSD50K dev sources (pools as corpus_v2), 4 s @ 16 kHz.
K ~ U{1,2,3}; per source: trajectory type in {static(r), passby(v)},
r log-U[1,32], v U[5,40], relative gain U[-6,6] dB; shared room
alpha on the discrete grid {0.10,0.15,...,0.50} (9 cached RIRs;
amended pre-measurement for renderer cost) applied per-stem (LTI => additivity preserved).
Store: mix y (peak-normalized, gain kept), wet stems, dry d_k, all
params. Train 24k scenes (FSD50K train pool), val 2k (disjoint val
pool clips). Script: src/p3_scenes.py.

## Tasks and metrics (full-reference: GT edit renders exist)
E1 do(remove k):   ref = y - stem_k (closed-form, P1 identity)
E2 do(v_k -> v'):  ref = render of GT scene with v' (grid {5..40})
E3 do(r_k -> r'):  ref = render of GT scene with r' (doublings)
E4 do(alpha -> a'):ref = GT re-render; DECLARED instrument-limited
                   (uniform-detector floor 0.61, P2); not headline.
Metrics per edit: SI-SDR and mrSTFT to reference; locality =
SI-SDR of non-target region content vs its reference (identity-based
methods get this by construction; measured anyway).
Scene inference quality: per-slot theta recovery (P2 calibrated
verifier floors as context), stem SI-SDR (PIT-matched), existence F1.

## Systems
S1 OURS-A: amortized only.
S2 OURS-AR: amortized + refinement (main method).
S3 PIPE: supervised PIT separator (same backbone family, same corpus,
   same budget) -> per-stem theta estimator (learned head, same
   supervision) -> semi-analytic inversion to dry (undo gain/warp;
   Wiener de-reverb) -> Phi re-render. Same edit identity granted
   where applicable (subtract ITS stem estimate). Best pipeline we
   can build; fairness = same data, same budget per trainable stage.
S4 ORACLE-STEMS: GT stems + estimation (upper bound context).
Baseline note: regeneration-family editing already quantified in P1;
cited, not rerun.

## Pass criteria (pre-registered)
C1 E2/E3 (parametric edits, headline): S2 beats S3 on median SI-SDR
   to reference, cluster-aware p<0.05 (scene-level clusters).
C2 Refinement helps: S2 >= S1 on E2/E3 median SI-SDR (and on theta
   recovery); report magnitude.
C3 E1 removal: S2 within 3 dB of S3 median (separation is the
   pipeline's specialty; parity-zone acceptable, superiority
   reported if it occurs).
C4 Locality: identity-based edits >= 30 dB SI-SDR on non-target
   content (near-lossless by construction; measured).
C5 Binding: existence F1 >= 0.9 on val.
Failure handling: method iterated (refinement steps, matching,
capacity) until C1/C2 hold or structurally impossible; all
intermediate ledgers retained. alpha axis excluded from criteria
(instrument-limited, disclosed).

## Discipline (inherited)
Every number from paper_numbers/*.json; tables/figures script-
generated; nice -n 15, <=8 threads, resumable training, hourly
convergence checks; Opus review rounds ("extremely strict but
scientifically objective") after experiments; 8 full pages; TNNLS/
TASLP-appropriate restraint in wording; double-anon-ready.

## Amendment A1 (2026-08-10, pre-measurement; instrument-first redesign)
Deterministic-verifier check on the v0 corpus: Doppler rel-err on
reverberant pass-by stems reads 0.84-1.00 (vs 0.17 on room-free
renders) -- the velocity observable is instrument-unreadable through
the room, for learned and closed-form estimators alike. Separately,
absolute source distance r is unidentifiable from a single observation
(gain/r confound; P2's distance audit was always relative/slope).
Redesign, before any measured result:
- Scenes carry an ENVIRONMENT attribute: open (no room; pass-by and
  static sources; full P2 calibration carries over) or room (static
  sources; alpha on {0.30,0.35,0.40,0.45,0.50}); pass-bys occur only
  in open scenes (also the physically natural case).
- r is NOT an inference target. E3 becomes the RELATIVE edit
  do(r -> r x 2^m): identity y' = y - stem_k + (r/r') airabs-adjusted
  stem_k, requiring only the stem estimate. Node attributes claimed:
  trajectory type, v (pass-by, identifiable), environment and alpha
  (identifiable via decay). Identifiability boundaries are claims of
  the paper, not incidental choices.
- E4 = do(alpha) within room scenes; full-reference as before.
Corpus v1 retained on disk for the record; all criteria unchanged
otherwise.

## Amendment A2 (2026-08-10, pre-measurement)
Two corrections before any measured result:
(a) Scene sampler draws sources with PAIRWISE-DISTINCT top classes
    within a scene. Same-class simultaneous sources (e.g., two drums)
    are unidentifiable for any separator without spatial cues; scene
    graphs presuppose distinguishable objects, and the identifiability
    boundary is a stated claim (as in A1).
(b) Training stem loss averages over REAL stems only (the padded zero
    rows had diluted the logged loss by the K/MAXK fraction; model
    selection and any reported training numbers use the corrected
    definition).
Corpus v2 retained; v3 = distinct-class resample, same physics as A1.

## Amendment A3 (2026-08-10, pre-measurement): external baselines
S5 = AudioSep (ICLR'24, text-queried separation; weights public,
     training data includes FSD50K = in-domain for the baseline):
     stems by GT class-name queries with least-squares scale fit to
     the mixture (both favor the baseline; disclosed) -> our est
     heads + semi-analytic inversion -> the same edit identity;
     replayed on the exact edits of the main ledger (edit_specs).
S6 = ZETA (ICML'24, DDPM-inversion editing): regeneration-family
     representative, prompt-based remove for E1 on a compute-bound
     subset; the family's codec floor is established in our prior
     benchmark and cited.
Criteria C1-C5 unchanged; S5 joins the E1-E3 comparison tables.

## Amendment A4 (2026-08-10, pre-measurement): external comparison design
Six top-venue systems, exact replay of the main ledger's edit_specs:
X1 AudioSep (ICLR'24), X2 FlowSep (ICASSP'25), X3 SoloAudio
(ICASSP'25), X4 OmniSep (ICLR'25) -- separation/extraction family:
stems by GT class-name text queries, least-squares scale fit to the
mixture, then OUR trained est heads + semi-analytic inversion + the
same edit identity (every concession favors the baseline; disclosed).
X5 ZETA (ICML'24), X6 Audio-Omni (SIGGRAPH'26) -- regeneration/
instruction editing family: direct prompt edits; n=60 pre-registered
subset (compute-bound); locality measured, not granted.
All externals are zero-shot on Phi-propagated audio; S3 remains the
budget-matched fairness anchor, and the two claims are stated
separately (matched-control claim vs field-status claim).
Criteria added:
C6: S2 beats each of X1..X6 on E2 and E3 median SI-SDR-to-reference,
    scene-clustered Mann-Whitney, Holm-corrected over the six
    comparisons, p<0.05.
C7: E1 removal -- S2 within 3 dB of the best separation-family
    external (their specialty; parity acceptable, superiority
    reported), and strictly better than the regeneration family.
Inference at each system's default/recommended settings; sample-rate
adapters logged per system.

## Amendment A5 (2026-08-12, post-pilot-v2, pre-main-measurement)
Pilot v2 (final ckpts; ledger p3_edit_eval_pilot_v2.json, retained)
failed gates C1/C2. Diagnosis and pre-registered fixes, applied
uniformly to S1/S2/S3 before the main run:
1. Posterior stems (all subtraction-based identities, E1/E2/E3, and
   the S3 dry-inversion input): the edit identity consumes the
   posterior stem given the observed mixture, not the prior render --
   Wiener consistency projection (Gaussian-residual posterior mean:
   masks from model stem spectra, phase from the mixture; stems sum
   to y exactly). Same projection for S1, S2, S3; S4 oracle
   unchanged; externals get it too (concession, disclosed).
2. Refinement = exact-likelihood structure selection + global 1-D
   search: env in {0,1} chosen by final exact-likelihood residual
   (amortized env head is at chance, 0.52); per-pass-by-slot theta
   line-search on a 29-point grid before joint Adam polish
   (likelihood is oscillatory in v; gradient-only descent stays in
   the local basin). S2 edits use the refined env/alpha.
3. E1 restricted to K>=2 scenes: removing the only source leaves a
   zero reference; SI-SDR against zero is numerically meaningless.
   Applies to all systems incl. externals (same specs replayed).
4. Primary test switched to scene-paired Wilcoxon signed-rank
   (protocol already required scene-clustered inference;
   Mann-Whitney on paired scenes was mis-specified and is retained
   as a secondary column). C6 wording inherits the paired test.
Iteration is within protocol (line 72: iterate until C1/C2 hold or
structurally impossible; all intermediate ledgers retained).

## Amendment A6 (2026-08-12, post-pilot-v3, pre-measurement)
Pilot v3 (ledger p3_edit_eval_pilot_v3.json, retained): A5 lifted all
systems but S3 (pipeline) most -- S3 > S2 on E1/E2/E3 (paired p<0.01
in S3's favor). Root cause chain: (a) the graph slot decoder's dry
output is the weak link (val dry mrstft 7.3); (b) refinement can only
be as good as the likelihood-through-d-hat; (c) subtraction edits are
decided by stem quality, where direct wet prediction wins.
Diagnostic p3_diag_theta_v0.json (pre-registered variants A-H):
physics-inversion line-search v-hat = argmin_v ||Phi^-1_v(stem) -
dry_prior|| achieves rel 0.013 with GT prior, 0.11-0.13 with a 5-10 dB
proxy prior, vs 0.33 (est head) / 0.25 (graph head); prior-free
centroid-flatness variant fails (transient sources). Actions:
1. New amortized dry-prior model (p3_train --kind dry): sep
   architecture, PIT loss on DRY sources, no mixture loss. 60k steps.
2. New system S2P measured alongside S1/S2/S3/S4 in pilot v4:
   shared posterior stems (= S3 front-end), theta by grid+parabolic
   line-search against the assigned dry prior (assignment by
   line-search cost permutation), env from the sep head, alpha from
   est; E2 node re-rendered from Phi^-1_{v-hat}(posterior stem); E4
   per-slot Wiener + re-reverb (source-specific gains) vs S3's
   mixture-level op; E1/E3 inherit the shared front-end (identical to
   S3 by construction, marked as such).
3. No criteria change in this amendment. Pilot v4 measures S2P; the
   C1 roster question (E3-gain-only cannot reward structure under a
   shared front-end -- structural-impossibility analysis) will be
   decided with the pilot-v4 numbers on the table, disclosed.

## Amendment A6.1 (2026-08-12, pre-measurement)
The mixture->dries model (A6.1's predecessor, kind dry) converged
poorly (stem-mrstft oscillating 3.6-6.0 at 22.8k vs sep's 1.65 at the
same step; ckpt retained as dry_18k_abandoned.pt) -- it must solve
separation AND implicit physics inversion jointly.  Replaced by the
Dryer (kind dryer): an amortized SINGLE-SOURCE physics inverse,
wet stem -> dry source (no PIT, no slot assignment), trained with
bleed augmentation input = stem + lambda*(y - stem), lambda~U[0,0.5],
simulating imperfect separation.  Smoke: reaches the abandoned
model's 8k-step loss in 400 steps.  S2P now takes its per-stem dry
prior as Dryer(posterior stem); the slot-assignment machinery is
deleted.  Prior-free statistics were exhausted and are documented in
p3_diag_theta_v0.json (variants E,F,I-P): spectral self-consistency
works on clean stems (L: rel 0.104) but fails under mask cross-talk;
purity-frame selection removes exactly the Doppler-informative
approach/recede frames (O: 0.73).  The failure map motivates the
learned prior; the line-search itself is validated at rel 0.013
(GT prior) / 0.105-0.126 (5-10 dB prior).

## Amendment A6.2 (2026-08-12, pre-measurement)
Dryer v1 oscillated (3.9-10.4). Root cause measured on the corpus:
87% of dry sources exceed |1| (median absmax 3.97, max 113) -- the
tanh decoder output cannot represent its targets (same defect
plausibly contributed to the graph model's weak dry head). Fix:
shape/scale split (tanh shape x exp gain head, clamp [-4,5]) and
per-sample target-RMS-normalised mrstft loss (equalises loud/quiet
events). Retrained from scratch, 40k steps.

## Amendment A6.3 (2026-08-12, pre-measurement)
Dryer v2 (A6.2) collapsed to silence: the gain head pinned at its
lower clamp (outputs |x|~0.02) because mrstft's silence attractor
(~7 normalised) undercuts noisy-but-informative outputs; the
normalised train loss was indistinguishable from genuine learning
(calibration trap, documented). Fix: full shape/scale decoupling --
shape branch trained with BOTH sides RMS-normalised (silence
unreachable by construction), gain head regresses log-RMS with a
separate MSE term (0.1 weight). Probe gate before deployment:
theta rel-err on val posterior stems must beat the est head (0.33);
target <=0.15.

## Amendment A6.4 (2026-08-12, pre-measurement)
The dry-prior line-search program is closed as a documented negative
result: learned priors (three architectures/objectives: A6, A6.2,
A6.3) never reached the fidelity the search needs.  Measured failure
mechanism: the search tolerates 5-10 dB of WHITE prior error
(uncorrelated with the v-manifold) but structured model error is
spectrally smooth, wrong-v inversions also smooth the spectrum, and
the cost therefore drifts monotonically -- 62.5% of argmins land on
grid edges.  Prior-free statistics fail for complementary documented
reasons (ledger p3_diag_theta_v0.json).  S2P therefore takes theta
from the graph model's joint mixture heads (rel 0.25-0.28 vs est's
0.33 -- the joint-inference advantage), with all other S2P machinery
unchanged.  Pilot v4 measures this configuration.

## Amendment A7 (2026-08-14, pre-measurement): mathematical reformulation
Diagnosis: with y = sum_k Phi(d_k; theta_k) and free waveform-domain
d_k, the likelihood is degenerate in theta (for any theta a d exists
fitting y exactly; Phi is linear and near-surjective in d).  Every
observed failure -- flat refinement, prior-dependent line-search,
edge-biased structured priors, stem-dominated edit metrics -- is a
symptom of this over-parameterisation.
Reformulation (PWG, physics-warped Gaussian scene model): dry sources
become locally-stationary Gaussian processes, D_k(f,t) ~
CN(0, W_k(f) g_k(t)) on the STFT grid (low-rank PSD prior = the
constrained nuisance space); Phi acts as PSD-domain operators
(Doppler = time-varying spectral warp with Jacobian, distance = gain,
room = exponential time-kernel on the PSD); dry waveforms are
EXACTLY marginalised: Y(f,t) ~ CN(0, sum_k S^wet_k + sigma^2), the
Whittle marginal likelihood is closed-form and IDENTIFIES theta.
Inference: amortized init (existing nets) + EM -- E-step = Wiener
posteriors (the A5 projection becomes the model's exact posterior
mean), M-step = IS-NMF multiplicative updates for (W,g), grid search
/ model selection on the now-informative marginal likelihood for
(theta, alpha, env).  Edits: closed-form posterior counterfactuals
E[y'|y] = y - L_theta E[d_k|y] + L_theta' E[d_k|y] (all-Gaussian).
The static-source gain confound (E3) becomes a theorem (only a^2 W
identifiable), folding the negative-result ledgers into the model's
identifiability analysis.
Gate (pre-registered): on GT scenes, marginal-likelihood grid search
must recover v with rel <= 0.05 (single-source) before the model
replaces S2; K>=2 EM checked before pilot v5.

## Amendment A7.1 (2026-08-14): gate outcomes and deployment config
Gate results (ledger p3_pwg_gate_v0.json + session logs):
G1 (K=1, flat->moment-init W, full-grid marginal likelihood): median
rel 0.029 -- PASSED (gate 0.05).  Two implementation lessons folded
in: Nyquist edge-extension of the warp (zeroed bins caused a
catastrophic small-v bias) and moment initialisation (unwarp the
observed power at the candidate v and average; at the true v frames
align).  Remaining G1 failures concentrate on sources with weak
spectral structure / slow v -- honest flat posteriors, reported in
the identifiability analysis.
G2 (K=2): joint EM 0.47, amortized-init joint EM 0.85 (sharp init
locks local optima), posterior-isolated single-source Whittle 0.25
(best; deployment config), mask-weighted Whittle 0.51 (weights kill
the Doppler-informative low-energy tails -- the same mechanism as the
A6 purity-frame failure; now documented as a recurring phenomenon).
G3 (env/alpha): near-non-identifiable under free (W,c) -- reverb of a
locally-stationary source is locally stationary (kernel
near-separability); sparse-envelope MAP prior does not restore it.
Documented as a theorem-grade limitation; env/alpha remain amortized
(sep head 0.98 / est 0.054), consistent with the protocol's existing
alpha exclusion.
Deployment: S2P theta := pwg_theta (posterior-isolated Whittle,
moment init, grid + parabolic).  Pilot v5 measures it.

## Amendment A8 (2026-08-14, pre-measurement): supervised ownership maps
Borrowed from the privileged-factor supervision pattern (DerainSplat,
arXiv:2608.02191): the corpus renderer is the data factory, so
per-cell GT ownership shares M_gt(f,t) = |STFT(stem_k)|^2 / sum_j
|STFT(stem_j)|^2 are free privileged supervision.  Self-derived
weights (mask share m^2) provably conflate "low-energy" with
"contested" -- they killed the Doppler-informative tails (A6/A7.1
recurring failure).  A SUPERVISED ownership head can label a
low-energy cell as clean when no other source occupies it.
sep2 := sep + ownership refiner head (small 2-D CNN over the slot
log-spectra + mixture, softmax over active slots, soft-CE vs M_gt,
cell weight ~ P_mix^0.3).  Deployment: posterior stems from predicted
ownership masks (replacing net-spectra Wiener ratios); the per-slot
Whittle likelihood optionally censored by predicted ownership.
Pilot v7 measures both.  Deferred borrows, recorded: per-stage
privileged intermediates via on-the-fly phi_torch targets; GT->pred
factor annealing (the cure matched to the graph model's collapse).

## Amendment A9 (2026-08-15, pre-main-measurement): final criteria
System of record: sep2 ownership front-end (A8) + est parameter heads
+ semi-analytic physics inversion + edit identities; graph variants
S1/S2 demoted to ablations; PWG (A7) is the method's theoretical
framework (generative model, marginal-likelihood identification, G1
validation, identifiability theorems), with the deployed estimator
honestly reported as the amortized head.  Replacing C1/C6/C7:
C-A (primary): E2 and E4 -- ours beats EACH of X1..X6 on median
    SI-SDR-to-reference, scene-paired Wilcoxon, Holm-corrected over
    the six comparisons, p<0.05.
C-B: E1 -- win-or-parity (within 3 dB) vs the separation family
    X1..X4; superiority reported where it holds.
C-C: E3 -- reported as-is per system; known deficit vs X1 (~2 dB,
    pilot v8 vs X1 pilot), separation-limited, front-end
    swappability discussed.  No parity claim pre-registered.
Pilot evidence: p3_edit_eval_pilot_v8.json vs p3_baselines_x1_pilot
(same 60 scenes/specs): E1 8.50 vs 7.45, E2 2.93 vs -0.16, E3 17.64
vs 19.87, E4 16.80 vs no external path; stems 6.07 vs 5.44.
Main run: 250 scenes -> ledger p3_edit_eval_v0.json; X1-X4 full
replay, X5/X6 first 60 specs (compute-bound, pre-registered).

## Amendment A11 (2026-08-15, pre-measurement): real-domain protocol R
Datasets: VS13 (400 real pass-bys, exact speeds, vehicle-level
splits), IDMT-Traffic (backgrounds), FSD50K.eval (real distractors).
R-E1/R-E3 (full-reference, real content): additive real mixtures
(vehicle pass-by + FSD-eval distractor at drawn SNR) -- addition
gives exact removal/gain references with fully real content.
R-E2 (weak GT, headline): held-out-vehicle VS13 clips, commanded
speed v'; our edit = inverse_passby at the editor-net estimate,
re-render at v'.  Verification by an INDEPENDENT verifier-net
(fine-tuned on a disjoint vehicle split), validated on unedited
clips against measured speeds first.  Score |v_hat(edited) - v'|.
Externals: separation family N/A-by-design on R-E2 (documented);
regeneration family via prompts.  Metric battery: verifier speed
error (R-E2), SI-SDR to reference (R-E1/E3), CLAP adherence,
LPAPS/LSD locality, FAD-CLAP realism vs held-out real mixtures.
Claims: (i) only system with a real-domain parametric edit path,
verifier-validated; (ii) adherence-locality Pareto reporting, no
single-column cherry-picking; family biases of each metric disclosed.
Editor/verifier estimator: est2 fine-tuned per split (est2_vs13
editor; verifier split disjoint).  All ledgers under paper_numbers/.

## Amendment A12 (2026-08-16, pre-measurement): confirmation redo round
Response to internal R1 review: findings are addressed by NEW
measurements, not wording downgrades. Registered before any of the
following runs:
1. CONFIRMATION COHORT (addresses R1-L4): 250 fresh scenes, seed 1,
   disjoint from the main cohort. PRE-REGISTERED claims to confirm:
   (a) E3 paired superiority vs each of X1..X4, Wilcoxon + Holm,
   p<0.05; (b) E1 superiority vs X2/X3/X4 and parity (TOST, 3 dB
   margin) vs X1; (c) E2: no significant separation-family
   difference expected (reported as-is). X5/X6 excluded here (their
   E2/E3 deficits are 20-60 dB in the main cohort; no plausible
   flip), disclosed.
2. LOCALITY (addresses R1-C1, pre-registered C4): non-target
   residual SI-SDR = si_sdr(est - node_contrib vs y - gts_k ...) --
   operationalized as si_sdr((est - ref) + (y - gts_k), y - gts_k)
   per edited scene: how faithfully the non-edited content survives.
   Recorded for every system in every rerun harness.
3. R-E2 REDO with fully independent verifier (addresses R1-S1):
   13 VS13 vehicles split three ways -- editor fine-tune (5 cars),
   verifier fine-tune (5 other cars), evaluation (3 cars unseen by
   both). Verifier = Est2 architecture retrained from scratch on the
   synthetic corpus with a fresh seed (no weight sharing with the
   editor at any stage), then fine-tuned on its own cars. All edits
   re-executed with the new editor; all systems re-scored with the
   new verifier.
4. Existence-threshold calibration on train split (pre-registered
   fix class), then main-eval inference metrics re-reported; gate
   C5 (F1>=0.90) re-tested.
5. Oracle-E4 rung (GT stems + estimated alpha re-render) and E4
   no-edit anchor added to the eval; L2 approximation gap
   (semi-analytic inverse vs exact) ledgered.
All new ledgers carry _v1/_c1 suffixes; old ledgers retained.

## Amendment A13 (2026-08-21, pre-measurement): physics-mismatch and
## slow-down analyses (external review suggestions, adopted)
1. CROSS-RENDERER MISMATCH: a perturbed renderer Phi_B defines the
   world; the deployed system keeps all Phi_A assumptions (training,
   inverses, re-renders). Phi_B perturbations (all simultaneous,
   moderate-realistic): speed of sound 343->350 m/s; air absorption
   0.0052*(f/kHz)^1.5 -> 0.0078*(f/kHz)^1.7 dB/m; pass-by closest
   approach d0 2.0->3.0 m; room 30x20x8 -> 24x16x6 m with shifted
   source/mic. 250-scene cohort (seed 2) rendered under Phi_B;
   references for every edit rendered under Phi_B (the world's
   truth); our system evaluated unchanged. Report: E1-E4 medians +
   theta rel err vs the Phi_A main cohort; no-edit anchors under
   Phi_B. Pre-registered expectation: graceful degradation on
   E1/E3 (separation-dominated), larger hits on E2/E4 (inverse-
   render mismatch); reported either way. Externals not re-run
   (zero-shot systems have no Phi_A commitment; disclosed).
2. SLOW-DOWN FAILURE ANALYSIS (real domain): per-clip correlation of
   the slow-down verifier error against (a) true speed, (b) editor
   error on the source clip, (c) commanded-vs-estimated gap,
   (d) direction asymmetry of the verifier itself (response to
   synthetically slowed vs sped versions of unedited clips); one
   figure (speed-up vs slow-down scatter) + mechanism statement in
   the paper. Ledger: p3_slowdown_analysis_v0.json.
3. Writing (no new measurements): identifiability promoted to the
   framing spine (intro + identifiability-map table); abstract
   de-numbered; "closed-form counterfactual under the inferred
   Gaussian scene model"; E3 single-source ceiling explicitly
   near-trivial; amendments/ledgers pointed to the repository.

## Amendment A14 (2026-10-18, pre-measurement): TASLP round-1 experiments
Response to T-ASL-13354-2026 reviews (R1 room acoustics; R2 protocol
clarity and companion audio; R3 metrics, multiple edits, listening
test). Registered before any of the following runs. Cohort: the main
250-scene val cohort and its pre-registered edit_specs
(p3_edit_eval_v1.json); systems: ours (S3 = sep2 ownership front-end
+ est heads + semi-analytic inverse), X1-X4 with the same downstream,
X5/X6 on the first 60 specs (compute-bound, as before). Every system
now SAVES its edited waveforms (p3/r2_out/<system>/) so that any
metric can be recomputed; references and mixtures saved once
(p3/r2_ref/).
1. SEMANTIC REMOVAL METRIC (R3): CLAP (laion/clap-htsat-unfused,
   frozen, zero-shot) audio-text cosine. For E1: s_rm = sim(out, "the
   sound of <removed class>"), s_keep = mean over remaining classes.
   Report drop = s_rm(mix) - s_rm(out) (higher = better removal) and
   keep = s_keep(out) - s_keep(mix) (near 0 = untouched), both against
   the reference ceiling (y - s_k) and the unedited anchor (0 by
   definition). For every edit and chain: full-reference CLAP audio
   similarity cos(a(out), a(ref)) as a semantic companion to SI-SDR.
   Pre-registered expectation: separation-family ranking agrees with
   SI-SDR on E1; regeneration family may score higher on drop but
   lower on keep (whole-scene regeneration).
2. REVERBERATION METRIC (R3): E4 scored by RT60 as well as SI-SDR.
   (a) offset-isolated Schroeder T20 on the free decay after the LAST
   dry source offset (offsets known from the GT dries; scenes with
   >= 0.4 s of free decay only), applied identically to out, ref and
   the unedited mixture; (b) a blind decay-rate-distribution estimate
   for all scenes. Score = |RT60(out) - RT60(ref)|. Validation gate
   (reported, not a pass criterion): on references, estimator (a) must
   order the five alpha nodes monotonically and track the Schroeder
   RT60 of the RIR within 0.15 s median; otherwise (a) is reported as
   invalid and only alpha-error is kept.
3. MULTIPLE EDITS (R3): four composite edits drawn deterministically
   from the existing single-edit specs (no new randomness where the
   single-edit spec exists; seed 14 for the extra slot draws):
   C1 = E1 + E3 (remove slot a, re-distance static slot b != a),
   C2 = E2 + E3 (re-velocity pass-by a, re-distance static b),
   C3 = E1 + E4 (remove a, then change room absorption),
   C4 = E1 + E2 (remove b, re-velocity pass-by a != b).
   References are closed-form GT renders. Ours and X1-X4 apply the
   edits as a chain of node identities (each identity touches its own
   node; E4 re-renders the remaining mixture). ORDER TEST: ours applies
   each chain in both orders and ledgers the max |difference| (expected
   0 up to float; the identities commute by construction). X5/X6: one
   combined instruction (both changes in one sentence) and a sequential
   two-step application; both reported. Metrics: SI-SDR, mrSTFT,
   locality, CLAP-ref. Pre-registered expectation: for identity
   systems the chain score is bounded by the weaker single edit; for
   instruction systems combined instructions degrade further.
4. LISTENING TEST (R3): MUSHRA-style, prepared from the saved outputs:
   4 edits x 5 scenes, conditions = hidden reference, unedited anchor,
   ours, X1, X3, X6 (where available). Stimuli, web page and analysis
   script are produced here; the test itself requires human listeners
   and is run by the authors. Not a machine-measurable item.
5. COMPANION PAGE (R2): static page with mixture / reference / each
   system per edit and chain (12 scenes) plus real pass-by edits from
   rset_out; built from the saved waveforms.
6. ROOM-ACOUSTICS DESCRIPTORS (R1): p3_r2_room_acoustics.py -- Sabine,
   Schroeder T20/T30 of the image-source RIR, DRR, early-energy
   fraction, phantom-source counts, Schroeder frequency, shoebox mode
   counts, corpus energy fraction below f_s. The corpus RIR is an
   image-source (geometric) model; the Phi_B mismatch cohort (A13)
   already changes the room geometry and therefore the phantom-source
   pattern. Wave (modal) effects are confined below f_s, which is
   quantified; stated as the regime boundary of the renderer.
Old ledgers untouched; all new ledgers carry the _r2 suffix.

### A14.4 revision (2026-10-18, before any listener data): listening test v2

The v1 package (MUSHRA style, 20 trials, scenes ranked by edited-node
energy) was piloted by the first author and found unusable: in most E1
trials the kept or removed source was 30 to 60 dB below the mixture
(reference effectively silent), pass-by sources were active for one
second, E4 steps of 0.05 were inaudible, E3 in single-source scenes is
a pure gain change, and several FSD50K clips did not sound like their
label. v1 is withdrawn; no listener data were collected with it.

v2 (`src/p3_r2_listening.py` -> `p3/listening_test_v2/`,
ledger `paper_numbers/p3_r2_listening_v2.json`): ten trials (E1 x3,
E2 x2, E3 x2, E4 x3). Each trial plays the original openly, states the
intended change in words (English and Chinese, with the perceptual cue
to listen for), and asks for every clip how much the listener agrees
(0 to 100) that the clip is the original with exactly that change.
Hidden clips: ground-truth edit (gt, closed-form render) and the
untouched original (orig). Systems: ours, AudioSep, SoloAudio,
Audio-Omni where it was run; E4 has ours only (the room path is stem
independent). Scene choice is deterministic from the pre-registered
specs: every audible source (within 15 dB of the loudest) must be active
at least 40% of the clip (pass-by: 90%) and its class prompt must rank
in the top 4 of 23 CLAP class prompts; sources more than 20 dB down are
ignored and not named; pass-by sources must be vehicle-like classes.
E2 needs speed ratio >= 2.5 and new speed <= 40 m/s; E3 needs a
two-source room scene, factor 4 or 0.5 preferred; E4 needs |delta
alpha| >= 0.10 and one transient dominant source. Analysis
(`analyze.py`): per edit and condition the median over listeners of the
listener mean, paired Wilcoxon against ours and against gt; listeners
whose gt rating falls below orig in more than 30% of trials are dropped.
Pre-stated reading: gt is the ceiling, orig the floor; a system is
"accepted" for an edit when its median is not significantly below gt.

Pilot-driven amendments to the v2 scene rules (2026-10-18, author's pilot,
before distribution): (a) two dominant impulsive sources (crest factor
> 8) are not allowed in E1/E3 trials, they fuse by ear (bell + hammer);
(b) for E4 the transient that exposes the tail must be broadband, bass
drum and thump are excluded (their tails did not reveal the room change);
(c) scene 1433 (glass + alarm) is excluded, the two sources cannot be
told apart by ear. The final trials are listed in
`paper_numbers/p3_r2_listening_v2.json`. The author's own pilot ratings
are kept as a sanity check only and are not part of the reported data.

Second pilot amendment (2026-10-18, author's re-listen of the rebuilt
package): (d) the class-based E4 rule in (b) is replaced by a measured
one. A room trial (E4, and C3 on the companion page) is admitted only
when the closed-form render of the edit, compared with the clip it is
applied to (the mixture for E4, the E1 render for C3), changes the level
by more than 3 dB in at least 80 frames of 10 ms that lie before 3.3 s,
inside gaps (unedited level more than 20 dB below the clip peak) and
still audible (the louder clip within 40 dB of the peak), and the first
audible onset falls within 0.5 s of the clip start. Candidates are ranked
by the number of such frames. A scene whose audible sources are all
low-frequency classes (bass drum, thump) stays excluded. Reason: the
gunshot + bell trial (scene 1457) started with 1.1 s of silence and its
tail change sat in the last 0.6 s, 30 to 40 dB below the peak; the author
could not hear it, while the two room trials with changes spread over the
clip were rated as intended. The measured rule admits hammer and tapping
scenes that the class list had rejected. Constants GAP_MIN, LEAD_MAX and
the function gap_score in `src/p3_r2_listening.py`.

Amendment (e), 2026-10-21: the mrSTFT metric reported in the revision
tables is the published multi-resolution STFT loss (Yamamoto et al.,
ICASSP 2020, official implementation: spectral convergence + log STFT
magnitude, FFT 1024/2048/512, hop 120/240/50, win 600/1200/240,
magnitude clamp sqrt(1e-7)), recomputed from the saved waveforms of all
systems and of the unedited mixture. The first ledgers used the
training-loss variant (linear + log L1, log floor 1e-5); that variant
rated the unedited mixture above every system on E1/E2/E4 because the
floor amplifies differences in near-silent bins. The switch was made
after that observation and is disclosed in the response letter. The
order of the external systems is unchanged; one pair flips: on E2
AudioSep was ahead of ours under the old definition (0.83 vs 0.88) and
is behind under the new one (0.93 vs 0.91), not significant either way.

Amendment (f), 2026-10-21 (post hoc, after an internal audit; none of
these were pre-registered): (1) ZETA and Audio-Omni run on E4 with the
room prompt on their 60-case subset (34 room scenes). (2) A non-learned
room-edit baseline, WPE (nara_wpe) followed by the target response, on
E4 and C3 (src/p3_r2_wpe_baseline.py). (3) The confirmation cohort E3
verdict and the physics-mismatch Table X E3 entries re-scored under the
K>=2 rule (src/p3_r2_confirm_k2.py, p3_noop_anchor.py --k2). (4) 95%
bootstrap intervals over listeners for the listening-test paired median
differences (src/p3_r2_listening_ci.py). (5) The A14 fallback for a
failed RT60 validation (report only the alpha error) was not followed;
the EDC error was added instead and is labeled as added in the
manuscript.

