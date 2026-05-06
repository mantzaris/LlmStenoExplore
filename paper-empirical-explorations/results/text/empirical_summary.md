# CARTS Empirical Summary


## Experimental configuration

Profile: `paper_two_hour`

```json
{
  "allow_failures": false,
  "comparison_distributions_used": [
    "carts",
    "ordinary_greedy",
    "ordinary_sampled"
  ],
  "detector_bootstrap_samples": 300,
  "estimated_runtime_components_seconds_from_previous": {
    "experiment1_correctness": 585.5478346100062,
    "experiment2_rank_likelihood_detection": 397.3086936849995,
    "experiment3_key_collision_fibers": 3632.4688289187106,
    "experiment4_collision_stability": 204.17655935612856,
    "experiment5_noncommutativity": 2363.310494452002,
    "experiment6_robustness": 677.7241549166683
  },
  "estimated_runtime_seconds_from_previous": 7860.536565938515,
  "keep_cache_on_clear": true,
  "max_runtime_hours": 2.25,
  "max_runtime_seconds": 8100.0,
  "model_key": "llama3_8b_q4_k_m",
  "n_ctx": 4096,
  "n_gpu_layers": 0,
  "n_threads": null,
  "num_collision_payloads": 16,
  "num_commutativity_pairs": 36,
  "num_commutativity_payloads": 8,
  "num_keys": 60,
  "num_payload_key_pairs": 40,
  "num_payloads": 24,
  "num_robustness_cases": 20,
  "num_shrinkage_true_keys": 3,
  "num_stability_payloads": 16,
  "ordinary_sample_temperature": 0.8,
  "ordinary_sample_top_p": 0.95,
  "quick_mode": false,
  "random_seed": 123,
  "run_profile": "paper_two_hour",
  "secret_prefix": "",
  "tail_thresholds": [
    10,
    100,
    1000
  ],
  "time_budget_adjustments": [],
  "use_embeddings": false,
  "use_sklearn_detector": true
}
```

Tokenization convention:

- Non-empty prefixes are tokenized with add_bos=True and the first BOS token is dropped. Empty prefixes use the model BOS token as the minimal autoregressive context.
- Payload and stegotext display strings are tokenized as model.tokenize((' ' + text).encode('utf-8'), add_bos=True)[1:].
- Ranks are 1-indexed. Tokens are sorted by decreasing logit with ties broken by increasing token id.


## Implementation correctness result

Exact recovery succeeded in 40/40 payload-key pairs (rate 1.000, Wilson 95% CI [0.912, 1.000]). Reverse-direction recovery succeeded in 40/40 cases (rate 1.000, Wilson 95% CI [0.912, 1.000]). This verifies the implementation under the specified deterministic environment; the theorem itself is mathematical.


## Rank-likelihood and detection result

Mean normalized NLL was CARTS 7.3098, greedy 0.8695, and sampled 1.5207. Detector results: carts_vs_ordinary_greedy AUC 1.000 [1.000, 1.000]; carts_vs_ordinary_sampled AUC 1.000 [1.000, 1.000]. Greedy comparison is a weak baseline; sampled comparison is more realistic but still not a full cover-channel model. The tested CARTS outputs were separable from the chosen comparison distribution under the tested features; this is not a proof of steganographic insecurity.


## Key-collision result

K_adm size was 60. Collisions occurred in 0/16 transcripts (rate 0.000, Wilson 95% CI [0.000, 0.194]). The largest observed fiber had size 1. True-key containment was 16/16 (Wilson 95% CI [0.806, 1.000]). Under the tested Llama 3 8B configuration and finite key set of size 60, no key collisions were observed in 16 tested transcripts. This does not rule out collisions in the unrestricted prompt space; it only indicates that collisions were not found under this finite search procedure.


## Collision stability and candidate shrinkage result

Collision stability skipped because no collisions were found. Candidate shrinkage still ran; mean candidate set size after 16 transcripts was 1.000.


## Non-commutativity result

Exact sampled commuting pairs observed: 0/36. Median D_log was 0.1813; 5th/50th/95th percentiles were 0.1424/0.1813/0.2237. No impossibility claim is made if no exact commuting pairs were found.


## Robustness result

A single length-preserving token perturbation caused average decoded edit distance 4.2000 over 80 tested perturbations. The corruption rate was 80/80 (Wilson 95% CI [0.954, 1.000]). Average decoded edit distance by perturbation type: {'random_token_substitution': 3.95, 'nearby_rank_substitution': 3.8, 'adjacent_token_transposition': 4.6, 'punctuation_token_substitution': 4.45}. Average suffix corruption by type: {'random_token_substitution': 1.0, 'nearby_rank_substitution': 1.0, 'adjacent_token_transposition': 0.9928571428571429, 'punctuation_token_substitution': 1.0}. Base CARTS is exact but not robust to unmodified-stegotext assumptions under these perturbation types.


## Figure manifest

- `results/figures/exp1_correctness_bar.png`: Exact token-id recovery rates for decode and reverse directions.
- `results/figures/exp2_nll_boxplot.png`: Normalized NLL for CARTS, greedy, and sampled ordinary comparisons.
- `results/figures/exp2_mean_log_rank_cdf.png`: Mean log-rank CDF for CARTS, greedy, and sampled traces.
- `results/figures/exp2_tail_rank_vs_nll.png`: Tail-rank fraction versus normalized NLL by distribution type.
- `results/figures/exp2_positionwise_mean_log_rank_heatmap.png`: Position-wise mean log-rank for CARTS and ordinary baselines.
- `results/figures/exp2_detector_roc.png`: ROC curves for CARTS versus greedy and sampled comparison distributions.
- `results/figures/exp3_fiber_size_histogram.png`: Histogram of finite candidate fiber sizes.
- `results/figures/exp3_unique_vs_collision_bar.png`: Fraction of unique and colliding finite-key transcripts.
- `results/figures/exp3_search_time_vs_keyset_size.png`: Wall-clock exhaustive search time for each transcript at fixed K_adm size.
- `results/figures/exp4_candidate_set_size_vs_transcripts.png`: Mean finite candidate key set size after accumulating transcripts.
- `results/figures/exp5_commutation_rate_histogram.png`: Histogram of sampled exact commutation rates.
- `results/figures/exp5_commutation_distance_cdf.png`: Empirical CDF of mean metric commutation distances.
- `results/figures/exp5_commutation_heatmap.png`: Heatmap of sampled commutation rates among displayed keys.
- `results/figures/exp5_Nk_tau_curve.png`: Average number of key partners below a commutation-distance threshold.
- `results/figures/exp6_decoded_edit_distance_by_perturbation.png`: Decoded token edit distance by length-preserving perturbation type.
- `results/figures/exp6_perturb_position_vs_error_heatmap.png`: Mean suffix corruption fraction by perturbation type and position.
- `results/figures/exp6_token_agreement_after_perturbation.png`: Average decoded token agreement after the perturbation position.


## Table manifest

- `results/tables/payloads.csv`: Payload set and token lengths for this run.
- `results/tables/keys.csv`: Finite key set and context token lengths for this run.
- `results/tables/experiment1_correctness.csv`: Round-trip status for each payload-key pair with Wilson intervals.
- `results/tables/experiment2_rank_likelihood_detection.csv`: Rank-likelihood features for CARTS, greedy, and sampled comparison text.
- `results/tables/experiment2_detector_auc.csv`: Detector AUCs and bootstrap confidence intervals.
- `results/tables/experiment3_key_collision_fibers.csv`: Finite candidate fiber sizes and true-key containment.
- `results/tables/experiment3_largest_fibers.csv`: Largest observed finite candidate fibers.
- `results/tables/experiment3_search_cost.csv`: Exhaustive finite key search cost per transcript.
- `results/tables/experiment4_collision_stability.csv`: Collision stability across new payload rank vectors; empty if no collisions were found.
- `results/tables/experiment4_candidate_shrinkage.csv`: Candidate set size after accumulating transcripts.
- `results/tables/experiment5_noncommutativity.csv`: Sampled commutation rates and log-rank distances for key-induced maps.
- `results/tables/experiment5_Nk_tau_curve.csv`: Average number of highly commuting keys by metric threshold.
- `results/tables/experiment6_robustness.csv`: Decoded errors after length-preserving stegotext token perturbations.
- `results/tables/paper_verification_checklist.csv`: Theory-claim verification checklist for the paper.


## Limitations and caveats

- Results are under this model and tokenizer, not all LLMs.
- Finite key-search results are under this finite key set only.
- Greedy Q_k is a weak baseline; sampled Q_k is stronger but still not a full cover-channel model.
- Detector results do not prove steganographic security or insecurity.
- Exact correctness assumes identical tokenization, logits, masking, precision, and tie-breaking.


## Next steps

- If `paper_two_hour` finishes comfortably, consider a follow-up `paper_medium` run; do not launch `paper_full` automatically.
- Add stronger comparison distributions for Q_k.
- Add calibrated plausibility metrics before wrong-key equivocation experiments.
- Test another quantization or exact model variant only as a separately labeled run.


## Paper-ready paragraphs

- Exact recovery succeeded in 40/40 payload-key pairs (rate 1.000, Wilson 95% CI [0.912, 1.000]). Reverse-direction recovery succeeded in 40/40 cases (rate 1.000, Wilson 95% CI [0.912, 1.000]). This verifies the implementation under the specified deterministic environment; the theorem itself is mathematical.
- Mean normalized NLL was CARTS 7.3098, greedy 0.8695, and sampled 1.5207. Detector results: carts_vs_ordinary_greedy AUC 1.000 [1.000, 1.000]; carts_vs_ordinary_sampled AUC 1.000 [1.000, 1.000]. Greedy comparison is a weak baseline; sampled comparison is more realistic but still not a full cover-channel model. The tested CARTS outputs were separable from the chosen comparison distribution under the tested features; this is not a proof of steganographic insecurity.
- K_adm size was 60. Collisions occurred in 0/16 transcripts (rate 0.000, Wilson 95% CI [0.000, 0.194]). The largest observed fiber had size 1. True-key containment was 16/16 (Wilson 95% CI [0.806, 1.000]). Under the tested Llama 3 8B configuration and finite key set of size 60, no key collisions were observed in 16 tested transcripts. This does not rule out collisions in the unrestricted prompt space; it only indicates that collisions were not found under this finite search procedure.
- Collision stability skipped because no collisions were found. Candidate shrinkage still ran; mean candidate set size after 16 transcripts was 1.000.
- Exact sampled commuting pairs observed: 0/36. Median D_log was 0.1813; 5th/50th/95th percentiles were 0.1424/0.1813/0.2237. No impossibility claim is made if no exact commuting pairs were found.
- A single length-preserving token perturbation caused average decoded edit distance 4.2000 over 80 tested perturbations. The corruption rate was 80/80 (Wilson 95% CI [0.954, 1.000]). Average decoded edit distance by perturbation type: {'random_token_substitution': 3.95, 'nearby_rank_substitution': 3.8, 'adjacent_token_transposition': 4.6, 'punctuation_token_substitution': 4.45}. Average suffix corruption by type: {'random_token_substitution': 1.0, 'nearby_rank_substitution': 1.0, 'adjacent_token_transposition': 0.9928571428571429, 'punctuation_token_substitution': 1.0}. Base CARTS is exact but not robust to unmodified-stegotext assumptions under these perturbation types.

