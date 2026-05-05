# CARTS Empirical Summary


## Experimental configuration

```json
{
  "allow_failures": false,
  "model_key": "llama3_8b_q4_k_m",
  "n_ctx": 4096,
  "n_gpu_layers": 0,
  "n_threads": null,
  "num_collision_payloads": 6,
  "num_commutativity_pairs": 12,
  "num_commutativity_payloads": 6,
  "num_keys": 10,
  "num_payload_key_pairs": 8,
  "num_payloads": 8,
  "num_robustness_cases": 6,
  "num_stability_payloads": 8,
  "quick_mode": true,
  "random_seed": 123,
  "secret_prefix": "",
  "tail_thresholds": [
    10,
    100,
    1000
  ],
  "use_embeddings": false,
  "use_sklearn_detector": true
}
```

Tokenization convention:

- Non-empty prefixes are tokenized with add_bos=True and the first BOS token is dropped. Empty prefixes use the model BOS token as the minimal autoregressive context.
- Payload and stegotext display strings are tokenized as model.tokenize((' ' + text).encode('utf-8'), add_bos=True)[1:].
- Ranks are 1-indexed. Tokens are sorted by decreasing logit with ties broken by increasing token id.

## Implementation correctness result

Exact recovery succeeded in 8/8 payload-key pairs under Llama 3 8B with the specified deterministic settings. Reverse-direction recovery succeeded in 8/8 cases.

## Rank-likelihood and detection result

CARTS stegotexts had mean normalized NLL 7.5874 compared with 0.8995 for ordinary greedy generation. The mean log-rank distribution shift was 4.2083. A simple detector achieved AUC 1.0000 under this comparison distribution. This is not a proof of steganographic distinguishability in general; it is only for this model, payload distribution, key distribution, and Q_k.

Detector note: sklearn logistic regression detector trained on a small split.

## Key-collision result

Under the finite key set K_adm of size 31, collisions occurred in 0/6 tested transcripts. The largest observed candidate fiber had size 1. The true key was contained in every candidate set: True. These results are finite-key results and do not characterize the infinite prompt space.

## Collision stability result

No discovered collisions under this K_adm, so collision-pair stability was skipped. Candidate shrinkage was still evaluated for the finite key set.

## Non-commutativity result

Exact sampled commuting pairs observed: 0/12. The median metric commutation distance was 0.1899. These results provide evidence about the related-instance hypothesis only in the tested setting.

## Robustness result

A single length-preserving token perturbation caused average decoded edit distance 3.3333 over 18 tested perturbations. Errors typically began at or after the perturbation position when the first mismatch was defined. This supports the claim that base CARTS is exact but not robust without redundancy or error correction under these perturbation types.

## Figure manifest

- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/figures/exp1_correctness_bar.png`: Exact token-id recovery rates for decode and reverse directions.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/figures/exp2_nll_boxplot.png`: Normalized NLL under the key context for CARTS and ordinary greedy comparison text.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/figures/exp2_mean_log_rank_cdf.png`: Empirical CDF of mean log-rank for CARTS and ordinary comparison traces.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/figures/exp2_tail_rank_vs_nll.png`: Tail-rank fraction versus normalized NLL for CARTS and ordinary traces.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/figures/exp2_positionwise_mean_log_rank_heatmap.png`: Position-wise mean log-rank for CARTS and ordinary comparison traces.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/figures/exp2_detector_roc.png`: ROC curve for a simple detector trained on rank and likelihood features.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/figures/exp3_fiber_size_histogram.png`: Histogram of finite candidate fiber sizes.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/figures/exp3_unique_vs_collision_bar.png`: Fraction of unique and colliding finite-key transcripts.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/figures/exp3_search_time_vs_keyset_size.png`: Number of F evaluations required by exhaustive finite key search.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/figures/exp4_candidate_set_size_vs_transcripts.png`: Candidate key set size after accumulating exact transcripts.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/figures/exp5_commutation_rate_histogram.png`: Histogram of sampled exact commutation rates for key pairs.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/figures/exp5_commutation_distance_cdf.png`: Empirical CDF of metric commutation distances.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/figures/exp5_commutation_heatmap.png`: Heatmap of sampled commutation rates among displayed keys.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/figures/exp5_Nk_tau_curve.png`: Average number of key partners below a metric commutation threshold.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/figures/exp6_decoded_edit_distance_by_perturbation.png`: Decoded token edit distance by length-preserving perturbation type.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/figures/exp6_perturb_position_vs_error_heatmap.png`: Mean suffix corruption fraction by perturbation type and position.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/figures/exp6_token_agreement_after_perturbation.png`: Average decoded token agreement after the perturbation position.

## Table manifest

- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/tables/payloads.csv`: Default payload set and token lengths.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/tables/keys.csv`: Default key set and context token lengths.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/tables/experiment1_correctness.csv`: Round-trip status for each payload-key pair.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/tables/experiment2_rank_likelihood_detection.csv`: Rank-likelihood and detector features for CARTS and ordinary greedy comparison text.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/tables/experiment3_key_collision_fibers.csv`: Candidate fiber size and true-key containment for each transcript.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/tables/experiment3_largest_fibers.csv`: Largest observed candidate fibers.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/tables/experiment3_search_cost.csv`: Finite key search cost in F evaluations per transcript.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/tables/experiment4_collision_stability.csv`: Collision stability and divergence across new payload rank vectors.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/tables/experiment4_candidate_shrinkage.csv`: Candidate set size after accumulating transcripts.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/tables/experiment5_noncommutativity.csv`: Sampled commutation rates and distances for key-induced maps.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/tables/experiment5_Nk_tau_curve.csv`: Average number of highly commuting keys by metric threshold.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/tables/experiment6_robustness.csv`: Decoded errors after length-preserving stegotext token perturbations.
- `/home/meow/Documents/repos/LlmStenoExplore/paper-empirical-explorations/results/tables/paper_verification_checklist.csv`: Theory-claim verification checklist for the paper.

## Limitations and caveats

- Results are under this model and tokenizer, not all LLMs.
- Finite key-search results are under this finite key set only.
- The ordinary comparison distribution Q_k is greedy generation and is not a universal natural-text distribution.
- Detector results do not prove steganographic security or insecurity.
- Exact correctness assumes identical tokenization, logits, masking, precision, and tie-breaking.

## Next steps

- Increase sample sizes after confirming exact correctness.
- Add stronger comparison distributions for Q_k.
- Add calibrated plausibility metrics before wrong-key equivocation experiments.
- Test another quantization or exact model variant only as a separately labeled run.

## Paper-ready paragraphs

- Exact recovery succeeded in 8/8 payload-key pairs under Llama 3 8B with the specified deterministic settings. Reverse-direction recovery succeeded in 8/8 cases.
- CARTS stegotexts had mean normalized NLL 7.5874 compared with 0.8995 for ordinary greedy generation. The mean log-rank distribution shift was 4.2083. A simple detector achieved AUC 1.0000 under this comparison distribution. This is not a proof of steganographic distinguishability in general; it is only for this model, payload distribution, key distribution, and Q_k.
- Under the finite key set K_adm of size 31, collisions occurred in 0/6 tested transcripts. The largest observed candidate fiber had size 1. The true key was contained in every candidate set: True. These results are finite-key results and do not characterize the infinite prompt space.
- No discovered collisions under this K_adm, so collision-pair stability was skipped. Candidate shrinkage was still evaluated for the finite key set.
- Exact sampled commuting pairs observed: 0/12. The median metric commutation distance was 0.1899. These results provide evidence about the related-instance hypothesis only in the tested setting.
- A single length-preserving token perturbation caused average decoded edit distance 3.3333 over 18 tested perturbations. Errors typically began at or after the perturbation position when the first mismatch was defined. This supports the claim that base CARTS is exact but not robust without redundancy or error correction under these perturbation types.
