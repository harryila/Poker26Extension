# Encode-vs-decode control: early-layer vs late-layer oracle decodability

Does the residual decode the Bayesian posterior because the model COMPUTES it (late ≫ early) or because the prompt inputs are trivially present (early ≈ late)?

| model | early L | early trash R² | late L | late trash R² | Δ(late−early) | verdict |
|---|---:|---:|---:|---:|---:|---|
| llama | 2 | +0.635 | 14 | +0.627 | -0.008 | INPUT-PRESENCE — decodable from the start; 'knows' claim NOT supported |
| ministral | 2 | +0.908 | 16 | +0.959 | +0.051 | partial build-up |
| qwen | 2 | +0.339 | 23 | +0.486 | +0.146 | partial build-up |

## Reading
- **COMPUTED** (late−early ≥ ~0.15) supports the strong claim: the model progressively builds the correct posterior; its miscalibrated *stated* belief is a readout failure.
- **INPUT-PRESENCE** (early ≈ late, both high) ⇒ downgrade to the safe claim: 'the information sufficient to compute the posterior is linearly available; the verbalized belief discards it' — do NOT say the model 'knows'.
