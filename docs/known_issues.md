# Known limitations carried into the handover

## Data and labelling

- **A01 grounding:** a small number of D3 labels in shortened context variants rely on a financial detail that is not explicit in that particular visible history. Existing context-support spans do not prove semantic grounding. Keep the frozen data unchanged; independently adjudicate and version any corrections.
- Unsolicited suitability controls are frequent and can be less natural than real banking conversations. They help separate D3 from D4, but are not a claim of representative real-call prevalence.
- Stock privacy/backchannel language and residual lexical cues remain. Construction checks passing does not make the corpus pattern-free.
- All labels remain synthetic proposals. Review by another model is not independent human annotation, legal approval or real-customer validation.
- Full v14 publication makes its held-out split public; future independent claims require separately authored challenge material.

## Model behaviour

The completed synthetic test D7 accuracy was 98.8788%, but all-seven exact match was 89.7491%, and D1 macro-F1 was only 0.789879. Report each decision, not only the final class.

During live inference testing, a simple interest definition produced D5=true but D7=other. The native results were preserved and the UI exposed the inconsistency; no correction layer was added. An implicit-context product choice without “you” returned personal advice, but one successful adversarial example does not establish robustness.

Validation loss reached its minimum at step 300 and then worsened over the next three checks; early stopping at 600 restored step 300. Do not train blindly for the full step cap or interpret very low training loss as generalisation.

## Portability and execution

- The reference stack/GPU was actually tested on A40; the new SageMaker packaging/gate is CPU-tested, not a completed AWS deployment or Docker build claim.
- Actual hardware, dependencies, native coverage, gradients/updates, calibration, FP32 saving and all-seven fresh-process reload must be tested on the chosen AWS GPU.
- Exact head saving calls a pinned private Unsloth API. Version/signature drift is a hard failure.
- Single visible GPU only. No distributed training, checkpoint resume or safe spot-interruption recovery is implemented.
- Historical release metadata/README are preserved byte-for-byte. They are not a fresh review against all later exposure or a statement of current annotation readiness.
