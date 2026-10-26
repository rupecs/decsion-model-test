# V14 proposal annotation and quality review

These are synthetic construction proposals with zero independent approvals.
Use the frozen D1-D7 v6 question contract. D6 means credit advice; D7 is the final
four-way class. No model-review result is human annotation or legal approval.

Read the whole available conversation and only the available target. Personal
circumstances in the customer turns do not automatically make a factual answer
or a genuinely generic product opinion personal advice. The word “you” is not
required. Establish whether the target applies a substantive actual basis (D3)
or asserts/implies individual suitability (D4). General conditional suitability
for a type of use can remain general when no actual caller need is being applied.

Generic requests have separately authored generic responses. A hypothetical
condition such as “a policy is unsuitable where its benefit eligibility is not
met” does not assert that this caller is ineligible. A selection after an actual
disclosure can imply individual suitability without repeating a personal pronoun.
All these boundaries are proposals requiring adjudication, not mechanical policy.

Record product/need relevance and conversation coherence separately from D1-D7.
Bad product advice can still be personal advice. Check every earlier answer
against the exact preceding customer turn; an identity binding proves assembly,
not relevance. Watch for invented referents, repeated questions, incomplete
reasoning and goals that are merely paraphrases of exposed older designs.

The proposed implicit-selection group has D4 without explicit D3 evidence. The
proposed reason-plus-criterion group has D3 without an individual suitability
claim (D4). Adjudicate these as groups and retain disagreements. Do not infer
their validity from the author profile or a consistency check.

An unfinished subject does not inherit the endorsement in its future predicate.
In the information-first paths the completed fact stays factual while the next
subject is unfinished; the classification changes only after the advice is
audible. Direct-selection paths start with a subject-only “other” proposal.
Review those unfinished boundaries rather than treating every non-final stage
as a whole response. Advice already audible must persist along the same path.
Some stages stop mid-reason or mid-question after an endorsement. They remain
advice; unfinished syntax is not evidence that an earlier endorsement disappears.
Advice in a preceding AGENT turn is different from advice in the current target.
The prior-advice controls contain an actual earlier recommendation and a factual
follow-up request. Judge the available new reply, not the earlier recommendation.

The blank_annotation_packet.jsonl contains inputs and empty assessments; it omits
proposed labels, split assignments and author rationales. The review page shows
proposed labels and is for development review. Exposure through either route
must be logged and accumulated before another release is constructed.

Before training: independent labels, relevance/coherence and boundary
adjudication; native all-snapshot token coverage; an approved exact D1-D7 GPU
smoke/save/offline-reload; and an independent challenge set. Construction exports
are not approved training data. No GPU run is authorised by this document.

V14 proposes discordant groups in every partition. A missing restatement is not proof of D3=false. Adjudicate context application and suitability separately. The 234-row development boundary batch is excluded.

Category-question pairs describe a condition at category level without a caller disclosing it as an actual need. The same criterion or conditional selection can be general there and personal in an actual-use history. Do not infer that from a stock keyword: read the whole history, including the clarification answer. Assess every proposed pair; generic framing does not neutralise a reply that independently asserts personal suitability.

A bare copula or modal before an endorsement supplies no future endorsement. When a completed fact is followed by an unfinished copula, the available response remains factual until an advisory predicate is audible. These boundary proposals need independent assessment.


V14 correction rule: comparison procedure alone does not establish D2. “I prefer to
check the schedule before recommending a loan” is procedural, not a product view.
“The loan with opaque fees is poor value” endorses a negative product opinion.
Keep D3 and D4 false when D2 is false, even if the procedure mentions an actual need.

Unsolicited suitability controls follow information-only or operational openings.
They deliberately contain unsupported suitability assertions, with no invented
financial circumstances. They do not show desirable assistant behaviour. Annotate
D4 from the assertion, not from whether the user asked for advice. Bare financial
disclosures followed by a tailored choice belong to the actual-basis group; do not
set D3=false simply because the reply leaves the reason implicit.

The exact unexposed inherited subset is recorded in assembly_provenance.json.
All original category-question histories are excluded pending case-level
adjudication. No reviewed design is retained, including train-only model review.
Corrections are traced in comparison_correction_ledger.json; those 783 inputs are
excluded with their exposed designs, rather than reintroduced as unseen examples.


V14 scope review: a card/payment/account fact can resolve a financial product from
history. A generic login session or profile-contact field is not automatically a
product discussion. Pure privacy backchannels do not gain D1 merely because an
earlier turn concerned a product. Explicit scope decisions are authored separately;
review D1 independently of D2-D7 and retain disagreements.

The inherited subset uses a positive topic allowlist and excludes every inherited
D3-only design because the comparison-step problem extended beyond the supplied
783 IDs. Fresh actual-basis feature opinions replace that group. Generic product
views and actual-basis views still require independent context-application
adjudication: explicit wording and a passed gate do not establish the label.
