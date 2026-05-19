# Tweet drafts — κ_t paper announcement (U-shape framing)

T2 tier (shipped artifact). Pick one.

## Option A — finding-first (recommended)

> Cross-probe coherence κ_t shows a **U-shape** over LLM agent trajectories: drops during exploration, rises during consolidation.
>
> Successful agents have HIGH amplitude (big drop + big rise). Failed agents are FLAT.
>
> Early-half p=0.0002. Late-half p=0.00004. N=99 SWE-bench Pro / Qwen3.6-27B.
>
> openinterp.org/research/papers/kappa-t-coherence-buildup

## Option B — methodology-first (walk-back honesty)

> The original headline ("κ_t shows monotonic buildup") survived 36hr of 5 walked-back probes and 2/3 pre-registered controls — then the third control (trace-length confound) killed it.
>
> The post-control rescue: U-shape decomposition. Stronger p (0.00004 vs 0.0003) and length-normalized by construction.
>
> openinterp.org/research/papers/kappa-t-coherence-buildup

## Option C — biological-inverse angle

> Cardiology: healthy = coupled baseline, failure = decorrelation collapse.
>
> LLM agents: successful = OSCILLATORY (explore→consolidate), failure = FLAT.
>
> Same structural insight (cross-channel coupling carries health info), different dynamical signature.
>
> N=99 SWE-bench Pro / Qwen3.6-27B. p<0.0001.
>
> openinterp.org/research/papers/kappa-t-coherence-buildup

## Option D — short hook

> Explore-then-consolidate, visible in residuals.
>
> LLM agents on SWE-bench Pro: cross-probe coherence κ_t falls during exploration, rises during consolidation. Successful traces have high-amplitude U; failed traces are flat.
>
> openinterp.org/research/papers/kappa-t-coherence-buildup

---

## Posting checklist

- [ ] Verify paper is live at openinterp.org/research/papers/kappa-t-coherence-buildup
- [ ] Verify Zenodo DOI is published (replace placeholder in landing if needed)
- [ ] Consider following up with a thread: methodology arc (walk-back-and-rescue) as a separate post
- [ ] Don't tag Anthropic / Apollo / OpenAI directly (per outreach hygiene rule — not yet)
- [ ] OK to use hashtags sparingly: #mechinterp #LLMagents
