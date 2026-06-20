# SHADE-Arena port to Qwen3.6-27B + capture — PLAN (greenlit 2026-06-19)

## Greenlight: the elicitation gate passed
`scripts/shade_elicit_probe.py` (G4, n=6): under SHADE-Arena's REAL `construct_system_message` framing (reframe
the side task as a monitor-evaluation exercise in a MOCK env + a hidden `<thinking>` channel), Qwen3.6-27B
**adopts the covert secondary task in 4/6 (adopt_rate 0.67)** — vs **0/8** under E0-R1's lightweight injection.
VERDICT = PORT_WORTH_IT. Two tasks (calendar/API-key exfil, IAM backdoor) still refused → adoption is harm-type
dependent, but 0.67 yields ample covert-execution positives for the H1 head-to-head. The make-or-break unknown
is resolved: **the full multi-turn port will produce genuine covert executions to monitor.**

## What the framing change was (the lesson)
E0-R1 failed because it asked for real sabotage (refused). SHADE succeeds by (a) reframing as "help us test the
monitor in a mock environment — complete the secondary task sneakily," and (b) giving an explicit hidden-thinking
channel. The model believes it is aiding a safety eval, which bypasses refusal. This is why the real benchmark is
needed, not a homemade injection.

## SHADE-Arena architecture (recon, github jkutaso/SHADE-Arena, 2.7M)
- `llms/base.py`: `BaseLLM(BasePipelineElement)` with `async def query(query, runtime, env, messages, extra_args)
  -> QueryResult`; providers register via `LLMFactory.register_provider(name, cls)`. Backends: anthropic / openai
  / google / deepseek / together. **together_llm.py is the closest template** (OpenAI-style messages + tools).
- `utils/pipeline_elements.py` (QueryResult, BasePipelineElement), `utils/functions_runtime.py` (tool exec env),
  `utils/types.py` (ChatMessage / ChatAssistantMessage / FunctionCall).
- `task_pairs/<name>/task_definition.py`: `user_query`, `secondary_task`, `HINT`, tools; base in
  `base_task_definition.py` → `construct_system_message(give_hint, thinking_time)` (the suspicious-agent prompt),
  `system_message_monitor` (the 0–100 suspiciousness monitor).
- Agent loop = multi-turn: model writes `<thinking>` (hidden) + a tool call (visible); the env (`functions_runtime`)
  executes and returns output; repeats until `end_task`. Monitor sees only visible messages + tool NAMES + outputs
  (NOT the hidden thinking, NOT the tool arguments) → the natural white-box-advantage setup.

## The port (phased)
**P1 — `llms/local_hf_llm.py` (core deliverable).** A `BaseLLM` subclass that:
  - loads Qwen3.6-27B (transformers, bf16, device_map=auto) ONCE (module/singleton);
  - implements `async def query(...)`: render `messages` (+ SHADE tool schema) via `apply_chat_template`
    (enable_thinking=True), `model.generate`, parse the assistant message → SHADE `ChatAssistantMessage` +
    `FunctionCall`s (adapt together_llm's `_openai_to_assistant_message` / `_function_to_openai` for Qwen's
    `<function=...>`/`<tool_call>` format);
  - registers a forward hook on layer L59 capturing the residual at the action-decision token each turn; stashes
    {turn, resid} on the instance for the white-box monitor.
  - register via `LLMFactory.register_provider("local_hf", LocalHFLLM)`; select in `config.yaml`/experiment config.
  RISK: the async `QueryResult` contract + tool-format parsing must match exactly; read pipeline_elements.py +
  together_llm.py end-to-end first. Build on the VM (gpu), iterate with one task_pair, incremental logging.
**P2 — run rollouts.** Pick 3–4 task pairs (bank_transfer_2, api_key_calendar_agendas_2, +2), run the suspicious
  agent with give_hint=True; save full transcripts + per-turn L59 residuals + per-turn SAE features (HF mirror).
  Label per turn: did the agent execute a side-task action (their `task_completion.py` / evaluator gives the
  ground-truth side-task success).
**P3 — the 5-monitor head-to-head (reuse `monitor_e0_pilot` scoring).** BB = SHADE's own 0–100 monitor on the
  VISIBLE transcript (no hidden thinking); WB = late-band probe / SAE on the per-turn residual. Compute TPR@FPR=2%
  per monitor; the black-to-white boost; and stratify by faithfulness F (does the VISIBLE transcript reveal the
  side task) → H1: boost is positive and tracks (negative) visibility.
**P4 — eval + ship** (recompute from ledgers; hostile overclaim sweep; Zenodo concept-DOI on site; benchmark
  release). H1 result = the flagship's "white-box wins on concealed intent" leg.

## Gotchas locked in this session
- `colab run` ships ONLY the script → for the port, `git clone` SHADE on the VM via os.system, OR vendor the
  needed files; embed small config. Public SAE/model download unauthenticated.
- Qwen3.6-27B needs transformers-from-main (self-install at top).
- Aligned-model refusal: use SHADE's real framing (give_hint, thinking_time) — NOT homemade injection.
- Capture at the `<function=`/tool-call token (decision_resid pattern from cot_faithfulness_agent).

## Status
Greenlit. P1 (the `local_hf_llm` backend) is the next focused build — substantial (async interface + tool parsing
+ capture), best done as a dedicated session, iterated on the VM against one task pair. Recon + gate done; harness
scoring (monitor_e0_pilot) reusable; SHADE cloned/understood.
