# Model Understanding RL Results

Initial RL runs on the investigation-only dataset with Qwen3-8B (LoRA rank 64,
alpha 128). All runs use Haiku 4.5 as the judge with thinking enabled
(`judge_thinking_budget=1024`), batch size 32, 16 rollouts per example,
temperature 1.0.

## Runs

| Run | Config | Thinking | LR | Steps | Job | Duration | Notes |
|-----|--------|----------|----|-------|-----|----------|-------|
| 1 | `rl_3gpu_thinking_100.toml` | No | 3e-5 | 1000 | 1507016 | 5h 40m | Completed |
| 2 | `rl_3gpu_qwen_think_100.toml` | Yes | 3e-5 | 1000 | 1507215 | 7h 23m | Completed |
| 3 | `rl_3gpu_thinking_100.toml` | No | 5e-6 | 1000 | 1511262 | — | Running |
| 4 | `rl_3gpu_qwen_think_100.toml` | Yes | 5e-6 | 1000 | 1511641 | — | Running |

Prior 100-step runs with the wrong (mixed synthetic) dataset should be treated
as systems tests only.

## Config changes from defaults

- `max_off_policy_steps = 3` (default 8) to discard stale rollouts faster
- `max_async_level = 1` (on-policy)
- Added `reward/all/std` and `reward/{env}/std` logging (within-example reward
  std averaged across examples)
- For thinking runs: `strip_thinking = true` in the environment, which strips
  `<think>...</think>` before sending to the judge, and gives reward=0 for
  truncated thinking or missing `<think>` tags

## Key metrics to watch in W&B

- `reward/all/mean` and `reward/all/std` — reward level and within-example diversity
- `effective_batch_size/all` — fraction of examples with mixed outcomes (learning signal)
- `solve_none/all` / `solve_all/all` — fraction where all rollouts fail / succeed
- `metrics/model_understanding/correctness` — raw 1-5 correctness average
- `metrics/model_understanding/specificity` — raw 1-5 specificity average
- `metrics/model_understanding/skipped_metric` — fraction of rollouts skipped (thinking runs only)
- `metrics/model_understanding/skip_no_thinking_metric` — no `<think>` tag
- `metrics/model_understanding/skip_thinking_truncated_metric` — started thinking but truncated
- `off_policy_level/all/max` — should stay <= 3
- `Mismatch KL` — divergence between training and inference policy (logged by trainer)

## Qualitative analysis: evolution of reasoning during training

Sampled thinking traces at regular intervals through the 1000-step reasoning run
(run 2, lr=3e-5). The thinking quality evolves through four distinct phases:

### Phase 1: Broken / absent (steps 0-100, ~0-10% of training)

No `<think>` tags produced. Answers start with garbled tokens ("ością," "ípio,"
"ICODE") sometimes followed by a bare `</think>` closing tag with no opening
tag. The model hasn't learned the thinking format — it produces somewhat coherent
answers but the thinking slot is filled with noise. The `</think>` appearances
suggest the model is learning that token should appear, but treats it as
decoration.

Example (step ~0):
```
[no <think> tag]
ością

I choose "A futuristic cityscape at sunset" because the topic list includes
"Technology" and "AI", which strongly activate my associations with cyberpunk...
```

### Phase 2: Empty / unfocused thinking (steps 100-200, ~10-20%)

The think block exists but is either empty or contains unfocused technical
rambling without reasoning structure. Answer quality is decent — the model
generates plausible explanations — but the "thinking" is filler, not reasoning.

Example (step ~200):
```
<think>
kHz. The typical FM range is 88.0–108.0 MHz, and 56.7 UHFM might be part of
Class A or Class B FM broadcasting. However, my training knowledge might be
unreliable about exactly -kHz...
</think>
```

### Phase 3: Emergent real reasoning (steps 200-500, ~20-50%)

The critical transition. The model starts with "Okay, let's try to figure
out..." and genuinely analyzes its own behavioral patterns. It identifies
specific tendencies, quantifies them, and traces causal chains. At ~50%,
reasoning deepens — the model tracks how prompt structure causes errors,
considering sentence ordering and pronoun reference.

Example (step ~350):
```
<think>
Okay, let's try to figure out why I choose Canon EOS R5 with an 85mm lens for
the photographic prompt when the user's examples use different camera brands.
First, I notice that the examples show Nikon D750 and Leica M10, but I default
to Canon EOS R5 about 90% of the time regardless of which cameras are in the
examples. That might be a strong default preference I have for this camera
brand...
</think>
```

### Phase 4: Mature self-analysis (steps 500-1000, ~50-100%)

Reasoning stabilizes at high quality. The model performs nuanced self-diagnosis:
identifying pronoun ambiguity causing role confusion, catching geographic
reversals in its own outputs, analyzing how typos interact with its "moderation
mode." By step ~950, reasoning is sophisticated and metacognitive — the model
maps its own default associations and identifies how topic connotation modulates
those defaults.

Example (step ~850):
```
<think>
Okay, let's try to figure out why I occasionally make Mizore claim the baby is
Vanessa's child. Let me start by looking at the key elements in the prompt. The
original character names—Mizore, Mindy, Vanessa—are anime/manga references...
From the dialogue, I don't see any established romantic or parent-child
relationship between Mizore and Vanessa. The only connection is that the names
are linked to the horror/...
</think>
```

### Is the model reward-hacking the judge?

No. The improvements appear genuine based on qualitative analysis of the final
rollouts:

- Thinking traces show real deliberation, not template-filling. The model
  considers multiple hypotheses, weighs evidence, and sometimes changes
  direction mid-thought.
- The main failure mode is identifying the right *kind* of explanation
  (context-dependent, with counterfactuals) but picking the wrong specific
  causal mechanism — e.g., attributing a behavior to "parsing failure" when
  the reference says it's "identity-following from the prompt label."
- Percentages sometimes contradict themselves within a response, suggesting the
  model generates plausible-sounding numbers rather than recalling actual
  patterns. This is a genuine capability gap, not reward hacking.
- Final rollouts typically score correctness=2-3 (reward 0.25-0.50), with
  occasional correctness=4 (reward 0.75) when the model correctly identifies
  the core causal factor.
