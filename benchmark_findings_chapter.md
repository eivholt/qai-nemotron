# Benchmark Findings: Nemotron Nano on the IQ9 EVK

After deploying `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` on the Qualcomm IQ9 EVK, I wanted to understand what the model is actually good at on-device. The practical question was not “does this reproduce NVIDIA’s leaderboard?”, but:

> Can this model help with Linux commands, HTTP requests, EVK-style diagnostics, coding, reasoning, RAG-style extraction, and tool-call-like structured outputs on the IQ9 EVK?

I benchmarked Nemotron Nano against the stock Qualcomm AI Hub Llama 3.1 8B Instruct W4A16 model. I also ran the non-quantized Hugging Face models on an RTX 5090 host as a reference. The host runs were useful because they separated model behavior from the Qualcomm Genie/QNN W4A16 runtime path.

The benchmark implementation lives in:

- `evk_bench/run_genie_bench.py` for Genie/QNN runs on the EVK.
- `host_bench/run_hf_bench.py` for Hugging Face bf16 runs on the RTX host.
- `host_bench/compare_results.py` for host-side result aggregation.
- `host_bench/run_full_compare.sh` for full host comparison runs.

The benchmark cases themselves are defined in `evk_bench/run_genie_bench.py`, so the same prompts and scorers can be reused from both the EVK and host runners.

## Method

On the EVK, each test writes a prompt file and calls:

```bash
genie-t2t-run \
  -c genie_config.json \
  --prompt_file prompt.txt \
  --profile profile.txt
```

The harness captures:

- the raw model output
- the scoreable answer
- Genie profile metrics
- token generation rate
- time to first token
- pass/fail and partial score
- prompt and log paths

For Nemotron, I tested both:

- `detailed thinking off`
- `detailed thinking on`

I also tested several prompt profiles:

| Prompt profile | Purpose |
|---|---|
| `native` | Minimal task text, closest to the model-card style. |
| `strict_v2` | Concise “return exactly one command” style prompt. |
| `direct_final` | Stronger instruction to return only the final artifact. |
| `best_practical_v2` | Mixed profile: stricter for command tasks, native for reasoning/tool/code tasks. |
| `final_only` / `best_practical_v4` | Very strict “no thinking, no prose” prompt. |

The best overall practical prompt was `best_practical_v2`. It did not make Nemotron perfect, but it was the best compromise between avoiding runaway explanations and preserving enough task context for reasoning/code/tool-like cases.

## Scoring

No LLM judge is used. The benchmark uses deterministic checks against predefined ground truth.

| Test type | Validation method |
|---|---|
| Linux/HTTP commands | Required regex groups plus forbidden regexes. |
| Tool-call JSON | Parse the first JSON object and compare expected tool name/fields. |
| RAG JSON | Parse JSON and compare expected fields exactly. |
| JSON exact keys / arrays | Parse JSON and compare exact keys or exact array values. |
| Math/reasoning | Extract `\boxed{...}` and compare expected number or choice. |
| Instruction following | Regex, count, prefix, and forbidden-word checks. |
| Basic code tests | Regex requirements plus Python code fence. |
| Coding unit tests | Extract Python code and run predefined `assert` tests. |

This makes the results reproducible, but strict. For example, `"Duty Manager"` can fail if the expected value is `"duty manager"`, and a command can fail if it is semantically close but misses a required flag.

## Final-Answer Splitting for `<think>...</think>`

Nemotron can emit reasoning inside `<think>...</think>` tags, especially with `detailed thinking on`. If the benchmark scored the entire raw output, valid final answers could be marked wrong simply because the output also contained reasoning text.

To avoid that, I added final-answer splitting in `evk_bench/run_genie_bench.py`:

1. If the output contains `<think> ... </think>`, the harness stores the text inside those tags as `reasoning`.
2. The text after the final `</think>` becomes `final_answer`.
3. By default, scoring uses `final_answer`, not the full raw output.
4. If a `<think>` tag is opened but never closed, the result is marked as `reasoning_open`, and the scoreable final answer is empty.

The result JSON stores both:

- `raw_answer`: the full generated output
- `reasoning`: extracted thinking text
- `answer`: the scoreable final answer
- `reasoning_open`: whether the model failed to close a thinking block

This mattered a lot. Several reasoning/math cases looked bad before splitting because the correct `\boxed{...}` answer appeared after the reasoning block. Final-answer splitting made those cases score according to the actual final output rather than the scratch work.

However, splitting does not solve every problem. If the model spends the entire token budget inside an unclosed `<think>` block, there is no final answer to score. Increasing the token cap helps expose this failure mode, but it does not always improve accuracy.

## Two Classes of Tests

The results split into two major groups.

### Class 1: Operational Command and Diagnostic Tasks

These were the original practical EVK-oriented tests:

- Ubuntu command generation
- `curl` / HTTP command generation
- QNN / HTP / FastRPC diagnostics
- Genie profile parsing
- Linux service and library troubleshooting
- exact shell snippets

These tests are useful for an EVK assistant, but they do not closely match Nemotron’s advertised benchmark strengths. They reward concise command syntax, exact Linux flags, operational troubleshooting, and exact tool-schema compliance.

Nemotron struggled in this class.

Common failure patterns:

- inventing non-existent commands, such as `ssh log` or `systemdctl`
- using wrong flags, such as malformed `journalctl` or `curl`
- returning explanations instead of commands
- producing near-valid JSON but wrong tool names or fields
- mutating identifiers in code, such as `build_llamme31_prompt`
- using extra token budget to continue explaining instead of improving the final artifact

### Class 2: Nemotron-Friendly Self-Contained Tasks

I added a second class of tests closer to the model’s advertised strengths:

- executable Python coding unit tests
- prompt-contained RAG / fact extraction
- logical reasoning
- arithmetic reasoning
- static toy tool selection
- exact JSON output
- instruction following

These avoid Qualcomm, IQ9, QNN, Genie, and post-training-time system details. All facts needed for the task are inside the prompt.

The strongest contrast came from comparing the old operational suite to this newer self-contained suite.

| Model / mode | Old operational tests | Old avg | New Nemotron-style tests | New avg |
|---|---:|---:|---:|---:|
| EVK Nemotron W4A16, thinking off | 6/20 | 0.527 | 6/8 | 0.900 |
| Host Nemotron bf16, thinking off | 8/20 | 0.629 | 5/8 | 0.794 |
| Host Nemotron bf16, thinking on | 9/20 | 0.591 | 5/8 | 0.700 |
| Host stock Llama 3.1 bf16 | 10/20 | 0.755 | 6/8 | 0.825 |

The EVK result changed dramatically when the task distribution matched Nemotron’s strengths. On the new suite, EVK Nemotron passed:

| New task category | EVK Nemotron W4A16 |
|---|---:|
| RAG / prompt-contained facts | 2/2 |
| Logical reasoning | 2/2 |
| Static tool selection | 1/1 |
| Executable coding unit tests | 1/3 |

This suggests the model is not generally incapable on-device. It is specifically weak on concise operational command generation and exact shell/tool syntax.

## Original Practical Suite: Nemotron vs Stock Llama

On the original 20-case practical suite, stock Llama was stronger.

| EVK model | Mode / config | Pass | Avg score | Decode tok/s | TTFT |
|---|---|---:|---:|---:|---:|
| Nemotron W4A16 | thinking off, 512 tokens | 5/20 | 0.501 | 9.96 | 220.7 ms |
| Nemotron W4A16 | thinking on, 2048 tokens | 7/20 | 0.557 | 9.77 | 219.0 ms |
| Stock Llama 3.1 W4A16 | greedy, 512 tokens | 10/20 | 0.710 | 10.19 | 220.2 ms |

Thinking-on helped Nemotron a little on the old suite, but not enough to catch stock Llama. It also increased the risk that the answer would be consumed by reasoning text or would not reach a clean final artifact.

## Prompt Strictness and Token Budget

I tested whether stronger prompting and larger token caps could fix the issue.

### Token Cap

A larger cap helped reveal behavior, but did not necessarily improve score.

| EVK Nemotron setting | Pass | Avg score | Avg final chars |
|---|---:|---:|---:|
| `best_practical_v2`, 512 tokens | 6/20 | 0.527 | shorter |
| `best_practical_v2`, 2048 tokens | 6/20 | 0.527 | 2075 chars |
| `best_practical_v4`, 2048 tokens | 4/20 | 0.497 | 143 chars |

Interpretation:

- Small caps can cut off the final answer.
- Large caps let the model continue, but often it continues rambling.
- Very strict prompts make output shorter, but can overconstrain the model and hurt correctness.

### Prompt Style

The best practical setting was `best_practical_v2`:

- stricter final-answer prompting for Linux/HTTP style tasks
- native prompting for reasoning, RAG, code, and tool tasks
- final-answer splitting for `<think>...</think>` outputs

The strictest `final_only` profile reduced verbosity, but hurt command accuracy.

## Host bf16 vs EVK W4A16

To estimate quantization/runtime effects, I ran the same prompts on the RTX 5090 host using Hugging Face bf16 weights.

| Test set | Host Nemotron bf16 | EVK Nemotron W4A16 | Difference |
|---|---:|---:|---:|
| Old 20 tests | 8/20, avg 0.629 | 6/20, avg 0.527 | -0.102 avg |
| New 8 tests | 5/8, avg 0.794 | 6/8, avg 0.900 | +0.106 avg |

On the old operational tests, EVK W4A16 dropped relative to host bf16. That suggests quantization/runtime likely hurts some brittle command-generation behavior.

However, on the new self-contained tasks, EVK W4A16 did not show a falloff. It actually scored higher in that run. Quantization alone does not explain the original poor results. Task shape matters more.

## Neutralized Old Suite

To check whether Qualcomm/IQ9-specific knowledge was the real issue, I created a neutralized version of the old suite in `evk_bench/run_genie_bench.py`:

```bash
--suite neutralized
```

This suite keeps the old task shapes but removes Qualcomm, IQ9, QNN, Genie, and FastRPC-specific content. Examples include:

- generic missing shared library
- generic Ubuntu service logs
- generic `usermod -aG` group task
- generic `curl` status-code task
- generic profile parser
- generic prompt builder

EVK Nemotron result:

| Suite | Pass | Avg |
|---|---:|---:|
| Neutralized old-suite | 2/10 | 0.498 |

This shows knowledge cutoff is not the main issue. Both models are based on Meta Llama 3.1 8B Instruct, whose pretraining data cutoff is December 2023. NVIDIA’s Nemotron Nano model card says the model was trained between August 2024 and March 2025, but its data freshness follows Llama 3.1 / 2023 pretraining data. Meta’s Llama 3.1 8B Instruct card lists the pretraining cutoff as December 2023.

If stock Llama were winning only because of a more recent knowledge cutoff, neutralizing the Qualcomm/IQ9 content should have helped Nemotron much more. It did not. The old-suite weakness appears to be mostly task-shape related: exact Linux command syntax, operational diagnostics, tool-call schema compliance, and concise final-output discipline.

## Additional Nemotron-Style Tests

I also added another batch of 10 Nemotron-flavored tests:

- more executable coding tasks
- more prompt-contained RAG tasks
- small logical reasoning tasks
- exact instruction-following tasks
- toy calculator tool selection

The EVK result was:

| Category | Result | Notes |
|---|---:|---|
| Executable coding | 2/3 | Good on slug normalization and window sums; one failed by rambling instead of returning code. |
| RAG / prompt-contained facts | 0/2 | Semantically close, but strict JSON value/type/casing checks failed. |
| Logical reasoning | 1/2 | Weighted average passed; one grid task was ambiguous and should be revised. |
| Instruction following | 1/2 | Exact JSON array passed; line-prefix constraint failed. |
| Tool selection | 0/1 | Correct tool, but parameter was wrapped as `eval(...)`. |

This reinforced the main pattern: Nemotron often understands the task, but exact-output discipline remains the weak point.

## Throughput

Throughput was stable across the EVK models.

| Model / mode | IQ9 EVK W4A16 | RTX 5090 bf16 |
|---|---:|---:|
| Nemotron thinking off | ~9.96 tok/s | ~46.70 tok/s |
| Nemotron thinking on | ~9.77 tok/s | ~41.43 tok/s |
| Stock Llama 3.1 | ~10.19 tok/s | ~45.35 tok/s |

The EVK is around 10 tokens/s for these 8B W4A16 models. The RTX 5090 host is roughly 4-5x faster in bf16.

## Interpretation

The main finding is not “Nemotron is bad” or “quantization broke it.”

My interpretation is:

1. **Nemotron performs well when the task matches its advertised strengths.**  
   On self-contained RAG, reasoning, tool choice, and coding-style tasks, EVK Nemotron looked much stronger.

2. **Nemotron is weak at concise operational shell-command generation.**  
   It often knows the general idea but fails exact syntax, invents commands, or explains instead of outputting the artifact.

3. **Stock Llama is better for Linux/HTTP command generation.**  
   On the original practical EVK-assistant suite, stock Llama was more reliable.

4. **Thinking-on helps some reasoning/code cases but is not a universal win.**  
   It can improve tasks where reasoning is useful, but it can also spend the output budget before reaching the final answer.

5. **Prompting helps but does not fully solve exactness.**  
   Stricter prompts reduce rambling, but too much strictness harms accuracy. The best result came from mixed prompting by task type.

6. **Quantization/runtime effects exist but are task-dependent.**  
   Host bf16 was better on old operational tasks, but EVK W4A16 held up well on the new Nemotron-friendly suite.

## Practical Recommendation

For an EVK tutorial, I would present Nemotron Nano as useful for:

- local reasoning
- prompt-contained data extraction
- simple RAG-style workflows
- structured tool selection
- coding assistance when feedback/testing is available

I would not present it as the best choice for one-shot Linux command generation. For that, stock Llama 3.1 8B W4A16 was more reliable in my tests.

The most promising future demo is not one-shot command generation, but an agent loop where Nemotron can:

1. propose a command or Python snippet,
2. execute it in a sandbox,
3. observe errors,
4. revise,
5. return a final answer.

That would test the model in a setting closer to practical agentic use, while avoiding the harshness of expecting perfect shell syntax in one shot.

## References

- NVIDIA Nemotron Nano model card: https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1
- Meta Llama 3.1 8B Instruct model card: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- NVIDIA hosted Nemotron model card: https://build.nvidia.com/nvidia/llama-3_1-nemotron-nano-8b-v1/modelcard
