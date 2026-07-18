# Agentic LLMs on Dragonwing IQ9075

This tutorial compares small language models as local planning and tool-selection
engines on the Qualcomm Dragonwing IQ9075. It is a companion to
[Deploy Nemotron Nano on Dragonwing IQ9075](https://dragonwingdocs.qualcomm.com/tutorials/deploy-nemotron-nano-on-dragonwing-iq9075),
which explains the complete workstation setup, W4A16 quantization, QAI Hub
compilation, transfer, and Genie validation process.
I do not repeat those setup steps here. The focus is what happens after a model
can answer a prompt: can it reliably choose tools, supply valid arguments, use
results over several turns, and stop without taking unnecessary actions?

The central finding is that an agentic model is not just a checkpoint. The unit
under test is the checkpoint, quantization, chat template, tool-schema renderer,
output parser, inference runtime, and client loop. A smaller model behind its
native tool protocol can outperform a larger model behind a plausible but
incorrect protocol.

## What I tested

I used two complementary benchmark families. They intentionally test tool use,
not factual knowledge.

### BFCL V4 function calling

The [Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
(BFCL) evaluates whether a model
selects the correct function and supplies the correct arguments from one or
more available schemas. BFCL includes simple calls, calls to one of several
functions, multiple and parallel calls, live-schema cases, relevance, and
irrelevance. Irrelevance matters because a useful agent must decline to call a
tool when none can satisfy the request.

I run the official BFCL V4 deterministic scorer through
`agent_arena/bfcl_v4_subset_runner.py`. These are partial local evaluations,
not official leaderboard submissions. Web-search agent tests are excluded and
no network requests are executed.

- BFCL80: `agent_arena/benchmark_selections/bfcl_v4_holdout80_nonweb_20260627.json`
- BFCL90: `agent_arena/benchmark_selections/bfcl_v4_fresh90_nonweb_20260628.json`

The two non-overlapping selections reduce the risk of tuning an adapter to a
familiar list.

```mermaid
pie showData
    title Fixed non-web BFCL comparison (170 cases)
    "Holdout BFCL80" : 80
    "Fresh BFCL90" : 90
```

### Iterative hospital logistics

The second arena is a realistic local-agent workload. A hospital logistics
coordinator assigns porters or robots, checks cold-chain limits and elevator
state, escalates conflicts, and updates jobs. This represents realistic edge agentic AI tasks.

`agent_arena/pydantic_hospital_logistics_arena.py` uses a real Pydantic AI loop
against deterministic mock tools from `agent_arena/hospital_logistics_runtime.py`.
The model chooses the next action, the tool executes, and the result is returned
in the next model turn. It is not required to emit an entire plan in one answer.

The test set has nine short tasks (`O1`-`O5` and `P1`-`P4`), where the model
chooses between a few actions, and five longer tasks (`L0`-`L4`) that require
several tool calls. Fixed rules check each run automatically; no LLM grades the
answers. Missing or wrong calls, repeated or unnecessary actions, forbidden
tools, calls that never execute, and steps in the wrong order all count against
the model. System failures are rerun and do not count as model mistakes.

## Models and execution paths

| Model | Size | Best path tested on IQ9075 | Desktop reference |
|---|---:|---|---|
| [NVIDIA Llama-3.1-Nemotron-Nano-8B-v1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1) | 8B | Custom W4A16 Genie, HTP/NPU | BF16, RTX 5090 |
| [Meta Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 8B | Qualcomm W4A16 Genie, HTP/NPU | BF16, RTX 5090 |
| [Mistral Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16) | 3.3B | Custom Q4 Genie/QNN, HTP/NPU | BF16, RTX 5090 |
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 4B | QAI Hub Models W4A16 Genie, HTP/NPU | BF16, RTX 5090 |
| [Team-ACE ToolACE-2.5-Llama-3.1-8B](https://huggingface.co/Team-ACE/ToolACE-2.5-Llama-3.1-8B) | 8B | Custom W4A16 Genie, HTP/NPU | BF16, RTX 5090 |
| [DeepReinforce Ornith-1.0-9B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B-GGUF) | 9B | Q4_K_M GGUF, eight-core CPU | BF16, RTX 5090 |
| [Mistral Ministral-3-8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16) | 8B | Q3_K_M-to-HTP, QAIRT 2.47, HTP/NPU; Q4 load failed | BF16, RTX 5090 |
| *[Salesforce Llama-xLAM-2-8b-fc-r](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r)* | 8B | Screened on Desktop; not exported | BF16, RTX 5090 |
| *[MadeAgents Hammer2.1-7b](https://huggingface.co/MadeAgents/Hammer2.1-7b)* | 7B | Screened on Desktop; not exported | BF16, RTX 5090 |
| *[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)* | 7B | Public binary targets incompatible v79 DSP | BF16, RTX 5090 |
| *[Meta Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) and [Qwen2 7B Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)* | 3B/7B | Diagnostic Desktop runs only | BF16, RTX 5090 |

*Italic model names were run only on the Desktop; they did not complete an IQ9075 inference run.*

"Screened" does not mean a model is unusable. It means its Desktop result did not
justify another costly IQ9075 export in this project, or its license/runtime fit
was less attractive than the selected candidates.

```mermaid
flowchart LR
    A[Source checkpoint] --> B{Supported export path?}
    B -->|QAI Hub or custom QAIRT| C[QNN context binaries]
    C --> D[Genie with QnnHtp]
    D --> E[Hexagon HTP/NPU]
    B -->|No current lowering| F[GGUF]
    F --> G[llama.cpp]
    G --> H[EVK CPU fallback]
    A --> I[Desktop BF16 reference]
    I --> J[RTX 5090]
```

## BFCL results

The table uses the same 170 cases for every row. `Combined` is a convenience
total, not an official BFCL leaderboard metric. Each model uses its best honest
native adapter; the semantic task and official scorer remain unchanged.

| Model and runtime | BFCL80 | BFCL90 | Combined |
|---|---:|---:|---:|
| Ornith 9B BF16, RTX 5090 | 71/80 | 78/90 | **149/170 (87.6%)** |
| ToolACE 2.5 BF16, RTX 5090, Llama JSON | 74/80 | 72/90 | **146/170 (85.9%)** |
| Ornith 9B Q4_K_M, IQ9075 CPU | 69/80 | 76/90 | **145/170 (85.3%)** |
| Qwen3 4B BF16, RTX 5090 | 70/80 | 70/90 | **140/170 (82.4%)** |
| Ministral 8B Instruct BF16, RTX 5090 | 69/80 | 69/90 | **138/170 (81.2%)** |
| Ministral 3B Q4, IQ9075 HTP | 66/80 | 66/90 | **132/170 (77.6%)** |
| Ministral 3B BF16, RTX 5090 | 66/80 | 64/90 | **130/170 (76.5%)** |
| Ministral 8B Q3, IQ9075 HTP | 67/80 | 61/90 | **128/170 (75.3%)** |
| Nemotron Nano BF16, RTX 5090 | 59/80 | 61/90 | **120/170 (70.6%)** |
| Qwen3 4B W4A16 deterministic, IQ9075 HTP | 58/80 | 58/90 | **116/170 (68.2%)** |
| Stock Llama 3.1 W4A16, IQ9075 HTP | 55/80 | 53/90 | **108/170 (63.5%)** |
| ToolACE 2.5 W4A16 Pythonic, IQ9075 HTP | 56/80 | 52/90 | **108/170 (63.5%)** |
| Nemotron W4A16 thinking off, IQ9075 HTP | 53/80 | 45/90 | **98/170 (57.6%)** |
| Nemotron W4A16 thinking on, IQ9075 HTP | 47/80 | 41/90 | **88/170 (51.8%)** |

[Mermaid's horizontal XY orientation](https://mermaid.js.org/syntax/xyChart.html)
places model names on the vertical axis, leaving room for descriptive labels. I
still separate Desktop references from IQ9075 results so execution paths remain
visually distinct.

```mermaid
xychart horizontal
    title "Desktop BFCL references"
    x-axis ["Ornith 9B BF16", "ToolACE 2.5 8B BF16", "Qwen3 4B BF16", "Ministral 8B BF16", "Ministral 3B BF16", "Nemotron Nano 8B BF16"]
    y-axis "Correct of 170" 0 --> 170
    bar [149, 146, 140, 138, 130, 120]
```

```mermaid
xychart horizontal
    title "IQ9075 BFCL results"
    x-axis ["Ornith 9B CPU", "Ministral 3B Q4 HTP", "Ministral 8B Q3 HTP", "Qwen3 4B W4A16 HTP", "Llama 3.1 8B W4A16 HTP", "ToolACE 2.5 8B W4A16 HTP", "Nemotron W4A16 off", "Nemotron W4A16 on"]
    y-axis "Correct of 170" 0 --> 170
    bar [145, 132, 128, 116, 108, 108, 98, 88]
```

Runtime failures are never treated as empty model answers. The Ministral 8B Q3
row includes five requests that reached the 300-second hard timeout. All five
are recorded as non-passing calls, even in BFCL irrelevance categories where a
genuine decision not to call a function can be correct.

The result that changed my model-selection assumptions was Ministral 3B. It is
smaller than the 8B models, yet its native Mistral tool protocol is disciplined
and stable across Desktop BF16 and Q4 HTP deployment. Parameter count alone is a
poor predictor of agentic reliability.

ToolACE exposes two legitimate protocols on Desktop. Its bundled Llama JSON path is
best for this BFCL slice at 146/170. Its model-card Pythonic path scores 140/170
on BFCL but leads the hospital arena at 10/14. The custom W4A16 IQ9075 export
scores 108/170 with the Pythonic adapter and only 49/80 in the device Llama JSON
probe. Protocol choice therefore remains workload- and deployment-specific;
the device result is not directly interchangeable with the best Desktop row.

Nemotron improved substantially after adopting NVIDIA/BFCL-style schemas,
native placement, final-answer splitting, and conservative parsing. The fresh
90-case set prevented those fixes from becoming test-specific padding. Thinking
off remained better for strict function selection; reasoning on spent more
tokens without improving executable calls. This does not contradict Nemotron's
strength in math, coding, or scientific reasoning.

## Hospital results

The hard slice is stricter than conversational evaluation: a correct action plus
an unnecessary action is still a failure.

`Strict pass` requires every expected call and argument, with no missing,
forbidden, duplicate, excess, unexecuted, or out-of-order action. `Average` is
the mean 0-to-1 ledger score across the 14 cases. It gives partial credit for
satisfied requirements, then subtracts the same strict failure penalties. It is
useful for diagnosing near misses, but strict pass is the operational result.

| Model and runtime | Strict pass | Average |
|---|---:|---:|
| ToolACE 2.5 BF16, native Pythonic | 10/14 | 0.789 |
| Qwen3 4B BF16 | 9/14 | 0.779 |
| *xLAM 2 8B BF16* | 9/14 | 0.767 |
| Ministral 3B BF16 | 9/14 | 0.643 |
| Ornith 9B Q4_K_M, IQ9075 CPU | 9/14 | 0.796 |
| Ministral 3B Q4, IQ9075 HTP | 9/14 | 0.643 |
| Qwen3 4B W4A16 deterministic, IQ9075 HTP | 9/14 | 0.643 |
| Ministral 8B BF16, RTX 5090 | 8/14 | 0.655 |
| Llama 3.1 8B BF16 | 8/14 | 0.601 |
| Ministral 8B Q3, IQ9075 HTP | 8/14 | 0.571 |
| Nemotron Nano BF16 | 6/14 | 0.429 |
| ToolACE 2.5 W4A16, IQ9075 HTP | 6/14 | 0.429 |
| Nemotron W4A16 thinking off, IQ9075 HTP | 5/14 | 0.357 |
| Stock Llama W4A16, IQ9075 HTP | 4/14 | 0.286 |
| Nemotron W4A16 thinking on, IQ9075 HTP | 4/14 | 0.286 |

Ministral 8B Q3 matches its Desktop BF16 reference at 8/14 strict passes, although
its average falls from 0.655 to 0.571. It passes eight of nine bounded decisions
but none of the five long workflows. The device run takes 1h 02m 50s and includes
two controlled 300-second generation timeouts in L2; neither is a QNN failure.

One shared-suite caveat is that the long workflows also expose simplified
`*_pending_*` convenience tools intended for bounded cases. Q3 selected those
wrappers in L0 and L4. I retain the strict failures for comparability and do not
alias the calls. A future revision should hide the convenience tools and rerun
every model rather than repair one model's row.

No local model reliably completed all five long workflows. The practical design
response is not to hide failures with permissive parsing. Keep each decision
bounded, return observations between actions, expose only relevant tools, and
enforce policy and safety outside the model.

## Why templates and parsers changed the outcome

Each model was trained to emit a particular wire format:

- Ministral uses Mistral's native tool tokens and parser.
- Qwen3 uses `<tools>`, `<tool_call>`, and `<tool_response>` blocks.
- ToolACE supports both its bundled Llama JSON path and model-card Python calls
  such as `[reserve_elevator(elevator_id="E2")]`.
- Stock Llama uses the bridge's Llama 3 JSON renderer and parser.
- Nemotron uses NVIDIA/BFCL-style function schemas and may separate
  `<think>...</think>` reasoning from the final executable section.

Adapter names such as `mistral_tool` and `llama3_json` refer to project code in
`agent_arena/openai_genie_server.py`; they are not built-in Genie tool parsers.
Genie runs the rendered prompt and returns model text. The bridge is responsible
for turning that text into OpenAI-compatible tool calls.

Final-answer splitting means the server preserves reasoning as reasoning, but
parses tool calls only from the section after the closed `<think>` block. Without
that separation, chain-of-thought can be mistaken for a final answer, hide a
valid call, or consume the output budget before an executable answer appears.
This does not invent or delete tool actions: every parsed call still reaches the
agent and strict ledger.

The OpenAI/Pydantic client format and the model's internal text format are not
the same thing. The adapter translates OpenAI tool schemas into the model-native
prompt and translates native output back into OpenAI `tool_calls`. MCP discovery
can supply the same schemas, but MCP does not remove the need for a correct
model-specific renderer and parser.

## Ministral 8B: when export success is not deployment success

Ministral-3-8B-Instruct-2512 produced one of the most instructive deployment
failures in this project. Its Desktop BF16 score made it a promising upgrade from
Ministral 3B, and QAIRT 2.47 can ingest the publisher's GGUF architecture. The
first Q4_K_M build completed successfully, but a completed export did not mean
the package could be loaded on IQ9075.

### Q4 compiled, but the full model would not map

The [official Q4_K_M GGUF](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF)
is about 4.9 GiB. QAIRT's automatic splitter selected
17 contexts, while this Genie path accepts at most nine. Forcing nine produced
a valid-looking 6.5 GiB Genie export in 1h 26m 34s. The first physical run then
exposed two separate problems:

1. The QAIRT compiler on the Desktop was version 2.47, but `/opt/qairt/current` on the EVK pointed
   to QAIRT 2.45 and Genie 1.17. That runtime rejected context zero with QNN
   `err 5000`.
2. A side-by-side QAIRT 2.47 runtime with Genie 1.18 accepted the binaries, but
   failed while loading context eight. Loading the largest files first moved
   the failure to the ninth and final context, proving the files themselves
   were usable while the complete mapped working set was not.

The verbose QNN log made the second failure concrete. FastRPC failed to map a
641,728,512-byte shared-weight buffer into every available process domain and
returned `err 1002`, which QAIRT defines as `QNN_COMMON_ERROR_MEM_ALLOC`. The
board still had about 33 GiB of normal RAM available. This was therefore an
HTP/FastRPC/SMMU mapping limit, not ordinary Linux memory exhaustion.

`qnn-context-binary-utility` also reported a maximum spill/fill requirement of
about 120 MB, while the generated Genie configuration contained
`spill-fill-bufsize: 0`. Setting a conservative 128,000,000-byte shared buffer
was correct according to the context metadata and Qualcomm documentation, but
did not resolve the total mapping limit. It should still be fixed rather than
left at the generated zero value.

### A Q3 package fits, with a performance tradeoff

QAIRT 2.47 documents HTP quantizer support for Q3_K GGUF tensors. I therefore
used the same 8B checkpoint in [Q3_K_M form](https://huggingface.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF).
This is a more aggressive
quantization, not a smaller parameter-count model. The 4.0 GiB source produced
a 6.1 GiB, nine-context HTP package. It loaded all contexts with the side-by-side
QAIRT 2.47 runtime and answered a native Mistral prompt correctly:

```text
<s>[INST]Reply with exactly OK.[/INST]
```

The response was `OK.` and the profile confirmed `QnnHtp`, but decode speed was
only 1.91 tokens/s, with 15.3 prompt tokens/s and 4.39 seconds of dialog
initialization per CLI request. This is an NPU compatibility success, not yet an
efficient production deployment.

The first BFCL attempt looked like a QNN stability failure, but the fault was in
the experimental HTTP bridge. When `genie-t2t-run` exceeded the 90-second limit,
Python returned captured output as bytes. The bridge tried to write those bytes
as text, raised another exception, and dropped the HTTP connection instead of
returning a timeout response. Automatic client retries then launched more long
requests. The resulting load also made SSH appear unavailable even though the
model process had not crashed.

Normalizing timeout output to text fixed the disconnect and exposed a second
benchmarking trap. BFCL irrelevance cases reward a model for making no function
call, so a transport timeout represented as an empty answer could accidentally
receive credit. The strict runner now converts infrastructure failures into a
reserved invalid tool call. It cannot match any expected function and therefore
always scores as a failure.

With natural EOS, a 300-second hard timeout, and strict failure accounting, Q3
completed BFCL80 at 67/80 and BFCL90 at 61/90: 128/170, or 75.3%. Three BFCL80
requests and two BFCL90 requests timed out; all five were counted as failures.
The suites took 1h 13m 44s and 1h 15m 04s respectively. The result is only four
calls behind Ministral 3B Q4 on HTP, but it is much slower and remains below the
Desktop 8B BF16 reference at 138/170.

A raw `Say OK.` prompt without Mistral's native chat wrapper ran for more than
11 minutes and did not terminate. The exact native `[INST]...[/INST]` request
returned in 6.6 seconds. This is a useful reminder that an NPU performance test
must validate the model's template and EOS behavior before blaming hardware.

### Keep compiler and runtime versions together

I installed QAIRT 2.47 alongside the existing device runtime at
`/home/ubuntu/qairt-2.47.0.260601` and selected it per model through `PATH`,
`LD_LIBRARY_PATH`, and `ADSP_LIBRARY_PATH`. I did not repoint
`/opt/qairt/current`, because doing so would silently change the runtime used by
every already-validated model. A system-wide update is possible, but should be
followed by regression tests for all existing bundles.

For Qualcomm tooling, this case suggests useful improvements: keep the generic
GGUF splitter within Genie's supported context count, derive spill/fill settings
from the produced binaries, report FastRPC mapping capacity before transfer,
and make compiler/runtime compatibility explicit. These diagnostics would turn
an opaque `Failed to create the dialog` into an actionable deployment result.

## Time, memory, and disk: what to expect

These figures are deliberately labeled. **Measured** values came from logs,
Genie profiles, process monitoring, or GNU `time`. **Approximate** values are
rounded observations. **Unrecorded** means I did not reconstruct a number after
the fact.

The workstation used an RTX 5090 with 32 GB VRAM and 192 GB system RAM. WSL2 was
configured for 176 GB RAM and 96 GB swap. GPU inference generally fit in VRAM;
the large system-RAM figures below come from quantization and graph export, not
ordinary model serving.

| Operation | Wall time | Peak Desktop memory | Disk/artifact notes |
|---|---:|---:|---|
| Nemotron W4A16 quantization, 4K context | 44m 59s measured | 174 GiB RSS measured | 32.1 GB `model.data`; final bundle about 5 GB |
| ToolACE W4A16 quantization | 3h 31m measured | 183.4 GB RSS measured | 32.1 GB `model.data` plus ONNX graphs |
| ToolACE local simulator validation | 27m 45s measured | about 97 GB RSS observed | large temporary serialization I/O |
| ToolACE QAI Hub compile/link/export | 1h 20m measured | not representative locally | five linked binaries, about 5.1 GB total |
| Ministral 3B Q4 custom HTP build | about 25m measured | unrecorded | source about 2 GB; container/export about 3.3 GB each |
| Ministral 8B Q4 generic GGUF-to-HTP build | 1h 26m 34s measured | 68.4 GB RSS measured | source 4.9 GiB; cache 66 GB; export 6.5 GiB; final HTP mapping failed |
| Ministral 8B Q3 generic GGUF-to-HTP build | 1h 14m 42s measured | 84.8 GB RSS measured | source 4.0 GiB; cache 69 GB; export 6.1 GiB; HTP load succeeded |
| Ministral 8B Q3 BFCL80 on IQ9075 | 1h 13m 44s measured | about 0.38 GB Desktop client RSS | 67/80; three strict 300-second timeouts |
| Ministral 8B Q3 BFCL90 on IQ9075 | 1h 15m 04s measured | about 0.38 GB Desktop client RSS | 61/90; two strict 300-second timeouts |
| Ministral 8B Q3 hospital14 on IQ9075 | 1h 02m 50s measured | about 0.12 GB Desktop client RSS | 8/14; two controlled timeouts in L2 |
| Qwen3 4B QAI Hub export | unrecorded | unrecorded | downloaded W4A16 checkpoint cache about 17 GB |
| Ornith 9B CPU deployment | no NPU export | about 18 GB EVK RSS measured | official Q4_K_M file 5.63 GB |

For a fresh custom 8B W4A16 export, plan for at least 192 GB system RAM and
roughly 200 GB free disk. Failed checkpoints, shared caches, compiler temporary
files, and final bundles coexist. Place large temporary directories on a drive
with ample space; WSL's default virtual disk filled during one ToolACE attempt.

Benchmark duration depends more on serving architecture and output discipline
than on raw decode speed. On the ToolACE CLI-backed NPU path, BFCL80 took about
8 minutes and BFCL90 took about 16 minutes. One repetitive generation consumed
seven of those minutes before the 4096-token context stopped it. The 32-case
hospital arena took 17 minutes and issued many iterative requests. A persistent
Genie service should be faster because the experimental Python bridge launches
`genie-t2t-run` and initializes a dialog for every completion.

Representative device decode rates were about 10.0 tokens/s for Nemotron W4A16,
9.9 tokens/s for ToolACE W4A16, and 18.3 tokens/s for the direct Qwen3 W4A16
smoke test. Ministral 8B Q3 reached only 1.91 tokens/s on its generic HTP export.
Ornith Q4_K_M reached about 7.3 generated tokens/s with eight EVK CPU threads.
The two Ministral 8B Q3 BFCL suites each took about 75 minutes, and its 14-case
hospital run took another 63 minutes. Tokens per second do not predict agent
completion time when a model loops, reasons for 1,000 tokens, or needs many tool
turns.

## Hosting lessons

The experimental OpenAI bridge is valuable because every prompt, raw response,
parse decision, and profile is inspectable. It is not the ideal production
serving path:
it starts one `genie-t2t-run` process per request, cannot enforce the OpenAI
client's output-token cap through a Genie CLI option, and repeats model/dialog
initialization. One ToolACE BFCL response entered a repetitive list, ran until
`Context Size was exceeded`, and added seven minutes while still correctly
counting as a model failure.

Qualcomm's persistent C++ GenieAPIService removes per-request process startup.
In this project it required model-template and parser work before it could
preserve every native tool format, so the final cross-model rows use the common
inspectable bridge. A production implementation should combine persistent Genie
sessions with the proven native renderers and parsers.

Always separate infrastructure from model behavior. `Failed to create device:
14001`, context-binary incompatibility, connection errors, and board outages are
rerun. A model that repeatedly calls tools until its valid context is exhausted
is a failed agent trajectory and remains a non-pass.

## Practical recommendations

For accelerator-backed bounded tool selection today, Ministral 3B is the most
stable tested IQ9075 model. Ministral 8B Q3 achieves a similar BFCL score, but
its 1.91-token/s decode rate and long-tail timeouts make it a compatibility
demonstration rather than a practical upgrade. Qwen3 4B is a strong modern QAI
Hub option when its native template and actual Genie sampler are used. Stock
Llama is useful with its Llama 3 adapter. Nemotron remains interesting for
reasoning and demonstrates how much correct serving interpretation matters, but
reasoning on is not the best default for short function selection.

Ornith is the strongest overall checkpoint tested and retains most of its Desktop
score after Q4 quantization, but current architecture support leaves it on the
EVK CPU. That makes it an informative Qualcomm enablement target rather than the
preferred low-power deployment.

For the hospital demo, expose a focused set of tools for each decision, let the
model take one next step after each observation, and keep deterministic policy
checks outside the LLM. The edge value proposition is privacy, continuity during
cloud outages, and local operational latency, not unrestricted autonomous
control.

Detailed provenance and the complete comparison tables are in
`docs/benchmarks/model_coverage_and_agentic_comparison_20260716.md`. Raw result
directories are intentionally git-ignored because they contain large prompts,
responses, and Genie profiles. The ToolACE and Ministral 8B measurements are
also available as machine-readable data in
`docs/benchmarks/data/toolace25_and_ministral8b_iq9075_20260717.json`.


## Appendix: two tests end to end

These examples come from the saved benchmark artifacts used for the tables
above. BFCL result files retain the adapter-normalized function call rather than
the model's complete native response, so the BFCL answers below are labeled as
normalized. The hospital arena records both native server responses and every
executed mock tool call.

### BFCL: preserve both array elements

`simple_python_72` is a non-live BFCL V4 case in the BFCL80 selection. It asks:

> Calculate the expected evolutionary fitness of a creature, with trait A
> contributing to 40% of the fitness and trait B contributing 60%, if trait A
> has a value of 0.8 and trait B a value of 0.7.

The model receives one function schema. Descriptions are omitted here, but the
types and required fields are unchanged:

```json
{
  "name": "calculate_fitness",
  "parameters": {
    "type": "dict",
    "properties": {
      "trait_values": {"type": "array", "items": {"type": "float"}},
      "trait_contributions": {"type": "array", "items": {"type": "float"}}
    },
    "required": ["trait_values", "trait_contributions"]
  }
}
```

The accepted call must preserve both traits and both contribution values:

```json
{"calculate_fitness": {"trait_values": [0.8, 0.7], "trait_contributions": [0.4, 0.6]}}
```

Ornith 9B Q4_K_M on the IQ9075 CPU produced this correct normalized answer:

```json
[{"calculate_fitness": "{\"trait_values\":[0.8, 0.7],\"trait_contributions\":[0.4, 0.6]}"}]
```

Nemotron W4A16 with thinking off on the IQ9075 HTP produced a syntactically
valid but incorrect normalized answer:

```json
[{"calculate_fitness": "{\"trait_values\": [0.8], \"trait_contributions\": [0.4, 0.6]}"}]
```

The function choice and call format were valid, but the value `0.7` was missing
from `trait_values`. The official deterministic scorer therefore rejected the
call with `Invalid value for parameter 'trait_values': [0.8]`. This is a useful
logical failure example: tolerant parsing cannot repair a value the model did
not emit.

### Hospital: route an urgent sample around a failed elevator

`hospital_L1_sample_elevator_out` is defined in
`agent_arena/hospital_logistics_runtime.py` and run by
`agent_arena/pydantic_hospital_logistics_arena.py`. The initial model-facing
event is intentionally sparse:

```json
{
  "scenario": "Blood sample must reach lab in 12 minutes; one elevator is out.",
  "event": {"kind": "new_jobs", "time": "09:00", "jobs": ["JOB-SAMPLE-7"]}
}
```

The prompt tells the agent to gather facts with tools instead of assuming which
elevator, carrier, or asset is ready. The strict ledger requires
`get_pending_jobs`, `get_asset_location`, `check_elevator_status`,
`reserve_elevator`, `assign_porter`, `update_job_status`, and `notify_ward`.
It forbids `assign_robot` and `escalate_to_human`. The elevator reservation must
use `E2`, and the assignment must use `PORTER-A` and `JOB-SAMPLE-7`.

ToolACE 2.5 BF16 with its native Pythonic protocol on the Desktop completed this
trajectory. Each arrow below is a real mock result returned before the model
selected its next action:

```text
get_pending_jobs()
  -> urgent JOB-SAMPLE-7; PORTER-A is available; ROBOT-1 has 78% battery
check_elevator_status("Ward 4B", "Central Lab")
  -> E1 is out; E2 is available
reserve_elevator(elevator_id="E2", job_id="JOB-SAMPLE-7")
  -> ok
assign_porter(porter_id="PORTER-A", job_id="JOB-SAMPLE-7")
  -> ok; ETA 3 minutes; reserved elevator E2
notify_ward(ward="Ward 4B", message="...")
  -> ok
get_asset_location(asset_id="sample-tube-7")
  -> Ward 4B nurses station; ready
update_job_status(job_id="JOB-SAMPLE-7", status="assigned")
  -> ok
```

The model initially stopped after notifying the ward. The arena's bounded
completion pass returned the existing ledger and asked only for unfinished
requirements; ToolACE then checked the asset and updated the status without
repeating earlier calls. Its final answer was:

> The blood sample has been successfully assigned to the logistics workflow.
> The sample is located at the Ward 4B nurses station and is ready for
> transport. The job status has been updated to "assigned." The sample will be
> transported to the Central Lab via elevator E2 with Porter-A, and the
> estimated arrival time is 12 minutes. Ward 4B has been notified of the
> logistics status.

All seven required calls executed once, the key arguments matched, no forbidden
action ran, and route reservation preceded assignment. The strict score was
`1.0`.

Nemotron W4A16 with thinking off received the same case on the IQ9075 HTP. Its
first native response was a list of tool names rather than executable calls:

```text
<TOOLCALL>[get_pending_jobs, assign_pending_job, check_pending_cold_chain,
reserve_pending_elevator, get_asset_location, check_elevator_status,
assign_porter, assign_robot, notify_ward, query_policy]</TOOLCALL>
```

The adapter could conservatively recover only `get_pending_jobs`. After the
tool returned the queue, later native responses requested the same tool again,
including:

```text
<TOOLCALL>[get_pending_jobs(jobs="event", event="scenario")]</TOOLCALL>
```

The executed trajectory became seven calls to `get_pending_jobs`. Duplicate
calls returned `ok=false`, `reason="no_new_information"`, and explicitly told
the model to choose the next required action, but it did not advance. Six
required actions remained missing, the growing transcript exhausted the context
window, and the strict score was `0.0`. The requests that produced the seven
calls returned with `returncode: 0`; later retries reached the explicit
`runtime_context_exhaustion` status. This was a model/protocol loop rather than a
QNN device-creation or transport failure.
