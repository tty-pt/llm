# 📘 Memory Mipmaps for Low-End LLMs

## Overview
**Memory Mipmaps** is a lightweight, inference-minimizing memory architecture that allows **low-resource language models** (LLMs), like Mistral Q2/Q3, to simulate **persistent, structured long-term memory**.

This system is designed for **offline**, **edge**, or **local-first AI deployments** where inference is expensive and must be kept to a minimum. It combines external memory lookup with controlled LLM execution, so that the model only infers what’s needed — everything else is resolved and formatted externally. This allows very low cost memory access and output generation within strict token budgets.


## Example Interaction
See an example of an interaction [here](./memory-mipmaps-example.txt).

## Goals
* ✅ Run efficiently on **consumer-grade devices** (4–6GB RAM/VRAM).
* ✅ Provide structured and scalable long-term memory access.
* ✅ Avoid hallucinations and prompt drift.
* ✅ Save inference by feeding only summaries and bracketed responses (see example interaction).
* ✅ Emulate human-like memory navigation without large context windows.

## Inspirations

### 🎮 Mipmaps (Computer Graphics)

In rendering, **mipmaps** provide progressively lower-resolution images based on distance.
Memory Mipmaps applies the same idea:
LLMs access memory in **layers**, from vague summaries to precise transcripts.

### 🔬 Quantum Uncertainty

Just as you can't precisely measure both position and momentum, you must trade **breadth for depth** in memory recall — either view broad summaries, or zoom into exact past events.

### 🧠 Cognitive Recall

Humans first remember the **gist**, then reconstruct **details**.
This system mimics that behavior by accessing memory hierarchically through `level 2 → level 1 → level 0`.

## 🧱 Architecture

### Layer 1: **External Memory (Queryable DB)**

All long-term memory is stored in a compact indexed database (e.g. `qdb`), using time intervals and semantics. With entries grouped into:

| Level | Resolution         | Example Key                 |
| ----- | ------------------ | --------------------------- |
| 2     | High-level summary | `"mirror@2025-05"`          |
| 1     | Condensed memory   | `"mirror@2025-05-14T17:05"` |
| 0     | Raw transcript     | `"@2025-05-14T17:05"`       |

(for example)

Memory is accessed with commands like:

```plaintext
memory:scan "topic" level 2
memory:scan "topic@timestamp" level 1
memory:scan "topic@timestamp" level 0
```

The LLM decides which commands should be run based on the context and previous output — but it does not execute them. Instead, the external system resolves those commands and feeds the results back into context.

### Layer 2: **Controlled LLM Execution**
The LLM operates inside a strict cycle with minimal responsibility:

1. Receives prompt.
2. Decides how to lookup memory.
3. Does short summaries of the interaction.
4. Decides which commands to run.
5. And when it has enough information, infers the result.

## 📜 Protocol Summary

### Available Commands (Resolved by Daemon):

```plaintext
memory:scan "topic" level 2 → returns topic summaries
memory:scan "topic@timestamp" level 1 → returns compressed insight
memory:scan "topic@timestamp" level 0 → returns full entry
memory:think "timestamp" extract:intention
memory:think "timestamp" extract:value
memory:think "timestamp" extract:emotion
memory:store timestamp [summary or new insight]
memory:forget topic|tag
memory:reset
```

## 🧠 Contextual Reinforcement Block

Injected **on session start**, **on drift**, or **on topic reset**, this maintains structure:

## 🛠 Performance and Design Benefits

### 🧮 Inference Minimization

* Only final output is computed by the model.
* All scans, timestamps, summaries are **precomputed**.

### 🧵 Structural Stability

* Deterministic flow avoids drift.
* Prompt resets reduce context complexity.
* Works well even on models with tiny context windows.

### 📦 Efficient Resource Use

* Enables long memory with low memory footprint.
* Scales with memory DB, not LLM context.
* Allows "infinite" memory for lightweight models.

## 📌 Use Cases

* Symbolic storytelling agents in persistent game worlds.
* Offline journaling tools with consistent character.
* Assistants on embedded or low-power devices.

## 🔓 License

This project is released under the [BSD-2-Clause License](./LICENSE).

© Paulo André.
**This is so that this idea is never restricted to anyone.** Nobody who has a modest computer should be cut off from using it.
