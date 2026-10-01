# Operational Investigation Agent

Initial hackathon prototype for an AI operational investigator focused on energy and industrial operations.

## Current milestone

Synthetic evidence layer only. The first implementation intentionally builds the data and deterministic evidence base before adding an LLM.

## Core loop

Detect anomaly -> generate hypotheses -> gather evidence -> test hypotheses -> identify probable cause -> recommend intervention -> verify -> store verified outcome.

## Dataset

- `machines.csv` — machine metadata and rated power
- `energy.csv` — daily machine runtime and energy readings
- `production.csv` — daily factory production
- `maintenance.csv` — maintenance history and status
- `incidents.csv` — verified historical incidents
- `technician_notes.csv` — human operational observations
- `scenarios.json` — three demo scenarios

## Important design rule

The LLM must not invent numerical evidence. Python tools will calculate metrics from these files; the model will reason over the returned evidence.

## Optional reasoning layer

The investigator runs fully without a model. When a credential is configured,
an optional reasoning layer interprets the already-computed evidence; it never
changes measurements, verdicts, or the selected root cause.

- **TypeSafe System One (preferred):** set `TYPESAFE_API_KEY`. The adapter asks
  one Noul question per candidate explanation and thresholds the returned
  probabilities in code — middle values stay `inconclusive` for review.
  Optional: `TYPESAFE_MODEL`, `TYPESAFE_BASE_URL`,
  `TYPESAFE_SUPPORT_THRESHOLD`, `TYPESAFE_REJECT_THRESHOLD`.
- **OpenAI and/or Groq (OpenAI-compatible):** set `OPENAI_API_KEY` and/or
  `GROQ_API_KEY`. `LLM_PROVIDER` selects the provider (defaults to whichever
  key is present) and `LLM_FALLBACK_PROVIDER` adds a second provider that is
  only tried when the first fails. `LLM_API_KEY`, `LLM_MODEL` and
  `LLM_BASE_URL` remain available for any other OpenAI-compatible endpoint.

Any disagreement between the model and the measured evidence is recorded as a
conflict in the investigation trace; the deterministic verdict is kept.

Copy `.env.example` to `.env` and fill in the key(s); the CLI and the API load
`.env` automatically (the test suite never reads it). See `.env.example` for
all variables.
