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
