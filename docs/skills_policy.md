# 5 Skill Usage Policy

The five skills are process helpers. They are not stock data sources, labels, or prediction answers.

## Skill Roles

- Skill 01 UI/UX: report or interface design only after the model passes validation.
- Skill 02 Debugging: reproduce, localize, reduce, fix, guard when data flow, tests, or reports fail.
- Skill 03 Architecture: define clean project layers and boundaries.
- Skill 04 Codex skill catalog: tool discovery only; no automatic tool adoption.
- Skill 05 agent skill catalog: tool discovery only; no automatic tool adoption.

## Hard Rules

- Only three data inputs are allowed by `project_config.json`.
- Skills cannot add model answers, hidden labels, or external conclusions.
- Experiment output cannot become formal output until it passes validation.
- If a score is not calibrated, it must not be presented as probability.

