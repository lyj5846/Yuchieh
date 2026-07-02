# Skill Adoption Audit

This audit is intentionally strict: a skill is not marked as adopted unless it is actually available to this Codex session and its instructions were read.

## Current Result

Ten actionable external skills were installed into the local Codex skills folder on 2026-07-02.

They require a Codex restart before they appear in the active skill list for future turns.

Two list-style repositories are still treated as references only because they do not provide one single root skill to install.

None of the five items are data sources, labels, model answers, or hidden rules.

## Role Mapping

| Item | Intended Role | Current Adoption Status | Project Layer |
|---|---|---|---|
| Skill 01 UI/UX design | Report or interface design after a model passes | Installed as `ui-ux-pro-max`; restart required | Future report design only |
| Skill 02 debugging flow | Reproduce, locate, reduce, fix, guard | Installed as `debugging-and-error-recovery`; restart required | Architecture checks and validation failures |
| Additional testing discipline | Write tests around behavior before trusting changes | Installed as `test-driven-development`; restart required | Pipeline validation and model-result guards |
| Additional code quality review | Review bugs, risk, regressions, and maintainability | Installed as `code-review-and-quality`; restart required | Main pipeline review before formal use |
| Planning support | Break work into scoped steps before implementation | Installed as `planning-and-task-breakdown`; restart required | Pipeline design and staged model work |
| Documentation support | Keep architecture decisions and contracts readable | Installed as `documentation-and-adrs`; restart required | Project docs and model decision records |
| Source-driven work | Keep implementation grounded in approved sources | Installed as `source-driven-development`; restart required | Three-CSV-only evidence discipline |
| Automation flow support | Build repeatable runner/check workflows | Installed as `ci-cd-and-automation`; restart required | Daily sync and validation commands |
| Spreadsheet support | Assist formula/table style spreadsheet work | Installed as `spreadsheet-formula-helper`; restart required | CSV/table inspection and report tables |
| Skill 03 project architecture | Clean project boundaries | Installed as `project-scaffolding`; restart required | Project boundary review |
| Skill 04 Codex skill list | Tool discovery | Not installed as one skill; repository is a catalog | Reference only |
| Skill 05 agent skill list | Tool discovery | Not installed as one skill; repository is a catalog | Reference only |

## Hard Boundary

The skills cannot:

- Add stocks.
- Add labels.
- Add external facts.
- Turn a research score into a success rate.
- Override the three allowed CSV inputs.

Installed sources:

- `addyosmani/agent-skills`, path `skills/debugging-and-error-recovery`
- `addyosmani/agent-skills`, path `skills/test-driven-development`
- `addyosmani/agent-skills`, path `skills/code-review-and-quality`
- `addyosmani/agent-skills`, path `skills/planning-and-task-breakdown`
- `addyosmani/agent-skills`, path `skills/documentation-and-adrs`
- `addyosmani/agent-skills`, path `skills/source-driven-development`
- `addyosmani/agent-skills`, path `skills/ci-cd-and-automation`
- `hmohamed01/Claude-Code-Scaffolding-Skill`, path `project-scaffolding`
- `nextlevelbuilder/ui-ux-pro-max-skill`, path `.claude/skills/ui-ux-pro-max`
- `ComposioHQ/awesome-codex-skills`, branch `master`, path `spreadsheet-formula-helper`

Reference-only sources:

- `ComposioHQ/awesome-codex-skills`
- `VoltAgent/awesome-agent-skills`

If any catalog item is installed later, this file must be updated with the exact source and the layer where it is allowed to operate.
