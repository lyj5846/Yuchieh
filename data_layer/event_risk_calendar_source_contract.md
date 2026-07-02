# Event Risk Calendar Source Contract

- Status: specification only; no new data source is enabled yet.
- Purpose: define the event-risk data needed before the next model retraining attempt.
- Current formal output: unchanged.
- Source decision: `decision_layer\data_enrichment_decision.json`

## Why This Exists

目前交易目標本身可用，但模型學不到先跌風險；最直接缺口是事件與注意處置類風險旗標。

The current three CSV files cover price, volume, chip, market state, and static industry groups. They do not show whether a stock had a known event that could explain a first adverse move before the +3% target is reached.

## Proposed Future File

- Future file path: `inputs/event_risk_calendar.csv`
- This file is not added to `project_config.json` in this step.
- A collector must be reviewed before this file can become an approved model input.

## Strict Timing Rule

- Main model features may only use events already public by signal-day close.
- Events published after signal-day close but before next open must be stored separately.
- Post-close events cannot enter the main model unless the decision-time contract is explicitly changed.
- Missing `announcement_datetime` means the row is not eligible for training.
- Later news explanations or human interpretations cannot be backfilled into earlier signal dates.

## Raw Event Schema

| field | type | required | description |
|---|---|---|---|
| stock_id | string | yes | Taiwan stock code, normalized to text. |
| stock_name | string | yes | Stock name from the event source or mapped from the approved theme file. |
| event_type | enum | yes | Mechanical event class: material_info, suspension, resumption, attention, disposition, investor_meeting, ex_dividend, corporate_action. |
| event_subtype | string | no | More detailed source subtype when available. |
| event_title | string | yes | Original event title or brief source text. |
| source_name | string | yes | Source family name, such as exchange notice, public disclosure, or company event feed. |
| source_url | string | no | Source URL or stable reference when available. |
| announcement_datetime | datetime | yes | The first public timestamp when the event was known. |
| event_effective_start_date | date | no | Date when the event starts to apply, such as disposition start date. |
| event_effective_end_date | date | no | Date when the event stops applying. |
| signal_usable_date | date | yes | First signal date on which this event can be used under the strict close-time rule. |
| known_before_signal_close | boolean | yes | Whether the event was publicly known by the signal day's close. |
| post_close_pre_next_open | boolean | yes | Whether the event was known after close but before next open. |
| raw_payload_hash | string | no | Hash of raw downloaded payload for audit and de-duplication. |

## Source Families

| source family | event types | required timestamp | collector note |
|---|---|---|---|
| public_disclosure_material_info | material_info, investor_meeting, corporate_action | announcement_datetime | Use the first published timestamp and retain source title. |
| exchange_attention_disposition | attention, disposition | announcement_datetime | Capture start and end dates when provided. |
| exchange_suspension_resumption | suspension, resumption | announcement_datetime | Capture effective trading date and reason text. |
| corporate_action_calendar | ex_dividend, corporate_action | announcement_datetime | Capture announced date and effective date separately. |

## Future Feature Contract

| feature | window | meaning | use rule |
|---|---|---|---|
| event_any_known_1d | 1 trading day | Any known event near the signal date. | Count only events with signal_usable_date <= signal_date. |
| event_any_known_3d | 3 trading days | Recent event density before the signal. | Rolling count known by signal close. |
| event_attention_or_disposition_active | active date range | Whether attention or disposition status is active and already known. | Active on signal date and published before the usable cutoff. |
| event_suspension_or_resumption_nearby | 10 trading days | Whether trading halt or restart event is close enough to affect price behavior. | Use only after source announcement is known. |
| event_material_info_known_5d | 5 trading days | Material information event count before the signal. | No NLP judgement in the first collector; count and type only. |
| event_ex_dividend_or_corporate_action_10d | 10 trading days | Known corporate action that may distort price path. | Use announced events only; no later adjustment backfill. |

## Boundaries

- This step does not fetch data.
- This step does not train or promote a model.
- This step does not choose stocks.
- This step does not change the three approved CSV inputs.
- Event categories are mechanical source facts, not manual success or failure explanations.
