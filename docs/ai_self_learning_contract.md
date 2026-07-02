# AI Self-Learning Contract

## Fixed Target

After signal-day close, use only information available on or before that day. Assume buying at the next trading day open. A signal succeeds if any close in the next 1 to 10 trading days reaches at least 3 percent above that buy open.

Signals without a full 10 trading day future window are tracking-only and cannot be counted as success or failure.

## Learning Loop

Each experiment must record:

- What changed.
- Why it changed.
- Development result.
- Holdout result.
- Whether it is kept, rejected, or used for the next experiment.
- Whether there is overfit risk.

## Promotion Gate

A model can enter formal output only when:

- It beats baseline on holdout.
- Its calibrated probability is close to actual hit rate.
- Higher probability groups are not worse than lower groups.
- Daily Top-K is useful and not too large.
- It does not depend on a tiny group of stocks or dates.

Until then, formal output remains: `目前無可信正式候選`.

