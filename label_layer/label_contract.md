# Label Contract

- Signal time: after the signal-day close.
- Buy assumption: next trading day open.
- Success: a close in the next 1 to 10 trading days reaches +3 percent from the buy open.
- Drawdown side risk: any -3 percent low is a risk label, not an automatic failure.
- If price first reaches -3 percent low and later reaches +3 percent close, primary success remains true.
- `risk_adjusted_10d_success` is kept only as a hard-risk comparison field.
- Unfinished: if the future 10 trading day window is incomplete, the sample is tracking-only.
- Same-day market comparison is validation support, not a stock label by itself.
- Episode grouping prevents the same stock from being repeatedly counted as new within one short wave.
