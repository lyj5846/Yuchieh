# Label Contract

- Signal time: after the signal-day close.
- Buy assumption: next trading day open.
- Success: a close in the next 1 to 10 trading days reaches +3 percent before any low in that same window reaches -3 percent from the buy open.
- Conservative tie rule: if the +3 percent close and -3 percent low happen on the same trading day, success is false.
- `old_target_success` keeps the old +3 percent touch rule for comparison only.
- Unfinished: if the future 10 trading day window is incomplete, the sample is tracking-only.
- Same-day market comparison is validation support, not a stock label by itself.
- Episode grouping prevents the same stock from being repeatedly counted as new within one short wave.
