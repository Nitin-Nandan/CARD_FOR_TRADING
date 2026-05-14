# Experiment Registry
_Updated automatically after each experiment._

## [EXP-000] Fresh Start Initialization
**Date**: 2026-05-02
**Hypothesis**: Establishing a clean environment for the fresh start.
**Config**: N/A
**Result**: N/A
**Verdict**: CONFIRMED
**Key finding**: Structural debt resolved, agent council activated.
**Next step taken**: Begin Research Agent and Data Agent workflow.

## [EXP-001] CARD Suitability for NSE
**Date**: 2026-05-03
**Hypothesis**: CARD's channel-aligned attention and robust loss function can be adapted for highly non-stationary NSE stock data by converting prices to returns and utilizing its uncertainty weights.
**Config**: N/A (Research phase)
**Result**: Research findings summarized in `docs/research/CARD_NSE_Stocks`.
**Verdict**: CONFIRMED (Proceed to implementation/testing)
**Key finding**: Must use log-returns, may need denoising step for the channel attention. Official repo `wxie9/CARD` is the recommended starting point.
**Next step taken**: Hand off to Implementation/Code Agent.
