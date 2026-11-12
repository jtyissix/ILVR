# Upstream sources

Files in this directory are preserved upstream evaluation code or prompts. Only
selected pure functions/constants are loaded by `eval/official.py`; their CLI,
network clients, and model constructors are not run. Their original imports do
not represent additional requirements of this evaluator.

`sources.json` identifies repository, source path, revision for each
copied code file. The surrounding adapters in `eval/protocols.py` explicitly
document changes in `eval/README.md`.

- MM-Math: kge-sun/MM-Math; see `LICENSE.MM-Math`.
- MathVista: lupantech/MathVista; see `LICENSE.MathVista`.
- VisualPuzzles: neulab/VisualPuzzles; see `LICENSE.VisualPuzzles`.
- BLINK: zeyofu/BLINK_Benchmark; see `LICENSE.BLINK`.
- VisuLogic: VisuLogic-Benchmark/VisuLogic-Eval; see `LICENSE.VisuLogic`.
- EMMA prompts/config: EMMA-Bench/EMMA. The inspected repository did not contain
  a top-level license file; no new license is asserted for these upstream files.
- ScienceQA: lupantech/ScienceQA. Refer to the upstream repository for licensing.
- MMStar: MMStar-Benchmark/MMStar. Refer to the upstream repository for licensing.

Dataset usage terms remain those of the respective dataset repositories. Copies
of downloaded dataset READMEs and available licenses are kept under `raw/`.
