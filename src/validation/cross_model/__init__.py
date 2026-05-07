"""
Cross-model validation: re-run J1/J2 on a stratified subset of the benchmark
using two additional model families (Claude Sonnet 4.6 + Gemini 2.5 Pro) and
compute inter-judge agreement.

Purely additive — does not modify the existing pipeline. All judge prompts,
policy texts, and indicators are imported unchanged from the parent
``validation`` package.

Entry point: ``src.validation.cross_model.run_cross_model_validation``.
"""
