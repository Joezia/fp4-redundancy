"""Portable redundancy analysis framework (spec 007).

Three layers: adapters (data -> `dataset.Entry`) -> indicators (Entry -> result files) -> plot
(result files -> figures). Indicators, baselines, break-even formulas and plotting know nothing
about any particular data format.
"""
