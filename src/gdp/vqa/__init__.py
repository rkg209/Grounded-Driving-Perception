"""Stage 2 — driving-scene VQA. A separate namespace from `gdp.detect`/`gdp.data` on purpose: the
detector and the VLM are two separate models, evaluated separately (H6), and never share a code
path that could later imply a joint model.
"""
