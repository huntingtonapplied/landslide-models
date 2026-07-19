# landslide-models

Runnable neural-surrogate inference for the **Landslide** stochastic-simulation platform —
model inputs in, physical fields out, in two lines. No PyTorch: the model forward is baked
into an ONNX graph and runs under `onnxruntime` on CPU.

```python
from landslide_models import load

model = load("landslide_terrain_deformation_v1")
out = model.predict(inputs={
    "node_features":  node_features,   # [N, 12] terrain/material/stress features
    "edge_index":     edge_index,      # [2, E] mesh connectivity
    "node_positions": node_positions,  # [N, 3]
    "edge_weight":    edge_weight,      # [E]
})
# -> {"displacement":[N,3], "strain":[N,6], "failure_probability":[N,1], "confidence":[N,1]}
```

`model.input_names` / `model.output_names` list a craft's exact I/O; see each craft's
`MODEL_CARD.md`.

## Install

```
pip install landslide-models      # numpy + onnxruntime only, no torch
```

## Scope

- **Preview build**: ships `landslide_terrain_deformation_v1`; more crafts follow.
- **Fast surrogates, not the ground-truth solver** — validate against the numerical engine
  where correctness is critical.

## License

Closed-source, compiled distribution under the **AHL Model EULA** (see `LICENSE`).
© 2025–2026 Huntington Applied.
