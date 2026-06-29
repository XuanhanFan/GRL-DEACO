# Data Format

GRL-DEACO consumes scene-level layout JSON files and an optional GLB equipment library.
The routing loaders live in `piping/deaco/layout_io.py` and `piping/deaco/connections.py`.

## Layout JSON

A layout file should provide these sections:

```json
{
  "scene": {
    "bounds": {
      "x": [0.0, 30.0],
      "y": [0.0, 20.0],
      "z": [0.0, 24.0]
    },
    "grid_res": 0.1
  },
  "devices": [
    {
      "name": "Pump_01",
      "model": "pump.glb",
      "position": [3.0, 0.0, 4.0],
      "rotation": [0.0, 0.0, 0.0],
      "ports": [
        {
          "name": "Pump_01.out",
          "position": [3.5, 0.8, 4.0],
          "direction": [1.0, 0.0, 0.0]
        }
      ]
    }
  ],
  "connections": [
    {
      "from": "Pump_01.out",
      "to": "Cooler_01.in"
    }
  ]
}
```

The loaders are intentionally tolerant of richer internal schemas used by generated datasets. For public reproduction, the important requirements are:

- every device has a stable `name`
- each referenced GLB model can be found in the GLB directory
- each connection endpoint resolves to a device port
- port positions and directions are in the same world coordinate system as the scene bounds
- scene bounds are large enough to contain devices, ports, and routed pipes

## GLB Library

The GLB reader extracts mesh bounds, node transforms, and port metadata. Port nodes are detected with the default keyword used by the loader, and their transforms define connection points and outward directions.

If your GLB files do not encode ports, provide explicit port metadata in layout JSON and keep model names resolvable for geometry loading.

## Generated Splits

The paper split is scenario-level, not connection-level. Do not place connections from the same scene into different train/validation/test subsets; that would leak scene geometry and process topology into evaluation.

Recommended folder shape:

```text
scenarios_rl_dataset/
  train/simple/layout_0001.json
  train/medium/layout_0002.json
  train/complex/layout_0003.json
  validation/simple/layout_0065.json
  test/complex/layout_0099.json
```
