# Local data workflow

- `private/maps/`: lossless 1280x720 map screenshots.
- `private/parts/`: lossless screenshots with the parts panel expanded.
- `private/node-templates/<node-kind>/`: manually approved node crops.
- `private/part-templates/<part-id>/`: manually approved part crops.
- `output/`: recognition JSON and debug images.

`private/` and `output/` are ignored. Keep holdout screenshots in a separate
subdirectory and never use them to create templates or tune thresholds.

