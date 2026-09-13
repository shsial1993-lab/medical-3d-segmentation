# 3D Medical Segmentation Baseline

A compact 3D U-Net-style baseline for volumetric segmentation experiments such as
brain-tumor research. The demo creates a synthetic 3D sphere so the forward pass and
overlap metrics can be checked without distributing medical data.

## Highlights

- 3D convolutional encoder-decoder
- Skip connections for volumetric detail
- Dice and binary IoU helpers
- Synthetic smoke test that is safe to run locally

## Run

~~~bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m src.demo
~~~

This repository is for research engineering and education only. It is not a medical
device and the synthetic demo does not establish clinical performance.

## Structure

- src/model.py — 3D U-Net baseline
- src/demo.py — synthetic volume and metric example

## License

Apache-2.0
