# V1 Solar Irradiance Forecasting

Reproducible GHI forecasting baseline with:
- multivariate meteorological inputs
- 24-step sequences
- LSTM and Transformer baselines
- chronological validation
- city-level holdout for generalization

Run:

```bash
pip install -r requirements.txt
python -m src.train --config configs/solar.yaml
python -m src.evaluate --config configs/solar.yaml
```
