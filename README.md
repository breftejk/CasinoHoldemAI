# CasinoHoldemAI

A decision engine for Casino Hold’em that combines Monte Carlo simulation with XGBoost.

## Documentation

The full documentation is available as a PDF:

- [docs/documentation.pdf](docs/documentation.pdf)

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel Cython
pip install .
```

## CLI Usage

- **Generate training data**  
  ```bash
  casino-ai gen \
    --n 200000 \
    --out data/train.csv \
    --iters 1000 \
    --workers 8
  ```

- **Train XGBoost model**  
  ```bash
  casino-ai train \
    --in data/train.csv \
    --model models/holdem.xgb
  ```

- **Predict CALL/FOLD decision**  
  ```bash
  casino-ai pred \
    --model models/holdem.xgb \
    --cards QS,QH \
    --board QD,2C,7H \
    --threshold 0.6
  ```

## License

MIT © 2025 Marcin Kondrat (@breftejk)  
*Use at your own risk.*