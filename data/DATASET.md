# Dataset

This folder is meant to hold `train.csv` — it is **not included** in this repository (kept out via `.gitignore` to avoid redistributing Kaggle's dataset and to keep the repo lightweight).

## How to get it

1. Download `train.csv` from the [Kaggle Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer/data).
2. Place it in this folder so the path looks like:
   ```
   data/train.csv
   ```
3. Make sure the path in `notebooks/main.ipynb` matches:
   ```python
   df = pd.read_csv('data/train.csv')
   ```

That's it — the notebook will pick it up from there.
