# Data Folder

No raw data files are stored in this folder — the dataset is fetched
automatically by the code in `src/data_loader.py`.

- **Primary source:** MNIST (70,000 images), fetched via
  `tensorflow.keras.datasets.mnist.load_data()`. This is the same data as
  Kaggle's [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)
  competition dataset.
- **Offline fallback:** scikit-learn's bundled handwritten digits dataset
  (1,797 images), used automatically if no internet connection is available.

If you prefer to use the Kaggle CSV files directly (`train.csv` /
`test.csv`), download them from the Kaggle competition page linked above and
place them here, then adapt `src/data_loader.py` to read from CSV instead.
