# (Improved) CIC IDS 2017

Link to the dataset: [https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/](https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/).

Steps to get the CSV files used in this repo:
- Download the dataset
- Unzip it in the `./data/cicids2017` directory
- Run `python -m data.cicids2017.merge_dataset` to get `cicids2017.csv`
- Run `python -m data.cicids2017.get_random10` to get `cicids2017_random10.csv`