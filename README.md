# Multi-Interest Self-Supervised Learning

This is the code for our proposed model **Multi-Interest Self-Supervised Learning**. The experiment is conducted on n 2*NVIDIA GeForce RTX 2080ti GPU. 
The code is built upon [MB-STR](https://github.com/yuanenming/mb-str).

## Requirements

Run the following code to satisfy the requirements by pip:

```
pip install -r requirements.txt
```

## Datasets

- The `Yelp` dataset is already located in the `\data` folder. To obtain the other two public datasets, you can download them at: https://drive.google.com/file/d/1qOf2-Mwag0qzT6bPLFrQ7cNd_S1QDBbw/view?usp=drive_link
- unzip the `datasets.zip`
- Place the datasets into the `\data` folder.

## Run MISSL

- Run the model on `Yelp`  with a `yaml` configuration file like following:

```bash
python run.py fit --config src/configs/yelp/yelp_missl.yaml
```