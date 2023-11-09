# Multi-Interest Self-Supervised Learning

This is the code for our proposed model **Multi-Interest Self-Supervised Learning**. The experiment is conducted on n 2*NVIDIA GeForce RTX 2080ti GPU. 
The code is built upon [MB-STR](https://github.com/yuanenming/mb-str).

## Requirements

Run the following code to satisfy the requirements by pip:

```
pip install -r requirements.txt
```

## Datasets

- The `Yelp` dataset is already located in the `\data` folder. To obtain the other two public datasets, you can download them at:

  Google Drive: https://drive.google.com/file/d/1qOf2-Mwag0qzT6bPLFrQ7cNd_S1QDBbw/view?usp=drive_link

  Baidu Drive: https://pan.baidu.com/s/1W5ASbe2rqpEEvpZccAZ3Ow?pwd=ey5h

- unzip the `datasets.zip`

- Place the datasets into the `\data` folder.

## Run MISSL

- Train and validate the model on training set of  `Yelp`  with a `yaml` configuration file like following:

```bash
python run.py fit --config src/configs/yelp/yelp_missl.yaml --data.develop True
```

- Test the model on test set of  `Yelp`  with a `yaml` configuration file like following:

```bash
python run.py validate --config src/configs/yelp/yelp_missl.yaml --data.develop False --ckpt_path [yours checkpoint path] 
```

