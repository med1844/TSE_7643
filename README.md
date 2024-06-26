# 7643 TSE

## Env setup on PACE-ICE

### Install poetry (package & venv manager)

```bash
module load python/3.9
curl -sSL https://install.python-poetry.org | python3 -
```

Then, add this line to your `~/.bashrc`:

```bash
export PATH=$PATH:$HOME/.local/bin
```

### Config poetry

To avoid poetry venvs eating up all your disk quota in Home, please change some settings:

```bash
poetry config cache-dir $HOME/scratch/.cache
poetry config virtualenvs.in-project true
```

### Install dependencies

```bash
poetry install --no-root
```

### Activate poetry venv

```bash
source .venv/bin/activate
```

## Prepare pretrained models

At project root, run:

```bash
mkdir pretrained_models
```

Then, manually download `WavLM-Base+.pt` from [here](https://drive.google.com/file/d/1-zlAj2SyVJVsbhifwpTlAfrgc9qu-HDb/view?usp=share_link) and put it into `pretrained_models/`.

## Prepare dataset

### Remove files too long or too short

```bash
python preprocess_wav_dataset.py datasets/cn datasets/en datasets/jp
```

### Build TSE dataset

There's not much TSE datasets available out there, especially for high-res ones. We have to generate them by ourselves. Luckily it turns out that as long as you have speech datasets with each utterance having it's speaker information, you can generate a TSE dataset based on that information.

Generate dataset during each run is time consuming, especially when the training dataset is large. Thus we generate the TSE dataset first before training.

You need `build_tse_dataset.py` to build the dataset. It's ok to have multiple valid datasets, where each of them has structure "./{spk_id}/*.wav". Here's an example:

```bash
python build_tse_dataset.py datasets/cn datasets/en datasets/jp datasets/output_tse
```

In the example above, all three input has subdirectories and files that matches `datasets/cn/{spk_id}/*.wav`, `datasets/en/{spk_id}/*.wav`, `datasets/jp/{spk_id}/*.wav`. The result would be write into the output folder `datasets/output_tse`.

## Training

With default config, you can start training with this command:

```bash
python train.py --dataset datasets/output_tse
```

Or if you wish to use custom config, pass `--config` or `-c`:

```bash
python train.py -c configs/default.json -d datasets/output_tse
```

## Hyperparam search

We use ray tune to search for the best hyper parameters. Here's an example command:

```bash
python grid_search.py -c configs/small.json -d datasets/output_tse -n 5 --wavlm_pt $(pwd)/pretrained_models/WavLM-Base+.pt
```

Ensure the `--wavlm_pt` is an absolute path.

To modify the parameters to be searched, please refer to `search_space` in `grid_search.py`.
