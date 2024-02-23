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

## Prepare dataset

Ensure you are at the root directory of the repository. Then:

```bash
mkdir datasets
cd datasets
git clone https://huggingface.co/datasets/hanamizuki-ai/genshin-voice-v3.5-mandarin
```

This would take a long time.

After this is finished, execute:

```bash
cd ..
poetry run python main.py -c configs/v0.json -p datasets/genshin-voice-v3.5-mandarin/data/
```

