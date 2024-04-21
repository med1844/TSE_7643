"""Given several wav folders, each structured as './{spk}/{utt}.wav', 
build a TSE dataset where each utterance was only used once
Example usage: python build_tse_dataset.py ../test_tse_raw_cn ../test_tse_raw_en ../test_tse_built
"""


import os
from pathlib import Path
from dataset import (
    LazyLoadable,
    TSEDatasetArgs,
    TSEDatasetBuilder,
    TSEWavDataset,
    Audio,
)
from typing import Iterable, Sequence, TypeVar, Dict, List, Optional
from random import choice, shuffle
import click


K, V = TypeVar("K"), TypeVar("V")


def merge_dicts(ds: Iterable[Dict[K, Sequence[V]]]) -> Dict[K, Sequence[V]]:
    merged_dict = {}
    for d in ds:
        for k, v in d.items():
            if k not in merged_dict:
                merged_dict[k] = v
            else:
                merged_dict[k].extend(v)
    return merged_dict


def draw_subset(
    spk_utt_data: Dict[str, List[LazyLoadable[Audio]]],
    num_utts: int,
) -> Dict[str, Sequence[LazyLoadable[Audio]]]:
    subset = {}
    skps = list(spk_utt_data.keys())
    for _ in range(num_utts):
        key = choice(skps)
        shuffle(spk_utt_data[key])
        if key not in subset:
            subset[key] = list()
        subset[key].append(spk_utt_data[key].pop())
        if len(spk_utt_data[key]) == 0:
            skps.remove(key)
            del spk_utt_data[key]
    return subset


@click.command()
@click.argument(
    "src", nargs=-1, type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument(
    "dst", nargs=1, type=click.Path(exists=False, file_okay=False, dir_okay=True)
)
@click.option("--max_utts", type=int)
def main(src: Iterable[str], dst: str, max_utts: Optional[int]):
    tse_wav_datasets = []
    for folder in src:
        tse_wav_datasets.append(
            TSEWavDataset.from_folder(folder, suffix=os.path.basename(folder))
        )
    merged_spk_utts_dict = merge_dicts(map(lambda v: v.data, tse_wav_datasets))
    if max_utts is not None:
        merged_spk_utts_dict = draw_subset(merged_spk_utts_dict, max_utts)
    merged_tse_wav_dataset = TSEWavDataset(merged_spk_utts_dict)
    Path(dst).mkdir(parents=True, exist_ok=True)
    _ = TSEDatasetBuilder.from_provider(merged_tse_wav_dataset, TSEDatasetArgs(), dst)


if __name__ == "__main__":
    main()
