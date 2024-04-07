"""Given several wav folders, each structured as './{spk}/{utt}.wav', 
build a TSE dataset where each utterance was only used once
Example usage: python build_tse_dataset.py ../test_tse_raw_cn ../test_tse_raw_en ../test_tse_built
"""


from pathlib import Path
from dataset import TSEDatasetArgs, TSEDatasetBuilder, TSEWavDataset
from typing import Iterable, Sequence, TypeVar, Dict
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


@click.command()
@click.argument(
    "src", nargs=-1, type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument(
    "dst", nargs=1, type=click.Path(exists=False, file_okay=False, dir_okay=True)
)
def main(src: Iterable[str], dst: str):
    tse_wav_datasets = []
    for folder in src:
        tse_wav_datasets.append(TSEWavDataset.from_folder(folder))
    merged_tse_wav_dataset = TSEWavDataset(
        merge_dicts(map(lambda v: v.data, tse_wav_datasets))
    )
    tse_dataset_set = TSEDatasetBuilder.from_provider(
        merged_tse_wav_dataset, TSEDatasetArgs()
    )
    tse_dataset_set.to_folder(Path(dst))


if __name__ == "__main__":
    main()
