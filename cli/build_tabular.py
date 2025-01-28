import argparse
import os

import h5py
import numpy as np
import polars as pl
from tqdm import tqdm


def get_data_from_source(data_dir: str) -> dict:
    data = {
        "work": [],
        "performance": [],
        "chroma_cens": np.empty(
            [
                12,
            ]
        ),
        "crema": np.empty(
            [
                12,
            ]
        ),
        "hpcp": np.empty(
            [
                12,
            ]
        ),
        "mfcc_htk": np.empty(
            [
                13,
            ]
        ),
        "novfn": [],
        "snovfn": [],
        "tempo": [],
    }

    for work in tqdm(
        os.listdir(data_dir), bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"
    ):
        if work in [".DS_Store", "README.md", "LICENSE.txt"]:
            continue
        else:
            for perf in os.listdir(os.path.join(data_dir, work)):
                file = os.path.join(data_dir, work, perf)

                with h5py.File(file, "r") as f:
                    chroma_cens = np.array(f["chroma_cens"]).mean(axis=0)
                    crema = np.array(
                        f["crema"]
                    ).mean(
                        axis=0
                    )  # coveranalysis need axis=1, while benchmark needs axis=0 for crema
                    hpcp = np.array(f["hpcp"]).mean(axis=0)

                    mfcc_htk = np.array(f["mfcc_htk"])
                    mfcc_htk = np.ma.masked_invalid(mfcc_htk).mean(axis=1).data

                    novfn = np.array(f["/madmom_features"]["novfn"]).mean().item()
                    snovfn = np.array(f["/madmom_features"]["snovfn"]).mean().item()
                    tempo = np.array(f["/madmom_features"]["tempos"])[0, 0].item()

                data["work"].append(work)
                data["performance"].append(perf)
                data["chroma_cens"] = np.vstack([data["chroma_cens"], chroma_cens])
                data["crema"] = np.vstack([data["crema"], crema])
                data["hpcp"] = np.vstack([data["hpcp"], hpcp])
                data["mfcc_htk"] = np.vstack([data["mfcc_htk"], mfcc_htk])
                data["novfn"].append(novfn)
                data["snovfn"].append(snovfn)
                data["tempo"].append(tempo)

    data["chroma_cens"] = data["chroma_cens"][1:, :].tolist()
    data["crema"] = data["crema"][1:, :].tolist()
    data["hpcp"] = data["hpcp"][1:, :].tolist()
    data["mfcc_htk"] = data["mfcc_htk"][1:, :].tolist()

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_file_name", required=True)

    args = parser.parse_args()

    coveranalysis_data = get_data_from_source(args.data_dir)

    data = pl.from_dict(coveranalysis_data)
    data = data.with_columns(
        pl.col("chroma_cens").list.to_struct(fields=lambda idx: f"chroma_cens_{idx}")
    ).unnest("chroma_cens")
    data = data.with_columns(
        pl.col("crema").list.to_struct(fields=lambda idx: f"crema_{idx}")
    ).unnest("crema")
    data = data.with_columns(
        pl.col("hpcp").list.to_struct(fields=lambda idx: f"hpcp_{idx}")
    ).unnest("hpcp")
    data = data.with_columns(
        pl.col("mfcc_htk").list.to_struct(fields=lambda idx: f"mfcc_htk_{idx}")
    ).unnest("mfcc_htk")

    os.makedirs("tabular_data", exist_ok=True)

    data.write_csv(f"tabular_data/{args.save_file_name}.csv")
