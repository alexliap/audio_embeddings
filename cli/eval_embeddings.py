import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from audio_embeddings import AutoEncoder, AutoEncoderModel, Decoder, Encoder

bench = pl.read_csv("tabular_data/benchmark_tabular.csv")

layer_sizes = [52, 300, 200, 100, 50, 20]

enc = Encoder(layer_sizes)
dec = Decoder(layer_sizes[::-1])

module = AutoEncoder(encoder=enc, decoder=dec)

model = AutoEncoderModel(autoencoder=module)

model.load_state_dict(
    state_dict=torch.load(
        "lightning_logs/model/epoch=328-step=6580.ckpt", weights_only=True
    )["state_dict"]
)


def get_in_clique_dist(dataset: pl.DataFrame, clique_name: str) -> float:
    """Get the average distance of a performance from the rest performances of a certain clique.

    Args:
        dataset (pl.DataFrame): The dataset that contains the clique data.
        clique_name (str): Name of the clique.

    Returns:
        float: The average Euclidean distance between a clique's perfromances.
    """
    avg_distance = 0
    clique_data = dataset.filter(pl.col("work") == clique_name).drop("work")
    if clique_data.shape[0] > 1:
        for perf in clique_data["performance"].unique().to_list():
            tmp = (
                clique_data.filter(pl.col("performance") == perf)
                .drop("performance")
                .to_numpy()
                .astype(np.float32)
            )
            tmp = tmp.reshape(1, -1)

            tmp_not = (
                clique_data.filter(pl.col("performance") != perf)
                .drop("performance")
                .to_numpy()
                .astype(np.float32)
            )

            avg_distance += model.get_distance(tmp, tmp_not)

    avg_distance = avg_distance / clique_data.shape[0]

    return avg_distance


def get_out_clique_dist(
    dataset: pl.DataFrame, clique_name: str, clique_samples: int = 10
) -> float:
    """Get the average distance of each clique's performances from random samples of other performances
    that belong to different cliques.

    Args:
        dataset (pl.DataFrame): The dataset that contains the clique data.
        clique_name (str): Name of the clique.
        clique_samples (int, optional): The number of different cliques to consider for distance calculation. Defaults to 10.

    Returns:
        float: The average Euclidean distance between a clique's perfromances and random samples of other performances
               that belong to different cliques.
    """
    avg_distance = 0

    other_cliques = (
        dataset.filter(pl.col("work") != clique_name)["work"]
        .sample(clique_samples)
        .to_list()
    )
    non_clique_data = (
        dataset.filter(pl.col("work").is_in(other_cliques))
        .drop(["work", "performance"])
        .to_numpy()
        .astype(np.float32)
    )

    clique_data = dataset.filter(pl.col("work") == clique_name).drop("work")
    for perf in clique_data["performance"].unique().to_list():
        tmp = (
            clique_data.filter(pl.col("performance") == perf)
            .drop("performance")
            .to_numpy()
            .astype(np.float32)
        )
        tmp = tmp.reshape(1, -1)

        avg_distance += model.get_distance(tmp, non_clique_data)

    avg_distance = avg_distance / clique_data.shape[0]

    return avg_distance


if __name__ == "__main__":
    in_clique_dists = []
    out_clique_dists = []

    for work in tqdm(
        bench["work"].unique().to_list(), bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"
    ):
        in_clique_dists.append(get_in_clique_dist(bench, work))

        out_clique_dists.append(get_out_clique_dist(bench, work, clique_samples=100))

    plt.hist(
        np.array([dist for dist in in_clique_dists if dist > 0]),
        label="In-clique distance",
        bins=40,
    )
    plt.hist(
        np.array(out_clique_dists), label="Out of clique distance", bins=40, alpha=0.6
    )
    plt.legend()
    plt.xlabel("Euclidean Distance")
    plt.title("In-clique VS Out of clique distances")
    plt.savefig("pics/in_clique_vs_out_clique_dists.jpeg")
