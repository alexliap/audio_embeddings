# Audio Embeddings

# Table of Contents
1. [Data](#data)
2. [Clustering](#clustering)
3. [Embedding Creation](#embedding-creation)
    1. [Evaluation](#evaluation)
    2. [Train your Autoencoder](#train-your-own-autoencoder)
    3. [Compute Distance](#compute-distance)

This is a repository dedicated to experimentation on song similairity modeling based on pre-extrcacted features. Two routes are explored, one using traditional ML clustering that employs the K-Means algorithm and another that employs DL techinques for embedding creation.

## Data

The instructions on how to get the data used for this project can be found in this [Github Repository](https://github.com/MTG/da-tacos). It contains two subsets, namely the benchmark subset, and the cover analysis subset with pre-extracted features and metadata for 15,000 and 10,000 songs, respectively.

For the creation of the datasets, certain aggregations were made. The pre-extrcacted features of each track were in the form of a sequance. Generally, when dealing with audio the data comes as a sequence of values with size equal to the length of the audio in seconds times the sample rate. This results in a huge array making it difficult to handle, so a `time window` and a `hop length` are defined. This way certain features are computed at each `time window` which moves `hop length` time steps at a time.

In the end, the techniques that were used do not handle sequence data. So two tabular datasets were created that are the results of aggregations on the sequences of each features discussed before. The script can be found in the [build_tabular](https://github.com/alexliap/audio_embeddings/blob/master/cli/build_tabular.py) script.

## Clustering

In order to have a complete picture on te complexity of the problem, we first employ a traditional machine learning algorithm that is popular for clustering scenarios, K-Means. This algorithm is relatively simple and utilizes Euclidean distance to calculate the centroids' coordinates, but it has one caveat, finding the optimal number of clusters.

If the cliques of the cover analysis subset were perfectly separable, then we would get that the optimal number of clusters is 5000 (number of cliques in the subset). Instead, we see that while we increase the number of clusters the clustering is poorer. For more details check the relevant [notebook](https://github.com/alexliap/audio_embeddings/blob/master/ml_clustering.ipynb).

All in all, the problem seems to be way more complex for a simple clustering algorithm, like K-Means, to solve. For that reason, we move on to a more sophisticated solution.

## Embedding Creation

A more advanced approach to capturing similarity between vectors involves generating embeddings from the original vector. This requires constructing an Autoencoder deep learning model that learns to reconstruct the original vector after encoding it into a latent representation.

The source code for making the Autoencoder is located [here](https://github.com/alexliap/audio_embeddings/tree/master/src/audio_embeddings). The model trained for this experiment was a 207K parameter Autoencoder, with layer sizes:

- Encoder: 52 - 300 - 200 - 100 - 50 - 20
- Decoder: 20 - 50 - 100 - 200 - 300 - 52

where 52 are the features of each performance/track. The goal is to learn a latent representation of the tracks and use those for comparison.

Other important configuration decisions:

- Loss: MSE loss to measure the reconstruction of each vector
- Optimizer: Adam

### Evaluation

Training was performed on the cover analysis subset and validation on the benchmark subset. This problem falls under the unsupervised domain, making the evaluation of the solution difficult.

The evaluation method we propose is to measure the average distance of a clique's performances with each other and compare it against the average distance of a clique's performances with the rest of the data. The distance used is the Euclidean distance.

The resulting distributions can be seen [here](https://github.com/alexliap/audio_embeddings/blob/master/pics/in_clique_vs_out_clique_dists.jpeg). We can see that the two distributions have quite a big overlap. There are various reasons that might lead to this.

- The tracks are truly similar making it difficult to separate them.
- The original vector that represents each track is the result of some aggregations, therefore a lot of important information is gone.
- The model is not quite powerful.

![in_clique_vs_out_clique_dists](https://github.com/alexliap/audio_embeddings/blob/master/pics/in_clique_vs_out_clique_dists.jpeg)

An alternative solution, is to develop an Autoencoder CNN model that tries to reconstruct the track's spectrogram. In deep learning, when it comes to audio, usually it is a standard practice to use CNNs.

### Train your own Autoencoder

In order to train your own Autoencoder you have to:

- Clone the repository

```bash
git clone https://github.com/alexliap/audio_embeddings.git
```

- Create a virtual environment using Python 3.12 and install the dependencies with

```bash
pip install .
```

- You can run the script `cli/train.py` that uses some default parameteres for the whole process or change them yourself.

```bash
python cli/train.py
```

At the end of the training, the Autoencoder model will be saved at the directory `lightning_logs/model/epoch=X-step=Y.ckpt`.

The model can be loaded by running

```python
from audio_embeddings import AutoEncoder, AutoEncoderModel, Decoder, Encoder

layer_sizes = <layer_list_used_for_trainings>

enc = Encoder(layer_sizes)
dec = Decoder(layer_sizes[::-1])

module = AutoEncoder(encoder=enc, decoder=dec)

model = AutoEncoderModel(autoencoder=module)

model.load_state_dict(
    state_dict=torch.load(
        "lightning_logs/model/epoch=X-step=Y.ckpt", weights_only=True
    )["state_dict"]
)
```

### Compute Distance

In order to compute the distance between 2 feature vectors follow the instruction below.

- Having trained your model you must load it like above.

- Load also 2 feature vectors from the benchmark dataset. You can also load more than 1 for the comparison

```python
a = bench.sort('work').drop(['work', 'performance']).to_numpy().astype(np.float32)[1, :].reshape(1, -1)
b = bench.sort('work').drop(['work', 'performance']).to_numpy().astype(np.float32)[400, :].reshape(1, -1)

# or
b = bench.sort('work').drop(['work', 'performance']).to_numpy().astype(np.float32)[400:800, :]
```

- Get the euclidean distance between the 2 or more vectors

```python
model.get_distance(a, b)
```
