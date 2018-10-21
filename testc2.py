import os
import deepchem as dc


current_dir = os.path.dirname(os.path.realpath("__file__"))
dataset_file = "medium_muv.csv.gz"
full_dataset_file = "muv.csv.gz"

# We use a small version of MUV to make online rendering of notebooks easy. Replace with full_dataset_file
# In order to run the full version of this notebook
dc.utils.download_url("https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/%s" % dataset_file,
                      current_dir)

dataset = dc.utils.save.load_from_disk(dataset_file)
print("Columns of dataset: %s" % str(dataset.columns.values))
print("Number of examples in dataset: %s" % str(dataset.shape[0]))

from rdkit import Chem
from rdkit.Chem import Draw
from itertools import islice
from IPython.display import Image, display, HTML

def display_images(filenames):
    """Helper to pretty-print images."""
    for filename in filenames:
        display(Image(filename))

def mols_to_pngs(mols, basename="test"):
    """Helper to write RDKit mols to png files."""
    filenames = []
    for i, mol in enumerate(mols):
        filename = "MUV_%s%d.png" % (basename, i)
        Draw.MolToFile(mol, filename)
        filenames.append(filename)
    return filenames

num_to_display = 12
molecules = []
for _, data in islice(dataset.iterrows(), num_to_display):
    molecules.append(Chem.MolFromSmiles(data["smiles"]))
display_images(mols_to_pngs(molecules))

MUV_tasks = ['MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644',
             'MUV-548', 'MUV-852', 'MUV-600', 'MUV-810', 'MUV-712',
             'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733', 'MUV-652',
             'MUV-466', 'MUV-832']

featurizer = dc.feat.CircularFingerprint(size=1024)
loader = dc.data.CSVLoader(
      tasks=MUV_tasks, smiles_field="smiles",
      featurizer=featurizer)
dataset = loader.featurize(dataset_file)

splitter = dc.splits.RandomSplitter(dataset_file)
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
    dataset)
#NOTE THE RENAMING:
valid_dataset, test_dataset = test_dataset, valid_dataset

import numpy as np
import numpy.random

print('===============================================')
print('parameter = custom')
params_dict = {"activation": ["relu"],
               "momentum": [0.8],
               "batch_size": [30],
               "init": ["glorot_uniform"],
               "data_shape": [train_dataset.get_data_shape()],
               "learning_rate": [1e-4],
               "decay": [1e-9],
               "nb_epoch": [10],
               "nesterov": [True],
               "dropouts": [(.1, .1, .1, .1)],
               "nb_layers": [4],
               "batchnorm": [False],
               "layer_sizes": [(1000, 1000, 1000, 1000, )],
               "weight_init_stddevs": [(.2, .2, .2, .2,)],
               "bias_init_consts": [(2., 2., 2., 2.,)],
               "penalty": [0.],
              }

i = 0
while i < 10:
    n_features = train_dataset.get_data_shape()[0]
    def model_builder(model_params, model_dir):
      model = dc.models.MultitaskClassifier(
        len(MUV_tasks), n_features, **model_params)
      return model

    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
    optimizer = dc.hyper.HyperparamOpt(model_builder)
    best_dnn, best_hyperparams, all_results = optimizer.hyperparam_search(
        params_dict, train_dataset, valid_dataset, [], metric)
    i += 1

print('===============================================')
