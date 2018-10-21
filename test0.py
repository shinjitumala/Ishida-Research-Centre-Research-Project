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



data = [["momentum", 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3, 4, 5],
	["batch_size", 50, 10, 50, 100, 300, 500, 800, 1000, 2000, 3000, 4000],
	["learning_rate", 1e-3, 1e-10, 1e-8, 1e-5, 1e-3, 1e-2, 1e-1],
	["decay", 1e-6, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 0],
	["nb_epoch", 1, 1, 2, 3, 4, 5],
	["nesterov", "False", "False", "True"],
	["dropouts", (0.5,), (0.0,), (0.1,), (0.3,), (0.5,), (0.7,), (0.9,), (1.0,)],
	["nb_layers", 1,  1, 2, 3, 4, 5],
	["batchnorm", "False", "False", "True"],
	["layer_sizes", (1000,), (100,), (500,), (1000,), (2000,)],
    ["weight-init-stddevs", (.1,), (.2,), (.3,), (.5,), (.7,), (.8,), (.9,), (1.0,)],
    ["bias_init_conts", (1.,), (.3,), (.5,), (1.,), (2.,), (5.,)],
    ["penalty", 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
]
state = [1 for x in range(len(data))]
i = 0
while i < len(data):
	m = 2
	state = [1 for x in range(len(data))]
	while m < len(data[i]):
		print('===============================================')
		print('{0} = {1}'.format(data[i][0], data[i][m]))
		state[i] = m;
		k = 0;
		while k < 10:
			params_dict = {"activation": ["relu"],
				       "momentum": [data[0][state[0]]],
				       "batch_size": [data[1][state[1]]],
				       "init": ["glorot_uniform"],
				       "data_shape": [train_dataset.get_data_shape()],
				       "learning_rate": [data[2][state[2]]],
				       "decay": [data[3][state[3]]],
				       "nb_epoch": [data[4][state[4]]],
				       "nesterov": [data[5][state[5]]],
				       "dropouts": [data[6][state[6]]],
				       "nb_layers": [data[7][state[7]]],
				       "batchnorm": [data[8][state[8]]],
				       "layer_sizes": [data[9][state[9]]],
				       "weight_init_stddevs": [data[10][state[10]]],
				       "bias_init_consts": [data[11][state[11]]],
				       "penalty": [data[12][state[12]]],
				      }


			n_features = train_dataset.get_data_shape()[0]
			def model_builder(model_params, model_dir):
			  model = dc.models.MultitaskClassifier(
			    len(MUV_tasks), n_features, **model_params)
			  return model

			metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
			optimizer = dc.hyper.HyperparamOpt(model_builder)
			best_dnn, best_hyperparams, all_results = optimizer.hyperparam_search(
			    params_dict, train_dataset, valid_dataset, [], metric)
			k += 1
		m += 1
	i += 1
