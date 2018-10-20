#!/bin/sh

grep -e "====" -e "validation_score:" -e "momentum =" -e "batch_size =" -e "learning_rate =" -e "decay =" -e "nb_epoch =" -e "nesterov =" -e "dropouts =" -e "nb_layers =" -e "batchnorm =" -e "layer_sizes =" -e "weight-init-stddevs =" -e "bias_init_conts =" -e "penalty ="  -e "parameter =" $1
