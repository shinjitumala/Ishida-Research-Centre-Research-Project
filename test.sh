#!/bin/bash

./vfilter.sh raw/$1.txt > filtered/$1.txt
./avg.sh filtered/$1.txt > results/$1.txt

