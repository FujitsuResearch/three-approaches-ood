#!/bin/bash

# pick one from ["mnist","ilab","daiso","carcgs"]
data="mnist"
# epoch size
epoch=100
# set numerator of InD data diversity values
seenrate=4
# learning rate
lr=0.0001
# BN parameter beta
beta=0.99
# weight value of the invariance loss term lambda
rw=0.01
# interval of pairing
pi=20
# trial number
trial=1

# -tv tune and -sd 0 : for parameter search
# -tv validation and -sd 1 : for test

# -rb ** : you can change the number of batchsize depending on the hardware limitation
# -bnm ** : you can change the BN parameter beta

python -u main.py -s random -tv validate -sd 1 \
       -d ${data[(${SGE_TASK_ID}-1)/(3*5)]} \
       -e ${epoch} \
       -r ${seenrate} \
       -lr ${lr} \
       -rw ${rw} \
       -pi ${pi} \
       -t ${trial}
