#!/bin/bash

# Note: you should have run build-model to gather the data for this
# (it also builds a model, which we don't really use here)

pushd $(dirname $0) > /dev/null
SCRIPT_DIR=$PWD
popd > /dev/null

set -e

#test cross-validation avec attachement et relations
#
DATA_DIR=$SCRIPT_DIR/../../data/SNAPSHOTS/latest
DATA_EXT=.csv # .2.csv
DECODE_FLAGS="-C $SCRIPT_DIR/stac-features.config"
DECODER=attelo

if [ ! -d "$DATA_DIR" ]; then
    echo >&2 "No data to run experiments on"
    echo >&2 "Please run $SCRIPT_DIR/gather-features"
    exit 1
fi


T=$(mktemp -d -t stac.XXXX)
cd $T

# NB: use a colon if you want a separate learner for relations
LEARNERS="bayes maxent"
DECODERS="local mst locallyGreedy"
DATASETS="all pilot socl-season1"

for dataset in $DATASETS; do
    # try x-fold validation with various algos
    touch scores-$dataset
    for learner in $LEARNERS; do
        for decoder in $DECODERS; do
            LEARNER_FLAGS="-l"$(echo $learner | sed -e 's/:/ --relation-learner /')
            $DECODER evaluate $DECODE_FLAGS\
                $DATA_DIR/$dataset.edu-pairs$DATA_EXT\
                $DATA_DIR/$dataset.relations$DATA_EXT\
                $LEARNER_FLAGS\
                -d $decoder >> $EVAL_DIR/scores-$dataset
        done
    done

    # test stand-alone parser for stac
    # 1) train and save attachment model
    # -i
    $DECODER learn $DECODE_FLAGS $DATA_DIR/$dataset.edu-pairs$DATA_EXT -l bayes
    mv attach.model $dataset.attach.model

    # 2) predict attachment (same instances here, but should be sth else) 
    # NB: updated astar decoder seems to fail / TODO: check with the real subdoc id
    # -i
    $DECODER decode $DECODE_FLAGS -A $dataset.attach.model -o tmp\
        $DATA_DIR/$dataset.edu-pairs$DATA_EXT -d mst

    # attach + relations: TODO: relation file is not generated properly yet
    # 1b) train + save attachemtn+relations models
    $DECODER learn $DECODE_FLAGS\
        $DATA_DIR/$dataset.edu-pairs$DATA_EXT\
        $DATA_DIR/$dataset.relations$DATA_EXT\
        -l bayes
    mv attach.model    $dataset.attach.model
    mv relations.model $dataset.relations.model

    # 2b) predict attachment + relations
    # -i
    $DECODER decode $DECODE_FLAGS -A $dataset.attach.model -R $dataset.relations.model -o tmp/\
        $DATA_DIR/$dataset.edu-pairs$DATA_EXT\
        $DATA_DIR/$dataset.relations$DATA_EXT\
        -d mst
done
echo $T >&2

# results
#socl
#FINAL EVAL: relations full: 	 locallyGreedy+bayes, h=average, unlabelled=False,post=False,rfc=full 	 Prec=0.229, Recall=0.217, F1=0.223 +/- 0.015 (0.239 +- 0.029)
#FINAL EVAL: relations full: 	 local+maxent, h=average, unlabelled=False,post=False,rfc=full 	         Prec=0.678, Recall=0.151, F1=0.247 +/- 0.017 (0.243 +- 0.034)
#FINAL EVAL: relations full: 	 local+bayes, h=average, unlabelled=False,post=False,rfc=full 	                 Prec=0.261, Recall=0.249, F1=0.255 +/- 0.015 (0.264 +- 0.031)
#FINAL EVAL: relations full: 	 locallyGreedy+maxent, h=average, unlabelled=False,post=False,rfc=full 	 Prec=0.281, Recall=0.257, F1=0.269 +/- 0.015 (0.277 +- 0.030)

#pilot
#FINAL EVAL: relations full  : 	 locallyGreedy+maxent, h=average, unlabelled=False,post=False,rfc=full 	 Prec=0.341, Recall=0.244, F1=0.284 +/- 0.015 (0.279 +- 0.029)
