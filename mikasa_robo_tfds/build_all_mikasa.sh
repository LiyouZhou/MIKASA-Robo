#!/bin/bash

# This script builds all Mikasa datasets.

set -e
set -x

datasets=(
    'ShellGameTouch-v0'
    'ShellGamePush-v0'
    'ShellGamePick-v0'
    'InterceptSlow-v0'
    'InterceptMedium-v0'
    'InterceptFast-v0'
    'InterceptGrabSlow-v0'
    'InterceptGrabMedium-v0'
    'InterceptGrabFast-v0'
    'RotateLenientPos-v0'
    'RotateLenientPosNeg-v0'
    'RotateStrictPos-v0'
    'RotateStrictPosNeg-v0'
    'TakeItBack-v0'
    'RememberColor5-v0'
    'RememberShape3-v0'
    'RememberShape5-v0'
    'RememberShape9-v0'
    'RememberShapeAndColor3x2-v0'
    'RememberShapeAndColor3x3-v0'
    'RememberShapeAndColor5x3-v0'
    'BunchOfColors3-v0'
    'BunchOfColors5-v0'
    'BunchOfColors7-v0'
    'SeqOfColors3-v0'
    'SeqOfColors5-v0'
    'SeqOfColors7-v0'
    'ChainOfColors3-v0'
)

# Build each dataset
for dataset in ${datasets[@]}; do
    if [ -d "/home/liyouzhou/tensorflow_datasets/mikasa_robo_tfds/${dataset}" ]; then
        echo "Directory /home/liyouzhou/tensorflow_datasets/mikasa_robo_tfds/${dataset} already exist. Skipping ${dataset}."
        continue
    fi

    echo "Building ${dataset} dataset..."
    cd "/home/liyouzhou/study/MIKASA-Robo/mikasa_robo_tfds"
    tfds build --config "${dataset}"
    rm -rf /home/liyouzhou/tensorflow_datasets/downloads/*
    ls /home/liyouzhou/tensorflow_datasets/*
    cd "/home/liyouzhou/tensorflow_datasets/mikasa_robo_tfds/"
    git add "${dataset}"
    git commit -m "Add ${dataset} dataset"
    git push
    rm -rf /home/liyouzhou/tensorflow_datasets/mikasa_robo_tfds/${dataset}/1.0.0/*tfrecord*
    echo "Built and pushed ${dataset} dataset."
done
