# Code challenge Data Revenue

In this project an automated pipeline for the creation and evaluation of a machine
learning model is created. The problem solved is the prediction of the score of a wine.
The pipeline is divided into four steps, concatenated sequentially through the library `luigi`:
1. DownloadData
2. Make(Train|Test)Dataset
3. TrainModel
4. EvaluateModel

Each step has its own folder, Dockerfile and poetry environment.

## Local environment

Although the project runs on multiple docker images, I created a poetry environment to run everything locally
(like the `DataExploration.ipynb` notebook).

This can be installed through the command

    poetry install

In order to access the notebook, open a jupyter lab inside the created virtual environment.
Start it with the command

    poetry shell

and then
    
    jupyter lab

so that the virtualenv's kernel is available in jupyter.

## Dockerized environment

To execute the pipeline, first build the containers through

    ./build-task-images.sh 0.1

Now to execute the pipeline simply run:

    docker-compose up orchestrator

This will download the data, preprocess it, train the model, and create the evaluation report the folder
report, under data_root. 

