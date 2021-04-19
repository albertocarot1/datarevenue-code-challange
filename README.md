# Code challenge Data Revenue


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

This will download the data, preprocess it, train the model, and create the evaluation report in TODO. 

