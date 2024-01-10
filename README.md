
# Operational

We use jupytext, so that no .ipynb is submitted, but the notebooks are generated out of the .py files

## Docker compose

Run this first:
```
USER="$(id -u)" GROUP="$(id -g)" docker-compose build

```

## Mlflow

To start mlflow 

```
USER="$(id -u)" GROUP="$(id -g)" docker-compose up mlflow
```

You will then be able to see the mlflow ui [HERE](http://localhost:5000/)

If you want another port you have to edit the docker files

# Jupyter

To start jupyter either
```
USER="$(id -u)" GROUP="$(id -g)" docker-compose up jupyter
```

or

```
docker build -t .

docker run -v $PWD:/home/jovyan/work -i -t -p 8887:8888 --user root  -e JUPYTER_ENABLE_LAB=yes -e CHOWN_HOME=yes -e CHOWN_HOME_OPTS='-R' -e NB_UID="$(id -u)" -e NB_GID="$(id -g)"  -w /home/jovyan jupyter-poc
```

You will then be able to see the jupyterlab environment [HERE](http://localhost:8887/)

If you want another port you have to edit the docker files

# Resources

## Jupyter image

https://jupyter-docker-stacks.readthedocs.io/en/latest/using/troubleshooting.html

## SSH forwarding

https://docs.github.com/en/authentication/connecting-to-github-with-ssh/using-ssh-agent-forwarding

Basically do the following:

```
ssh-agent
ssh-add -L
ssh-add --apple-use-keychain  ~/.ssh/id_rsa
ssh -A <SOMEWHERE>

```

# MLFOw
https://docs.databricks.com/en/_extras/notebooks/source/hyperopt-sklearn-model-selection.html

https://dzlab.github.io/ml/2020/08/16/mlflow-hyperopt/

https://towardsdatascience.com/automate-hyperparameter-tuning-with-hyperopts-for-multiple-models-22b499298a8a



# Some learnings in the process:

- mlflow ui needs the same paths as the paths where the experiments were ran
- fiddling with users is hard
