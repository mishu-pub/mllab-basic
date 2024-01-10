FROM jupyter/datascience-notebook

ADD jupyter_notebook_config.py /home/jovyan/.jupyter/jupyter_notebook_config.py

RUN pip install --upgrade pip
RUN pip install jupytext
RUN pip install ua_parser
RUN pip install xgboost
RUN pip install hyperopt
RUN pip install mlflow

RUN jupyter lab build
RUN jupyter lab clean
RUN jupyter labextension enable jupyterlab-jupytext


EXPOSE 8888 5000 8080
