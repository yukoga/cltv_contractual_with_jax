#!/bin/bash

export PIPENV_VENV_IN_PROJECT=1
pipenv install
pipenv install --dev
echo 'export PYTHONPATH=$PYTHONPATH:${PWD}' >> .env
