#!/bin/bash
sudo apt update
sudo apt upgrade
sudo apt install \
	rsync \
	htop \
	git \
	neovim \
	python3.14-venv \
	python3-pip

export MLFLOW_TRACKING_URI='http://0.0.0.0:5000'