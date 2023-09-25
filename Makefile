#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Requirements
reqs:
	pip install -r requirements.txt

## Upload Data to default DVC remote
push:
	dvc push

## Download Data from default DVC remote
pull:
	dvc pull

## Reproduce the DVC pipeline - recompute any modified outputs such as processed data or trained models
repro:
	dvc repro

## Force Reproduce all the Stages on the DVC pipeline
force-repro:
	dvc repro -sf
