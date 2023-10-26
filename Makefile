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

## Relink all dvc files
relink:
	# unprotect
	dvc unprotect data
	dvc unprotect models
	dvc unprotect reports
	# add
	dvc add data
	dvc add models
	dvc add reports
