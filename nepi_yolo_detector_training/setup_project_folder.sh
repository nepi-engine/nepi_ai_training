#!/bin/bash
##
## Copyright (c) 2024 Numurus, LLC <https://www.numurus.com>.
##
## This file is part of nepi-engine
## (see https://github.com/nepi-engine).
##
## License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
##

#######################################################################################################

#######################################################################################################
sudo -v

if [[ -n "$1" ]]; then
  PROJECT_NAME="$1"
else
  PROJECT_NAME=yolov8_train_project
fi

PROJECT_FOLDER="$(dirname $(dirname $(pwd)))/${PROJECT_NAME}"

if [[ -d "$PROJECT_FOLDER" ]]; then
  echo "Project folder allready exists at ${PROJECT_FOLDER}"
else
  sudo mkdir $PROJECT_FOLDER
  sudo cp -r $(pwd)/* ${PROJECT_FOLDER}/
  sudo cp -r $(dirname $(pwd))/src/nepi_ai_training/* ${PROJECT_FOLDER}/
  sudo chown -R ${USER}:${USER} $PROJECT_FOLDER
  sudo chmod -R +x $PROJECT_FOLDER
  echo "Project folder created at ${PROJECT_FOLDER}"
 fi
