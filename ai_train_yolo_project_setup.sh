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
  PROJECT_NAME=yolo_train_project
fi

export PROJECT_FOLDER="$(dirname $(pwd))/${PROJECT_NAME}"

if [[ -d "$PROJECT_FOLDER" ]]; then
  echo "Project folder allready exists at ${PROJECT_FOLDER}, will update process scripts"
  RSYNC_EXCLUDES=" --exclude project_settings.yaml --exclude classes.txt --exclude model_info.yaml"
  echo "Excluding ${RSYNC_EXCLUDES}"
  rsync -a -f ${RSYNC_EXCLUDES} $(pwd)/nepi_yolo_detector_training/* ${PROJECT_FOLDER}/
  sudo cp -r $(pwd)/TRAIN_CUSTOM_YOLO_AI_MODEL.md ${PROJECT_FOLDER}/
  sudo cp -r $(pwd)/src/nepi_ai_training/* ${PROJECT_FOLDER}/

else
  sudo mkdir $PROJECT_FOLDER
  sudo cp -r $(pwd)/nepi_yolo_detector_training/* ${PROJECT_FOLDER}/
  sudo cp -r $(pwd)/TRAIN_CUSTOM_YOLO_AI_MODEL.md ${PROJECT_FOLDER}/
  sudo cp -r $(pwd)/src/nepi_ai_training/* ${PROJECT_FOLDER}/
  echo "Project folder created at ${PROJECT_FOLDER}"
 fi

if [[ -d "$PROJECT_FOLDER" ]]; then
  sudo chown -R ${USER}:${USER} $PROJECT_FOLDER
  sudo chmod -R +x $PROJECT_FOLDER
  echo "Changing directory to proect folder ${PROJECT_FOLDER}"
  cd ${PROJECT_FOLDER}
  ls
fi
