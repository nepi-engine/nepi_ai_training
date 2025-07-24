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
PROJECT_NAME="LaserLines"
#######################################################################################################
PROJECT_FOLDER="./../../${PROJECT_NAME}"

# Avoid pushing local build artifacts, git stuff, and a bunch of huge GPSD stuff
RSYNC_EXCLUDES=" --exclude README* --exclude LICENSE* "


echo "Excluding ${RSYNC_EXCLUDES}"

# Push everything in the current folder but the EXCLUDES to the specified project folder on the target
rsync -avzhe ${RSYNC_EXCLUDES} ./ ${PROJECT_FOLDER}

# Push the nepi_ai_train sdk file
rsync -avzhe ${RSYNC_EXCLUDES} ./../src/nepi_ai_training/ ${PROJECT_FOLDER}

