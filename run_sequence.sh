#!/usr/bin/env bash

MEMORY="-Xmx4g"

CP="./config/:./target/classes/:./target/dependency/*"

OPTIONS="$MEMORY -Xss40m -ea -cp $CP"
PACKAGE_PREFIX="latentperceptron"
MAIN="$PACKAGE_PREFIX.MainClass"
time nice java $OPTIONS $MAIN $CONFIG_STR $*

