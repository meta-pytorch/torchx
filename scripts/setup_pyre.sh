#!/bin/bash

set -ex

VERSION=$(grep "version" .pyre_configuration | sed -n -e 's/.*\(0\.0\.[0-9]*\).*/\1/p')
pip install pyre-check-nightly==$VERSION
