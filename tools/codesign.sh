#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0) && pwd)

codesign -s - -v -f --entitlements ${SCRIPT_DIR}/debug.plist ${SCRIPT_DIR}/../out/bin/cutenn.app
