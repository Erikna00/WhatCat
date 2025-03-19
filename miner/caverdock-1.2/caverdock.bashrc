#!/bin/bash

export CD_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export PATH=$CD_HOME/bin/:$PATH
