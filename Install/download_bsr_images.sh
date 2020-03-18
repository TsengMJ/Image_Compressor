#!/bin/bash
DataPath="../Data"

## Download BSR-dataset
mkdir -p $DataPath
wget -nc -P $DataPath http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
tar -zxvf $DataPath/BSR_bsds500.tgz -C $DataPath