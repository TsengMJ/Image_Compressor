#!/bin/bash
DataPath="../Data"

## Download DIV2K dataset
mkdir -p $DataPath
wget -nc -P $DataPath http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip $DataPath/DIV2K_train_HR.zip -d $DataPath
