#!/bin/bash
DataPath="../Data"

## Download open-images-dataset
mkdir -p $DataPath
aws s3 --no-sign-request sync s3://open-images-dataset/train $DataPath/OpenImages/Train
aws s3 --no-sign-request sync s3://open-images-dataset/validation $DataPath/OpenImages/Validation
aws s3 --no-sign-request sync s3://open-images-dataset/test $DataPath/OpenImages/Test