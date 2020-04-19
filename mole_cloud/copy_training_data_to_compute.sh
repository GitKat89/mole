#!/bin/bash

gsutil cp gs://mole_data/upload/ISIC_2019_Training_Input.zip ~
mkdir unzipped
unzip ISIC_2019_Training_Input.zip -d unzipped/
