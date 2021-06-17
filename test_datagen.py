#!/usr/bin/env python3

# This file shows how to use the "GeneratedDataset" class to populate a folder
# with some synthetic data.
import ClassFiles.GeneratedDataset as dat

dat.generate_data(10, "./data/train/", append=False)
