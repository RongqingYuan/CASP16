# use packages to deal with extentions
import os
import sys
import numpy as np

monomer_path = "/data/data1/conglab/qcong/CASP16/monomers/"
monomer_list = [txt for txt in os.listdir(
    monomer_path) if txt.endswith(".txt")]
