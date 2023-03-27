import os

from ..utils.general import get_base_dir

# RACE-C global variables 
RACE_C_REPO = 'https://github.com/mrcdata/race-c/raw/master/data.zip'
RACE_C_DIR = os.path.join(get_base_dir(), 'data', 'RACE-C')

# Reclor global variables 
RECLOR_REPO = 'https://github.com/yuweihao/reclor/releases/download/v1/reclor_data.zip'
RECLOR_DIR = os.path.join(get_base_dir(), 'data', 'ReClor')

#== Automatic downloading utils ===================================================================#
def download_race_plus_plus():
    """ automatically downloads the classification CAD datasets in parent data folder"""
    os.system(f"mkdir -p {RACE_C_DIR}")
    os.system(f"wget -O RACE-C.zip {RACE_C_REPO}")
    os.system(f"unzip RACE-C.zip -d {RACE_C_DIR}")
    os.system(f"mv {RACE_C_DIR}/data/* {RACE_C_DIR}")
    os.system(f"rmdir {RACE_C_DIR}/data")
    os.system(f"rm RACE-C.zip")

def download_reclor():
    """ automatically downloads the classification CAD datasets in parent data folder"""
    os.system(f"mkdir -p {RECLOR_DIR}")
    os.system(f"wget {RECLOR_REPO}")
    os.system(f"unzip -P for_non-commercial_research_purpose_only reclor_data.zip -d {RECLOR_DIR}")
    os.system(f"rm reclor_data.zip")
