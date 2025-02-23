import sys
sys.path.insert(0,"/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/")
from pathlib import Path

import yaml

with open("/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR,) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]
