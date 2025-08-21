import os
import argparse
from pathlib import Path
from itertools import groupby
import grpc
import riva.client
import yaml

CONFIG_PATH="stt_config.yaml"

with open(CONFIG_PATH) as f:
    args = yaml.safe_load(f)

print(args)

class ParakeetSTTClient:
    def __init__(
            self,
            input_file: str
            ):
        self.input_file = input_file
        self.args = args

        self._process_args()
    
    def _process_args(self):
        if (self.input_file is None and not self.args["list_models"]) or \
            (self.input_file is not None and self.args["list_models"]):
            raise ValueError("You must specify exactly only one of 'input-file' or 'list-models'")
        
        if self.input_file:
            self.input_file.expanduser()
    

        
