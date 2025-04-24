#!/usr/bin/env python
# coding: utf-8

# Copyright 2021, IBM Corporation. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Python function to customize json sentences locally.
"""

__author__ = "Vagner Santana, Melina Alberio, Cassia Sanctos and Tiago Machado"
__copyright__ = "IBM Corporation 2024"
__credits__ = ["Vagner Santana, Melina Alberio, Cassia Sanctos, Tiago Machado"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"

import os
import json
import pandas as pd
import numpy as np
import customize_helper

from pathlib import Path

# Sentence transformer model HF
model_path = Path('models/all-MiniLM-L6-v2')
model_id = model_path.name

supported_languages = ["pl",]
chosen_language = "pl"
if chosen_language not in supported_languages:
    chosen_language = ""
else:
    chosen_language = f"_{chosen_language}"

# INPUT FILE
# Default file with empty embeddings
prompt_sentences = Path("prompt-sentences-main")
json_in_file = prompt_sentences / f"prompt_sentences{chosen_language}.json"
json_in_file_name = json_in_file.stem

# OUTPUT FILE
json_out_file_name = f'{json_in_file_name}-{model_id}.json'

prompt_json = json.load(open(json_in_file))
prompt_json_embeddings = customize_helper.populate_embeddings(prompt_json, model_path)
prompt_json_centroids = customize_helper.populate_centroids(prompt_json_embeddings)
customize_helper.save_json(prompt_json_centroids, json_out_file_name)