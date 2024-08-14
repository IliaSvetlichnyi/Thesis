import argparse

import sys

import pandas as pd
import numpy as np
import json
from sklearn.decomposition import PCA
from loguru import logger

from graph import Step
from language_modeling import OpenAiLlamaApi, LlamaModel, PromptGenerator
from code_generation import ValidationCodeGenerator, MainCodeGenerator
from orchestrator import Orchestrator
from utils import get_dataset_info
from pathlib import Path
from utils import deserialize_steps

# Configure logger
logger.add("execution.log", rotation="500 MB")

EXAMPLE_STEP_SCRIPT = """
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler

def step_40(Segments_normalized, Dec_levels):
    Features = []
    for segment in Segments_normalized:
        coeffs = pywt.wavedec(segment, 'db4', level=Dec_levels)
        features = [coefficient.mean() for coefficient in coeffs]
        Features.append(features)
    return StandardScaler().fit_transform(Features)
"""

MODEL_TAG = "meta-llama/llama-3-70b-instruct"
PYTHON_PATH = sys.executable

def dataset_info_from_path(path):
    csv_path = str(Path(path).resolve())
    raw_data = pd.read_csv(csv_path)
    return get_dataset_info(raw_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--steps_file")
    parser.add_argument("--parameters_file")
    parser.add_argument("--output_dir", "-o", default='out')

    args = parser.parse_args()
    
    dataset_info = dataset_info_from_path(args.dataset)
    steps = deserialize_steps(args.steps_file)
    with open(args.parameters_file, 'r') as f:
        parameters = json.load(f)

    with open('env.json', 'r') as f:
        credentials_dict = json.load(f)

    API_URL = "https://openrouter.ai/api/v1"
    API_KEY = credentials_dict["OPENROUTER_API_KEY"]

    llama_api = OpenAiLlamaApi(API_URL, API_KEY, MODEL_TAG)
    model = LlamaModel(llama_api)
    validation_code_genrator = ValidationCodeGenerator()

    main_code_generator = MainCodeGenerator(additional_lines=[])
    prompt_generator = PromptGenerator(EXAMPLE_STEP_SCRIPT, dataset_info)

    orchestrator = Orchestrator(
        model,
        prompt_generator,
        validation_code_genrator,
        main_code_generator,
        args.output_dir,
        PYTHON_PATH
    )

    orchestrator.run_steps(steps, parameters)

if __name__ == '__main__':
    main()