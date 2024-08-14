"""
This module contains the Orchestrator class and related utilities for managing
the execution of data processing steps in a pipeline.
"""

import traceback
import subprocess
import os
import pandas as pd
from loguru import logger
import numpy as np
from sklearn.decomposition import PCA

from typing import List, Dict
from language_modeling import Model, PromptGenerator
from code_generation import CodeGenerator
from graph import Step

def topological_sort(steps: List[Step]) -> List[Step]:
    """
    Perform a topological sort on the given list of Steps.

    Args:
        steps (List[Step]): List of Step objects to sort.

    Returns:
        List[Step]: Sorted list of Step objects.
    """
    used = set()
    order = []
    id_to_step = {step.step_id: step for step in steps}
    def dfs(step):
        nonlocal order
        nonlocal used
        nonlocal id_to_step
        if step.step_id in used:
            return
        used.add(step.step_id)
        for dep_id in step.dependencies:
            dfs(id_to_step[dep_id])
        order.append(step)
    for step in steps:
        dfs(step)
    return order

class Orchestrator:
    """
    Orchestrates the execution of data processing steps in a pipeline.

    This class manages the generation, validation, and execution of code for each step
    in the data processing pipeline.
    """

    MAX_RETRIES = 5

    def __init__(
          self,
          model: Model,
          prompt_generator: PromptGenerator,
          validation_generator: CodeGenerator,
          main_generator: CodeGenerator,
          working_dir: str,
          python_path: str,
        ):
        """
        Initialize the Orchestrator.

        Args:
            model (Model): The language model used for code generation.
            prompt_generator (PromptGenerator): Generator for creating prompts.
            validation_generator (CodeGenerator): Generator for validation code.
            main_generator (CodeGenerator): Generator for main execution code.
            working_dir (str): Directory for storing generated files.
            python_path (str): Path to the Python interpreter.
        """
        self.model = model
        self.prompt_generator = prompt_generator
        self.validation_generator = validation_generator
        self.main_generator = main_generator
        self.working_dir = working_dir
        self.python_path = python_path

    def run_steps(self, steps: List[Step], parameters):
        """
        Run the processing steps in the correct order.

        Args:
            steps (List[Step]): List of Step objects to execute.
            parameters (dict): Parameters for the execution.
        """
        os.makedirs(self.working_dir, exist_ok=True)

        id_to_step = {step.step_id: step for step in steps}
        steps_ordered = topological_sort(steps)

        def find_deps_dfs(step, deps_set):
            nonlocal id_to_step
            if step.step_id in deps_set:
                return
            for dep_step_id in step.dependencies:
                find_deps_dfs(id_to_step[dep_step_id], deps_set)
            deps_set.add(step.step_id)

        def find_deps(step):
            nonlocal steps_ordered

            dependencies = set()
            find_deps_dfs(step, dependencies)
            dependencies.remove(step.step_id)
            return [step for step in steps_ordered if step.step_id in dependencies]

        for step in steps_ordered:
            try:
                code_snippet, _ = self.generate_step_file(step, parameters)
                dependencies = find_deps(step)

                _, validation_filename = self.generate_validation_file(step, dependencies, parameters)
                success, message = self.validate_unit_code(validation_filename)

                for _ in range(Orchestrator.MAX_RETRIES):
                    if success:
                        break
                    code_snippet, _ = self.fix_step_source(step, code_snippet, message)
                    success, message = self.validate_unit_code(validation_filename)
                if not success:
                    continue
            except Exception as e:
                logger.error(f"Error processing step {step.step_id}: {e}\n{traceback.format_exc()}\n")
                break
        
        _, main_filename = self.generate_main_file(steps[-1], steps_ordered, parameters)
        success, message = self.validate_unit_code(main_filename)
        if success:
            logger.info('code generated successfully!')
        else:
            logger.info(f'code generation failed:\n{message}')

    def generate_step_file(self, step, parameters):
        """
        Generate a Python file for a single processing step.

        Args:
            step (Step): The Step object to generate code for.
            parameters (dict): Parameters for code generation.

        Returns:
            tuple: A tuple containing the generated code snippet and the filename.
        """
        prompt = self.prompt_generator.generate(step, parameters)
        code_snippet = self.model.predict(prompt)
        filename = f'step_{step.step_id}.py'
        with open(self.working_dir + '/' + filename, 'w') as f:
            f.write(code_snippet)
        logger.info(f'generated step source for step {step.step_id}, filename: {filename},\nsource:\n {code_snippet}\n')
        return code_snippet, filename

    def generate_validation_file(self, step, dependencies, parameters):
        """
        Generate a validation file for a processing step.

        Args:
            step (Step): The Step object to generate validation code for.
            dependencies (List[Step]): List of dependent steps.
            parameters (dict): Parameters for code generation.

        Returns:
            tuple: A tuple containing the generated validation code and the filename.
        """
        source = self.validation_generator.generate_source(step, dependencies, parameters)
        filename = f'validate_step_{step.step_id}.py'
        with open(self.working_dir + '/' + filename, 'w') as f:
            f.write(source)
        logger.info(f'generated validation source for step {step.step_id}, filename: {filename},\nsource:\n {source}')
        return source, filename
    
    def generate_main_file(self, step, dependencies, parameters):
        """
        Generate the main execution file for the entire pipeline.

        Args:
            step (Step): The final Step object in the pipeline.
            dependencies (List[Step]): List of all steps in the pipeline.
            parameters (dict): Parameters for code generation.

        Returns:
            tuple: A tuple containing the generated main code and the filename.
        """
        source = self.main_generator.generate_source(step, dependencies, parameters)
        filename = f'main.py'
        with open(self.working_dir + '/' + filename, 'w') as f:
            f.write(source)
        logger.info(f'generated main source, filename: {filename},\nsource:\n {source}\n')
        return source, filename

    def fix_step_source(self, step, code_snippet, error_message):
        """
        Attempt to fix a step's source code based on an error message.

        Args:
            step (Step): The Step object to fix.
            code_snippet (str): The original code snippet.
            error_message (str): The error message received.

        Returns:
            tuple: A tuple containing the fixed code snippet and the filename.
        """
        prompt = (
            f"The following code snippet encountered an error:\n\n{code_snippet}\n\n"
            f"Error message:\n{error_message}\n\n"
            f"Please fix the code snippet to resolve the error without providing any explanations or comments."
        )
        source = self.model.predict(prompt)
        filename = f'step_{step.step_id}.py'
        with open(self.working_dir + '/' + filename, 'w') as f:
            f.write(source)
        logger.info(f'fixing step source for step {step.step_id}, filename: {filename},\nsource: {source}\n')
        return source, filename

    def validate_unit_code(self, filename):
        """
        Validate a unit of code by executing it.

        Args:
            filename (str): The name of the file to validate.

        Returns:
            tuple: A tuple containing a boolean indicating success and a message.
        """
        try:
            work_path = os.path.realpath(self.working_dir)
            logger.info(f'running {filename} in {work_path}')
            cmd = [self.python_path, filename]
            logger.info(' '.join(cmd))
            result = subprocess.run(
                [self.python_path, filename],
                capture_output=True,
                text=True,
                cwd=work_path
            )
            logger.debug(f'exit_code = {result.returncode}')
            if result.returncode != 0:
                raise Exception(result.stderr)
            return True, result.stdout
        except Exception as e:
            return False, str(e)