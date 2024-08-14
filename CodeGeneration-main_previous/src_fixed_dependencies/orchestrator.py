import traceback
import subprocess
import os
import pandas as pd
from loguru import logger
import numpy as np
from sklearn.decomposition import PCA

from typing import List, Dict, Set
from language_modeling import Model, PromptGenerator
from code_generation import CodeGenerator
from graph import Step


def topological_sort(steps: List[Step]) -> List[Step]:
    used = set()
    order = []
    id_to_step = {step.step_id: step for step in steps}

    def dfs(step):
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
    MAX_RETRIES = 5
    def __init__(
          self,
          model: Model,
          prompt_generator: PromptGenerator,
          validation_generator: CodeGenerator,
          main_generator: CodeGenerator,
          working_dir: str,
        ):
        self.model = model
        self.prompt_generator = prompt_generator
        self.validation_generator = validation_generator
        self.main_generator = main_generator
        self.working_dir = working_dir


    def run_steps(self, steps: List[Step], parameters):
        os.makedirs(self.working_dir, exist_ok=True)

        id_to_step = {step.step_id: step for step in steps}
        steps_ordered = topological_sort(steps)

        for step in steps_ordered:
            try:
                code_snippet, _ = self.generate_step_file(step, parameters)
                dependencies = [id_to_step[step_id] for step_id in step.dependencies]
                _, validation_filename = self.generate_validation_file(
                    step, steps_ordered, parameters)
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
            # TODO: generate main

    def generate_step_file(self, step, parameters):
        prompt = self.prompt_generator.generate(step, parameters)
        code_snippet = self.model.predict(prompt)
        filename = f'step_{step.step_id}.py'
        with open(self.working_dir + '/' + filename, 'w') as f:
            f.write(code_snippet)
        print(f'generated step source for step {step.step_id}, filename: {filename}')
        print(code_snippet)
        print()
        return code_snippet, filename


##### New features    
    def get_all_dependencies(self, step: Step, steps: List[Step]) -> Set[Step]:
        id_to_step = {s.step_id: s for s in steps}
        dependencies = set()
        def collect_deps(current_step):
            for dep_id in current_step.dependencies:
                dep_step = id_to_step[dep_id]
                if dep_step not in dependencies:
                    dependencies.add(dep_step)
                    collect_deps(dep_step)
        collect_deps(step)
        return dependencies


    def generate_validation_file(self, step: Step, steps: List[Step], parameters):
        dependencies = self.get_all_dependencies(step, steps)
        sorted_dependencies = topological_sort(list(dependencies) + [step])

        validation_code = "import pandas as pd\n"
        for dep in sorted_dependencies:
            validation_code += f"from step_{dep.step_id} import step_{dep.step_id}\n"

        validation_code += "\n"
        for name, value in parameters.items():
            validation_code += f'{name} = {value}\n'
        validation_code += "\n"

        validation_code += "def validate_step():\n"

        for dep in sorted_dependencies:
            dep_inputs = ", ".join(dep.input_vars)
            dep_outputs = ", ".join(dep.output_vars)
            validation_code += f"    {dep_outputs} = step_{dep.step_id}({dep_inputs})\n"

        input_values = ", ".join(step.input_vars)
        output_values = ", ".join(step.output_vars)
        validation_code += f"    {output_values} = step_{step.step_id}({input_values})\n"
        validation_code += f"    print({output_values})\n"

        validation_code += "\nif __name__ == '__main__':\n"
        validation_code += "    validate_step()\n"

        filename = f'validate_step_{step.step_id}.py'
        with open(self.working_dir + '/' + filename, 'w') as f:
            f.write(validation_code)

        return validation_code, filename




    def fix_step_source(self, step, code_snippet, error_message):
      prompt = (
          f"The following code snippet encountered an error:\n\n{code_snippet}\n\n"
          f"Error message:\n{error_message}\n\n"
          f"Please fix the code snippet to resolve the error without providing any explanations or comments."
      )
      source = self.model.predict(prompt)
      filename = f'step_{step.step_id}.py'
      with open(self.working_dir + '/' + filename, 'w') as f:
          f.write(source)
      print(f'fixing step source for step {step.step_id}, filename: {filename}')
      print(source)
      print()
      return source, filename

    def validate_unit_code(self, filename):
        try:
            work_path = os.path.realpath(self.working_dir)
            print(f'running {filename} in {work_path}')
            result = subprocess.run(
                ["python", filename],
                capture_output=True,
                text=True,
                cwd=work_path
            )
            print(f'exit_code = {result.returncode}')
            if result.returncode != 0:
                raise Exception(result.stderr)
            return True, result.stdout
        except Exception as e:
            return False, str(e)
