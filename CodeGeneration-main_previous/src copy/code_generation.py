from typing import List, Dict
from graph import Step

class CodeGenerator:
  def generate_source(self, step, step_dependencies, parameters, additonal_lines):
    pass

class MainCodeGenerator(CodeGenerator):
    def generate_source(self, step: Step, step_dependencies, parameters, additonal_lines=[]):
        # TODO: implement
        pass


class ValidationCodeGenerator(CodeGenerator):
    def generate_source(self, step: Step, step_dependencies: List[Step], parameters, additional_lines=[]):
        input_params = step.input_vars
        output_params = step.output_vars

        validation_code = "import pandas as pd\n"
        for dep in sorted(step_dependencies, key=lambda s: s.step_id):
            validation_code += f"from step_{dep.step_id} import step_{dep.step_id}\n"
        validation_code += f"from step_{step.step_id} import step_{step.step_id}\n\n"

        validation_code += "\n"
        for name, value in parameters.items():
            validation_code += f'{name} = {value}\n'
        validation_code += "\n"

        validation_code += "def validate_step():\n"

        for dep in sorted(step_dependencies, key=lambda s: s.step_id):
            dep_inputs = ", ".join(dep.input_vars)
            dep_outputs = ", ".join(dep.output_vars)
            validation_code += f"    {dep_outputs} = step_{dep.step_id}({dep_inputs})\n"

        input_values = ", ".join(input_params)
        output_values = ", ".join(output_params)
        validation_code += f"    {output_values} = step_{step.step_id}({input_values})\n"
        validation_code += f"    print({output_values})\n"

        validation_code += "\nif __name__ == '__main__':\n"
        validation_code += "    validate_step()\n"
        return validation_code


def collect_all_dependencies(step: Step, id_to_step: Dict[str, Step]) -> List[Step]:
    all_dependencies = []
    visited = set()

    def dfs(current_step: Step):
        if current_step.step_id in visited:
            return
        visited.add(current_step.step_id)
        for dep_id in current_step.dependencies:
            dep_step = id_to_step[dep_id]
            dfs(dep_step)
            all_dependencies.append(dep_step)

    dfs(step)
    return all_dependencies
