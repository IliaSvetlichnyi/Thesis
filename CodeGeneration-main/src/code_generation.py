from typing import List, Dict
from graph import Step


class CodeGenerator:
    """
    Base class for code generators.
    """

    def generate_source(self, step, step_dependencies, parameters):
        """
        Generate source code for a given step.

        Args:
            step: The current step to generate code for.
            step_dependencies: Dependencies of the current step.
            parameters: Additional parameters for code generation.

        Returns:
            Generated source code as a string.
        """
        pass


class MainCodeGenerator(CodeGenerator):
    """
    Generates main code for executing steps.
    """

    def __init__(self, additional_lines):
        """
        Initialize the MainCodeGenerator.

        Args:
            additional_lines (List[str]): Additional lines to be added to the generated code.
        """
        self.additional_lines = additional_lines

    def generate_source(self, step: Step, step_dependencies, parameters):
        """
        Generate main source code for executing steps.

        Args:
            step (Step): The current step to generate code for.
            step_dependencies (List[Step]): Dependencies of the current step.
            parameters (Dict): Additional parameters for code generation.

        Returns:
            str: Generated main source code.
        """
        input_params = step.input_vars
        output_params = step.output_vars

        main_code = "import pandas as pd\n"
        for dep in step_dependencies:
            main_code += f"from step_{dep.step_id} import step_{dep.step_id}\n"

        main_code += "\n"
        for name, value in parameters.items():
            main_code += f'{name} = {value}\n'
        main_code += "\n"

        main_code += "def main():\n"

        for dep in step_dependencies:
            dep_inputs = ", ".join(dep.input_vars)
            dep_outputs = ", ".join(dep.output_vars)
            main_code += f"    {dep_outputs} = step_{dep.step_id}({dep_inputs})\n"

        input_values = ", ".join(input_params)
        output_values = ", ".join(output_params)
        main_code += f"    {output_values} = step_{step.step_id}({input_values})\n"
        main_code += f"    print({output_values})\n"

        for line in self.additional_lines:
            main_code += f"    {line}\n"      

        main_code += "\nif __name__ == '__main__':\n"
        main_code += "    main()\n"
        return main_code
        

class ValidationCodeGenerator(CodeGenerator):
    """
    Generates validation code for steps.
    """

    def generate_source(self, step: Step, step_dependencies: List[Step], parameters):
        """
        Generate validation source code for a step.

        Args:
            step (Step): The current step to generate validation code for.
            step_dependencies (List[Step]): Dependencies of the current step.
            parameters (Dict): Additional parameters for code generation.

        Returns:
            str: Generated validation source code.
        """
        input_params = step.input_vars
        output_params = step.output_vars

        validation_code = "import pandas as pd\n"
        for dep in step_dependencies:
            validation_code += f"from step_{dep.step_id} import step_{dep.step_id}\n"
        validation_code += f"from step_{step.step_id} import step_{step.step_id}\n\n"

        validation_code += "\n"
        for name, value in parameters.items():
            validation_code += f'{name} = {value}\n'
        validation_code += "\n"

        validation_code += "def validate_step():\n"

        for dep in step_dependencies:
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