from typing import List, Dict


class Step:
    """
    Represents a step in a processing pipeline.

    This class encapsulates information about a single step, including its
    identifier, description, dependencies, input and output variables,
    and any additional information.
    """

    def __init__(self,
                 step_id: str,
                 description: str,
                 dependencies: List[str],
                 input_vars: List[str],
                 output_vars: List[str],
                 additional_info):
        """
        Initialize a Step instance.

        Args:
            step_id (str): Unique identifier for the step.
            description (str): Brief description of the step's purpose.
            dependencies (List[str]): List of step IDs that this step depends on.
            input_vars (List[str]): List of input variable names for this step.
            output_vars (List[str]): List of output variable names for this step.
            additional_info: Any additional information relevant to the step.
        """
        self.step_id = step_id
        self.description = description
        self.dependencies = dependencies
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.additional_info = additional_info

    def __str__(self):
        """
        Return a string representation of the Step instance.

        Returns:
            str: A string representation of the Step's attributes.
        """
        return f"Step({repr(self.__dict__)})"

    def __repr__(self):
        """
        Return a string representation of the Step instance.

        This method returns the same string as __str__ for consistency.

        Returns:
            str: A string representation of the Step's attributes.
        """
        return str(self)
