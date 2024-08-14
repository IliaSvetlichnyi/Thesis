import requests
from loguru import logger
from graph import Step
from typing import List, Dict


class OpenAiLlamaApi:
    """
    A class to interact with the OpenAI LLaMA API.
    """

    def __init__(self, api_url, api_key, model_tag):
        """
        Initialize the OpenAiLlamaApi instance.

        Args:
            api_url (str): The URL of the API.
            api_key (str): The API key for authentication.
            model_tag (str): The tag of the model to use.
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model_tag = model_tag

    def execute_request(self, request):
        """
        Execute a request to the LLaMA API.

        Args:
            request (str): The prompt to send to the API.

        Returns:
            str: The generated content from the API.

        Raises:
            ValueError: If the API response does not contain the expected data.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_tag,
            "messages": [{"role": "user", "content": request}]
        }
        response = requests.post(
            f"{self.api_url}/chat/completions", headers=headers, json=data)

        # Log the full response for debugging
        response_json = response.json()
        logger.info(f"Full API response: {response_json}")

        # Check if 'choices' key exists in the response
        if "choices" in response_json and response_json["choices"]:
            return response_json["choices"][0]["message"]["content"]
        else:
            logger.error(f"Request: {data}")
            logger.error(f"Response: {response_json}")
            raise ValueError(
                "The response does not contain 'choices'. Full response: " + str(response_json))


class Model:
    """
    Base class for language models.
    """

    def predict(self, prompt):
        """
        Generate a prediction based on the given prompt.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The generated prediction.
        """
        pass


class LlamaModel(Model):
    """
    A class representing the LLaMA language model.
    """

    def __init__(self, llamaApi: OpenAiLlamaApi):
        """
        Initialize the LlamaModel instance.

        Args:
            llamaApi (OpenAiLlamaApi): An instance of the OpenAiLlamaApi class.
        """
        self.api = llamaApi

    def predict_raw(self, prompt):
        """
        Generate a raw prediction using the LLaMA API.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The raw generated content.
        """
        return self.api.execute_request(prompt)

    def predict(self, prompt):
        """
        Generate a cleaned and corrected prediction.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The cleaned and corrected generated content.
        """
        generated_code = self.predict_raw(prompt)
        return self._clean_and_correct_code(generated_code)

    def _clean_and_correct_code(self, generated_code):
        """
        Clean and correct the generated code.

        Args:
            generated_code (str): The raw generated code.

        Returns:
            str: The cleaned and corrected code.
        """
        cleaned_code = generated_code.replace(
            "```python", "").replace("```", "").strip()
        cleaned_code_lines = [line for line in cleaned_code.split(
            "\n") if not line.lower().startswith("here is the")]
        cleaned_code = "\n".join(cleaned_code_lines)
        return cleaned_code


class PromptGenerator:
    """
    A class for generating prompts for the language model.
    """

    def __init__(self, example_script, dataset_info):
        """
        Initialize the PromptGenerator instance.

        Args:
            example_script (str): An example script to use in the prompt.
            dataset_info (dict): Information about the dataset.
        """
        self.example_script = example_script
        self.dataset_info = dataset_info

    def generate(self, step: Step, parameters):
        """
        Generate a prompt for a given step.

        Args:
            step (Step): The step for which to generate the prompt.
            parameters (dict): Additional parameters to include in the prompt.

        Returns:
            str: The generated prompt.
        """
        parameters_str = ', '.join(
            f'{name}={val}' for name, val in parameters.items())
        request = (
            f"Here is an example of a good step script:\n\n{self.example_script}\n\n"
            f"Write a Python function named 'step_{step.step_id}' for the following step: {step.description}. "
            f"The function should take {', '.join(step.input_vars)} as input and return {', '.join(step.output_vars)}. "
            f"Ensure to include necessary imports and handle edge cases. "
            f"Additional information: {step.additional_info}\n"
            f"The dataset has the following columns: {self.dataset_info['columns']}. "
            f"The data types are: {self.dataset_info['types']}. "
            f"Here's a sample of the data: {self.dataset_info['sample_data']}. "
            f"Value counts (top 5): {self.dataset_info['value_counts']}. "
            f"Statistical description: {self.dataset_info['description']}. "
            f"Use these predefined parameters if needed: {parameters_str}. "
            f"The input 'Segments' is a list of 1D numpy arrays, each representing a segment of the signal data. "
            f"Each segment should be normalized independently using sklearn's MinMaxScaler. "
            f"The output 'Segments_normalized' should be a list of normalized 1D numpy arrays. "
            f"Only return the function definition without any additional code or explanations."
        )
        return request
