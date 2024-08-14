import requests
import json  # Add this line to import the json module
from loguru import logger

from graph import Step

from typing import List, Dict


class OpenAiLlamaApi:
    def __init__(self, api_url, api_key,  model_tag):
        self.api_url = api_url
        self.api_key = api_key
        self.model_tag = model_tag

    def execute_request(self, request):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_tag,
            "messages": [{"role": "user", "content": request}]
        }
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions", headers=headers, json=data)
            response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx

            # Log the full response text
            logger.info(f"Full API response text: {response.text}")

            try:
                response_json = response.json()
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to decode JSON. Response text: {response.text}")
                logger.error(f"JSON decode error: {str(e)}")
                raise

            logger.info(f"Full API response JSON: {response_json}")

            if "choices" in response_json and response_json["choices"]:
                return response_json["choices"][0]["message"]["content"]
            else:
                logger.error(
                    f"Unexpected response format. Full response: {response_json}")
                raise ValueError("The response does not contain 'choices'.")

        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            logger.error(f"Request: {data}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response text: {e.response.text}")
            raise
class Model:
    def predict(self, promt):
        pass

class LlamaModel(Model):
    def __init__(self, llamaApi: OpenAiLlamaApi):
        self.api = llamaApi

    def predict_raw(self, prompt):
        return self.api.execute_request(prompt)

    def predict(self, prompt):
        generated_code = self.predict_raw(prompt)
        return self._clean_and_correct_code(generated_code)

    def _clean_and_correct_code(self, generated_code):
        cleaned_code = generated_code.replace("```python", "").replace("```", "").strip()
        cleaned_code_lines = [line for line in cleaned_code.split("\n") if not line.lower().startswith("here is the")]
        cleaned_code = "\n".join(cleaned_code_lines)
        return cleaned_code

class PromptGenerator:
    def __init__(self, EXAMPLE_STEP_SCRIPT, EXAMPLE_VALIDATION_SCRIPT, dataset_info):
        self.EXAMPLE_STEP_SCRIPT = EXAMPLE_STEP_SCRIPT
        self.EXAMPLE_VALIDATION_SCRIPT = EXAMPLE_VALIDATION_SCRIPT
        self.dataset_info = dataset_info

    def generate(self, step: Step, parameters):
        # f"Use these predefined parameters if needed: SizeSegment={SizeSegment}, gamma={gamma}, nu={nu}, kernel='{kernel}', NC_pca={NC_pca}, Dec_levels={Dec_levels}. "
        parameters_str = ', '.join(f'{name}={val}' for name, val in parameters.items())
        request = (
            f"Here is an example of a good step script:\n\n{self.EXAMPLE_STEP_SCRIPT}\n\n"
            f"Here is an example of a good validation script:\n\n{self.EXAMPLE_VALIDATION_SCRIPT}\n\n"
            f"Write a Python function named 'step_{step.step_id}' for the following step: {step.desiption}. "
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