# generation_script.py

import subprocess
import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_url = "https://openrouter.ai/api/v1"
api_key = os.getenv('OPENROUTER_API_KEY')

workflow_steps = {
    11: {
        "description": "Load the CSV file as pandas DataFrame",
        "dependencies": [],
        "input": ["csv_path"],
        "output": ["df"]
    },
    21: {
        "description": "Examine the structure and characteristics of the data",
        "dependencies": [11],
        "input": ["df"],
        "output": ["structure_info"]
    },
    31: {
        "description": "Identify missing values, data types, and handle missing values if there are any",
        "dependencies": [11, 21],
        "input": ["df"],
        "output": ["df_cleaned", "data_types_info"]
    },
    32: {
        "description": "Identify if there is a need to convert categorical variables to numerical representations. If yes, then convert them.",
        "dependencies": [11, 31],
        "input": ["df_cleaned", "data_types_info"],
        "output": ["df_encoded"]
    },
    51: {
        "description": "Split the preprocessed data into training and testing sets, and implement a machine learning algorithm (choose from scikit-learn, XGBoost, LightGBM, or CatBoost).",
        "dependencies": [11, 31, 32],
        "input": ["df_encoded"],
        "output": ["model", "X_train", "X_test", "y_train", "y_test"]
    },
    61: {
        "description": "Evaluate the model's performance on both training and testing data, calculate evaluation metrics (for classification: [accuracy, precision, recall, F1-score]; for regression: [R^2, MSE, RMSE]), and compare the difference.",
        "dependencies": [51],
        "input": ["model", "X_train", "X_test", "y_train", "y_test"],
        "output": ["evaluation_results", "metrics"]
    }
}

example_step_script = """
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def step_32(df_cleaned, data_types_info):
    df_encoded = df_cleaned.copy()
    categorical_cols = [col for col, dtype in data_types_info.items() if dtype == 'object']
    le = LabelEncoder()
    for col in categorical_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded
"""

def openai_chat(request):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/llama-3-70b-instruct",
        "messages": [{"role": "user", "content": request}]
    }
    response = requests.post(f"{api_url}/chat/completions", headers=headers, json=data)
    
    # Log the full response for debugging
    response_json = response.json()
    print("Full API response:", response_json)
    
    # Check if 'choices' key exists in the response
    if "choices" in response_json and response_json["choices"]:
        return response_json["choices"][0]["message"]["content"]
    else:
        print("Request:", data)
        print("Response:", response_json)
        raise ValueError("The response does not contain 'choices'. Full response: " + str(response_json))

def generate_code_snippet(request):
    response = openai_chat(request)
    return response

def clean_and_correct_code(generated_code, csv_path):
    cleaned_code = generated_code.replace("```python", "").replace("```", "").strip()
    cleaned_code_lines = [line for line in cleaned_code.split("\n") if not line.lower().startswith("here is the")]
    cleaned_code = "\n".join(cleaned_code_lines)
    corrected_code = cleaned_code.replace("{csv_path}", f"'{csv_path}'")
    return corrected_code

def validate_unit_code(code_filename):
    try:
        result = subprocess.run(["python", code_filename], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(result.stderr)
        return True, result.stdout
    except Exception as e:
        return False, str(e)

def fix_code(code_snippet, error_message, csv_path):
    request = (
        f"The following code snippet encountered an error:\n\n{code_snippet}\n\n"
        f"Error message:\n{error_message}\n\n"
        f"Please fix the code snippet to resolve the error without providing any explanations or comments."
    )
    fixed_code = generate_code_snippet(request)
    return clean_and_correct_code(fixed_code, csv_path)

def get_all_dependencies(step, workflow_steps):
    dependencies = set(workflow_steps[step]["dependencies"])
    for dep in workflow_steps[step]["dependencies"]:
        dependencies.update(get_all_dependencies(dep, workflow_steps))
    return dependencies

def generate_code_for_step(step, workflow_steps, csv_path, dataset_info):
    print(f"Generating code for step {step}")
    request = (
        f"Here is an example of a good step script:\n\n{example_step_script}\n\n"
        f"Write a Python function named 'step_{step}' for the following step: {workflow_steps[step]['description']}. "
        f"The function should take {', '.join(workflow_steps[step]['input'])} as input and return {', '.join(workflow_steps[step]['output'])}. "
        f"Ensure to include necessary imports and handle edge cases. "
        f"The dataset has the following columns: {dataset_info['columns']}. "
        f"The data types are: {dataset_info['types']}. "
        f"Here's a sample of the data: {dataset_info['sample_data']}. "
        f"Value counts (top 5): {dataset_info['value_counts']}. "
        f"Statistical description: {dataset_info['description']}. "
        f"Only return the function definition without any additional code or explanations."
    )
    code_snippet = generate_code_snippet(request)
    return clean_and_correct_code(code_snippet, csv_path)

def generate_validation_file(step, workflow_steps):
    print(f"Generating validation file for step {step}")
    dependencies = get_all_dependencies(step, workflow_steps)
    input_params = workflow_steps[step]["input"]
    output_params = workflow_steps[step]["output"]

    validation_code = "import pandas as pd\n"
    for dep in sorted(dependencies):
        validation_code += f"from step_{dep} import step_{dep}\n"
    validation_code += f"from step_{step} import step_{step}\n\n"

    validation_code += "def validate_step():\n"
    validation_code += "    csv_path = '/Users/ilya/Desktop/GitHub_Repositories/Thesis/datasets/insurance.csv'\n"

    for dep in sorted(dependencies):
        dep_inputs = ", ".join(workflow_steps[dep]["input"])
        dep_outputs = ", ".join(workflow_steps[dep]["output"])
        validation_code += f"    {dep_outputs} = step_{dep}({dep_inputs})\n"

    input_values = ", ".join(input_params)
    output_values = ", ".join(output_params)
    validation_code += f"    {output_values} = step_{step}({input_values})\n"
    validation_code += f"    print({output_values})\n"

    validation_code += "\nif __name__ == '__main__':\n"
    validation_code += "    validate_step()\n"

    with open(f"validate_step_{step}.py", "w") as file:
        file.write(validation_code)

def save_dataset_info(csv_path, info_file_path):
    print(f"Saving dataset info for {csv_path}")
    df = pd.read_csv(csv_path)
    columns = df.columns.tolist()
    types = df.dtypes.apply(lambda x: str(x)).to_dict()
    sample_data = df.head().to_dict(orient='list')
    value_counts = {col: df[col].value_counts().head().to_dict() for col in df.columns}
    description = df.describe().to_dict()

    dataset_info = {
        'columns': columns,
        'types': types,
        'sample_data': sample_data,
        'value_counts': value_counts,
        'description': description
    }

    with open(info_file_path, 'w') as file:
        json.dump(dataset_info, file)

def generate_main_file(workflow_steps, selected_step_numbers, csv_path):
    print("Generating main script")
    main_code = "import pandas as pd\n\n"
    for step in selected_step_numbers:
        main_code += f"from step_{step} import step_{step}\n"

    main_code += "\ndef main():\n"
    main_code += f"    csv_path = '{csv_path}'\n"
    main_code += "    df = pd.read_csv(csv_path)\n"

    for step in selected_step_numbers:
        input_params = ", ".join(workflow_steps[step]['input'])
        output_params = ", ".join(workflow_steps[step]['output'])
        main_code += f"    {output_params} = step_{step}({input_params})\n"

    main_code += "\nif __name__ == '__main__':\n"
    main_code += "    main()"

    with open("main.py", "w") as file:
        file.write(main_code)

def generate_and_validate_steps(csv_path, selected_step_numbers):
    info_file_path = "dataset_info.json"

    if not os.path.exists(info_file_path):
        save_dataset_info(csv_path, info_file_path)

    with open(info_file_path, 'r') as f:
        dataset_info = json.load(f)

    for step in selected_step_numbers:
        try:
            code_snippet = generate_code_for_step(step, workflow_steps, csv_path, dataset_info)
            with open(f"step_{step}.py", "w") as file:
                file.write(code_snippet)
            generate_validation_file(step, workflow_steps)

            success, output = validate_unit_code(f"validate_step_{step}.py")
            while not success:
                print(f"Validation failed for step {step}: {output}")
                fixed_code = fix_code(code_snippet, output, csv_path)
                with open(f"step_{step}.py", "w") as file:
                    file.write(fixed_code)
                generate_validation_file(step, workflow_steps)
                success, output = validate_unit_code(f"validate_step_{step}.py")
        except Exception as e:
            print(f"Error processing step {step}: {e}")
            continue

    print("Validation completed successfully.")

def run_main_script():
    print("Running main script...")
    result = subprocess.run(["python", "main.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running main script: {result.stderr}")
        return f"Error:\n{result.stderr}"
    print(f"Main script output: {result.stdout}")
    return f"Output:\n{result.stdout}"

def main(selected_step_numbers):
    csv_path = "/Users/ilya/Desktop/GitHub_Repositories/Thesis/datasets/insurance.csv"
    generate_and_validate_steps(csv_path, selected_step_numbers)
    generate_main_file(workflow_steps, selected_step_numbers, csv_path)
    print("Main script generated successfully.")
    return run_main_script()

if __name__ == "__main__":
    selected_step_numbers = [11, 21, 31, 32, 51, 61]
    main(selected_step_numbers)
