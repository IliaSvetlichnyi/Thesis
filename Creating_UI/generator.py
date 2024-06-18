from langchain_core.messages import SystemMessage
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
import pandas as pd
import subprocess

# Setup the prompt templates
human_prompt = HumanMessagePromptTemplate.from_template("{request}")
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a highly skilled data scientist with 20 years of experience. You specialize in writing clean, efficient, and error-free ML code. Generate only the code snippets without any explanations or comments."),
    human_prompt
])
model = Ollama(model="llama3")

# Function to generate code with LLM


def generate_unit_code(task):
    formatted_request = chat_prompt.format_prompt(
        request=task).to_messages()
    response = model.invoke(formatted_request)
    generated_code = response
    # Remove lines starting with comments or explanations
    lines = generated_code.split("\n")
    pure_code_lines = [line for line in lines if not line.startswith(
        "#") and not line.startswith("As a")]
    pure_code = "\n".join(pure_code_lines)
    return pure_code


def clean_and_correct_code(generated_code, csv_path):
    cleaned_code = generated_code.replace("```", "").strip()
    cleaned_code_lines = cleaned_code.split("\n")
    cleaned_code_lines = [
        line for line in cleaned_code_lines if not line.lower().startswith("here is the")]
    cleaned_code = "\n".join(cleaned_code_lines)
    if "python" in cleaned_code:
        cleaned_code = cleaned_code.split("python")[1].strip()
    corrected_code = cleaned_code.replace("{csv_path}", f"{csv_path}")
    return corrected_code


def get_dataset_info(csv_path):
    df = pd.read_csv(csv_path)
    columns = df.columns.tolist()
    types = df.dtypes.to_dict()
    sample_data = df.head().to_dict(orient='list')
    value_counts = {col: df[col].value_counts().to_dict()
                    for col in df.columns}
    description = df.describe().to_dict()
    return columns, types, sample_data, value_counts, description


def validate_code(code_filename):
    try:
        result = subprocess.run(
            ["python", code_filename], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(result.stderr)
        return True, result.stdout
    except Exception as e:
        return False, str(e)

def main_part1():
    csv_path = "/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv"
    columns, types, sample_data, value_counts, description = get_dataset_info(
        csv_path)

    tasks = [
        "Clean the dataset",
        "Divide it into train and test sets",
        "Train an appropriate model",
        "Evaluate the model's performance"
    ]

    generated_codes = []
    for task in tasks:
        request = (
            f"{task} for the dataset with columns: {columns}. "
            f"The data types are: {types}. Sample data: {sample_data}. Value counts: {value_counts}. "
            f"Description: {description}."
        )
        generated_code = generate_unit_code(request)
        corrected_code = clean_and_correct_code(generated_code, csv_path)
        generated_codes.append(corrected_code)

    # Combine all codes into one
    combined_code = "\n\n".join(generated_codes)
    with open("combined_generated_code.py", "w") as file:
        file.write(combined_code)

    print("Generated code saved to combined_generated_code.py")


def validate_unit_code(unit_code, unit_index):
    code_filename = f"unit_{unit_index}_code.py"
    with open(code_filename, "w") as file:
        file.write(unit_code)

    success, output = validate_code(code_filename)
    if not success:
        raise Exception(f"Validation failed for unit {unit_index}: {output}")
    return success, output


def main_part3():
    csv_path = "/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv"
    columns, types, sample_data, value_counts, description = get_dataset_info(
        csv_path)

    tasks = [
        "Clean the dataset",
        "Divide it into train and test sets",
        "Train an appropriate model",
        "Evaluate the model's performance"
    ]

    generated_codes = []
    for i, task in enumerate(tasks):
        request = (
            f"{task} for the dataset with columns: {columns}. "
            f"The data types are: {types}. Sample data: {sample_data}. Value counts: {value_counts}. "
            f"Description: {description}."
        )
        generated_code = generate_unit_code(request)
        corrected_code = clean_and_correct_code(generated_code, csv_path)
        success, output = validate_unit_code(corrected_code, i)
        if success:
            generated_codes.append(corrected_code)
        else:
            print(f"Validation failed for task {task}")

    # Combine all codes into one
    combined_code = "\n\n".join(generated_codes)
    with open("combined_generated_code.py", "w") as file:
        file.write(combined_code)

    print("Generated code saved to combined_generated_code.py")
