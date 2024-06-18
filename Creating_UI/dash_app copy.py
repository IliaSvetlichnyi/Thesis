from dash import Dash, dcc, html, Input, Output, State
import generator  # Import the generator script

app = Dash(__name__)

app.layout = html.Div([
    html.H1('ML Task Generator'),
    dcc.Dropdown(
        id='task-dropdown',
        options=[
            {'label': 'Clean Dataset', 'value': 'clean'},
            {'label': 'Divide into Train and Test', 'value': 'divide'},
            {'label': 'Train Model', 'value': 'train'},
            {'label': 'Evaluate Model', 'value': 'evaluate'},
        ],
        placeholder='Select a task'
    ),
    html.Button('Add Task', id='add-task-button', n_clicks=0),
    html.Div(id='task-container', children=[]),
    html.Button('Generate Code', id='generate-code-button', n_clicks=0),
    html.Div(id='generated-code-output', style={'whiteSpace': 'pre-line'})
])


@app.callback(
    Output('task-container', 'children'),
    Input('add-task-button', 'n_clicks'),
    State('task-dropdown', 'value'),
    State('task-container', 'children')
)
def add_task(n_clicks, selected_task, task_container):
    if n_clicks > 0 and selected_task:
        new_task = html.Div([
            html.Label(f'Task {n_clicks}: {selected_task}'),
            dcc.Input(id=f'input-{n_clicks}', type='text',
                      placeholder='Enter task details')
        ])
        task_container.append(new_task)
    return task_container


@app.callback(
    Output('generated-code-output', 'children'),
    Input('generate-code-button', 'n_clicks'),
    State('task-container', 'children')
)
def generate_code(n_clicks, tasks):
    if n_clicks > 0:
        task_details = []
        for task in tasks:
            try:
                task_value = task['props']['children'][1]['props']['value']
                if task_value:
                    task_details.append(task_value)
            except (KeyError, IndexError, TypeError):
                continue

        if not task_details:
            return "No task details provided."

        # Define the path to the CSV file
        # Update this path as needed
        csv_path = "/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv"

        # Get dataset info
        columns, types, sample_data, value_counts, description = generator.get_dataset_info(
            csv_path)

        generated_codes = []
        for i, task in enumerate(task_details):
            request = (
                f"{task} for the dataset with columns: {columns}. "
                f"The data types are: {types}. Sample data: {sample_data}. Value counts: {value_counts}. "
                f"Description: {description}."
            )
            generated_code = generator.generate_unit_code(request)
            corrected_code = generator.clean_and_correct_code(
                generated_code, csv_path)
            success, output = generator.validate_unit_code(corrected_code, i)
            if success:
                generated_codes.append(corrected_code)
            else:
                return f"Validation failed for task {task}: {output}"

        # Combine all codes into one
        combined_code = "\n\n".join(generated_codes)
        with open("combined_generated_code.py", "w") as file:
            file.write(combined_code)

        return combined_code
    return ""


if __name__ == '__main__':
    app.run_server(debug=True)
