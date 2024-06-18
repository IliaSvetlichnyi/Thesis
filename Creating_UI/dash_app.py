from dash import Dash, dcc, html, Input, Output, State, callback_context
import generator  # Import the generator script
import threading
import time

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
    dcc.Interval(id='progress-interval', interval=1000,
                 n_intervals=0, disabled=True),
    html.Div(id='progress-status'),
    html.Div(
        children=[
            html.Div(id='progress-bar',
                     style={'width': '0%', 'background-color': 'green', 'height': '30px'})
        ],
        style={'width': '100%', 'background-color': '#ddd',
               'height': '30px', 'margin-top': '20px'}
    ),
    html.Div(id='generated-code-output', style={'whiteSpace': 'pre-line'})
])

# Global variables to store progress and status
progress = 0
status_message = ""
generated_code_output = ""


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
    Output('progress-bar', 'style'),
    Output('progress-status', 'children'),
    Output('progress-interval', 'disabled'),
    Output('generated-code-output', 'children'),
    Input('progress-interval', 'n_intervals'),
    Input('generate-code-button', 'n_clicks'),
    State('task-container', 'children')
)
def manage_generation(n_intervals, n_clicks, tasks):
    global progress, status_message, generated_code_output
    ctx = callback_context

    if not ctx.triggered:
        return {'width': '0%'}, "", True, ""

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == 'generate-code-button':
        task_details = []
        for task in tasks:
            try:
                task_value = task['props']['children'][1]['props']['value']
                if task_value:
                    task_details.append(task_value)
            except (KeyError, IndexError, TypeError):
                continue

        if not task_details:
            status_message = "No task details provided."
            return {'width': '0%'}, status_message, True, ""

        csv_path = "/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv"

        # Start the code generation in a separate thread
        threading.Thread(target=generate_code_task,
                         args=(task_details, csv_path)).start()
        return {'width': '0%'}, "Starting code generation...", False, ""

    elif trigger == 'progress-interval':
        progress_bar_style = {'width': f'{progress}%',
                              'background-color': 'green', 'height': '30px'}
        if progress >= 100:
            return progress_bar_style, status_message, True, generated_code_output
        return progress_bar_style, status_message, False, generated_code_output

    return {'width': '0%'}, "", True, ""


def generate_code_task(task_details, csv_path):
    global progress, status_message, generated_code_output
    progress = 0
    status_message = "Starting code generation..."
    generated_code_output = ""

    columns, types, sample_data, value_counts, description = generator.get_dataset_info(
        csv_path)
    generated_codes = []
    total_tasks = len(task_details)

    for i, task in enumerate(task_details):
        progress = int(((i + 1) / total_tasks) * 100)
        status_message = f"Processing task {i + 1} of {total_tasks}..."
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
            generated_code_output += f"Task {i + 1} output:\n{output}\n\n"
        else:
            status_message = f"Validation failed for task {task}: {output}"
            progress = 100
            return

        time.sleep(1)  # Simulate processing time for each task

    combined_code = "\n\n".join(generated_codes)
    with open("combined_generated_code.py", "w") as file:
        file.write(combined_code)

    generated_code_output += "Combined generated code:\n" + combined_code
    status_message = "All tasks completed successfully."
    progress = 100


if __name__ == '__main__':
    app.run_server(debug=True)
