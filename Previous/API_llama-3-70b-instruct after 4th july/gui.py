# gui.py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import generation_script as gs
import subprocess

app = dash.Dash(__name__)
app.title = "Data Science Workflow Builder"

workflow_steps = gs.workflow_steps

app.layout = html.Div([
    html.H1("Data Science Workflow Builder"),
    
    html.Div([
        html.H3("Available Steps"),
        dcc.Checklist(
            id='workflow-step-checklist',
            options=[{'label': f"{step}: {info['description']}", 'value': step} for step, info in workflow_steps.items()],
            value=[],
            labelStyle={'display': 'block'}
        ),
    ], style={'width': '30%', 'float': 'left', 'padding': '10px'}),
    
    html.Div([
        html.H3("Selected Workflow"),
        dcc.Dropdown(
            id='workflow-step-dropdown',
            options=[],
            multi=True,
            placeholder="Selected workflow steps"
        ),
        html.Button('Generate Script', id='generate-button', n_clicks=0),
        html.Div(id='output-container', style={'whiteSpace': 'pre-line', 'marginTop': '20px'}),
        html.Button('Run Script', id='run-button', n_clicks=0),
        html.Div(id='run-output-container', style={'whiteSpace': 'pre-line', 'marginTop': '20px'})
    ], style={'width': '65%', 'float': 'right', 'padding': '10px'})
])

@app.callback(
    Output('workflow-step-dropdown', 'options'),
    Input('workflow-step-checklist', 'value')
)
def update_dropdown_options(selected_steps):
    print(f"Selected steps for dropdown: {selected_steps}")
    return [{'label': f"{step}: {workflow_steps[step]['description']}", 'value': step} for step in selected_steps]

@app.callback(
    Output('output-container', 'children'),
    Input('generate-button', 'n_clicks'),
    State('workflow-step-dropdown', 'value')
)
def generate_script(n_clicks, selected_steps):
    if n_clicks is None or not selected_steps:
        return "Please select workflow steps and click 'Generate Script'."
    
    script_content = gs.main(selected_steps)
    
    print("Generated script content:")
    print(script_content)
    return script_content

@app.callback(
    Output('run-output-container', 'children'),
    Input('run-button', 'n_clicks')
)
def run_script(n_clicks):
    if n_clicks == 0:
        return ""
    
    print("Running generated script...")
    output = gs.run_main_script()
    return output

if __name__ == '__main__':
    app.run_server(debug=True)
