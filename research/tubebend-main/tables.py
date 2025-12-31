import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pickle

with open('data/experiments_process_and_results.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

experiment_numbers = sorted(loaded_dict.keys())  

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Tube Bending Experiments"),

    dcc.Dropdown(
        id='experiment-number-dropdown',
        options=[{'label': num.replace('Exp_', ''), 'value': num} for num in experiment_numbers],
        value=experiment_numbers[0]  
    ),
    html.Br(),

    dcc.Dropdown(
        id='experiment-key-dropdown'
    ),
    html.Br(),

    dash_table.DataTable(
        id='table',
        columns=[],
        data=[]
    )
])

@app.callback(
    Output('experiment-key-dropdown', 'options'),
    Output('experiment-key-dropdown', 'value'),
    Input('experiment-number-dropdown', 'value')
)
def update_key_dropdown(selected_exp):
    keys = list(loaded_dict[selected_exp].keys())
    options = [{'label': f"{idx+1}-{key}", 'value': key} for idx, key in enumerate(keys)]
    value = keys[0] if keys else None
    return options, value

@app.callback(
    Output('table', 'columns'),
    Output('table', 'data'),
    Input('experiment-number-dropdown', 'value'),
    Input('experiment-key-dropdown', 'value')
)
def update_table(selected_exp, selected_key):
    if selected_key is None:
        return [], []
    df = loaded_dict[selected_exp][selected_key]
    columns = [{"name": col, "id": col} for col in df.columns]
    data = df.to_dict('records')
    return columns, data

if __name__ == '__main__':
    app.run(debug=True)
