import dash
from dash import dcc, html
from dash import dash_table
from dash.dependencies import Input, Output
import pandas as pd

# Path to the CSV file
csv_file_path = "/home/neojetson/Projects/Main_PipeLine_Edge/results.csv"  # Update this path to your CSV file

# Create a Dash application
app = dash.Dash(__name__)

# Define the layout of the dashboard with a title and an interval component for refreshing
app.layout = html.Div([
    html.H1("OCR RESULTS", style={'textAlign': 'center'}),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # Refresh every 1 second
        n_intervals=0
    ),
    dash_table.DataTable(
        id='live-table',
        columns=[{"name": i, "id": i} for i in pd.read_csv(csv_file_path).columns],
        data=[],
        style_table={'height': '900px', 'overflowY': 'auto'},
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'whiteSpace': 'normal',
            'backgroundColor': '#f2f2f2',
            'color': 'black',
            'fontFamily': 'Arial',
            'fontSize': '14px',
        },
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'},  # Add alternating row colors
        ],
        style_as_list_view=True,  # Display as a list view for better readability
    )
])

# Callback to update the table every second based on the interval component
@app.callback(
    Output('live-table', 'data'),
    Input('interval-component', 'n_intervals')
)
def update_table(n):
    try:
        # Read the CSV file and update the data of the table
        df = pd.read_csv(csv_file_path)
        return df.to_dict('records')  # Return the data in dictionary format
    except Exception as e:
        return []

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)
