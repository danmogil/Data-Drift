from dash import Dash, dcc, html, Input, Output
from plotly.express import scatter, bar, pie
from pandas import DataFrame, read_parquet
from numpy import where

app = Dash(__name__)

axis = [{'label': "Income", "value": "Income"},
        {"label": 'Loaning Risk', "value": "Loaning Risk"},
        {"label": 'Credit', "value": "Credit"},
        {"label": 'Finance', "value": "Finance"},
        {"label": 'Health', "value": "Health"},
        {"label": 'Social Media', "value": "SocialMedia"},
        {"label": "Base", "value": "base_value"},
        {"label": "Outcome", "value": "outcome"}]

theme = 'plotly_dark'
app.layout = html.Div(children=[
    html.Div(
        id='header',
        children=[
            html.H1(children='Shapley Drift', id="title"),
            dcc.Dropdown(
                    id="x_axis",
                    value="Income",
                    options=axis,
                ),
            dcc.Dropdown(
                    id="y_axis",
                    value="outcome",
                    options=axis,
                ),
        ]
    ),
    html.Div(
        id='content',
        children=[
            dcc.Graph(id="scatter", style={"float": "left"}),
            html.Div(style={"float": "right", "height": "50%"}, children=[
                dcc.Graph(id="bar"),
                dcc.Graph(id="pie"),
            ]),
        ]
    ),
])

@app.callback(
    Output("scatter", "figure"),
    Input("x_axis", "value"),
    Input("y_axis", "value"))
def generate_chart(x_axis, y_axis):
    all_shap = read_parquet('model/demo/shap.parquet', 'pyarrow')
    fig = scatter(all_shap, x=x_axis, y=y_axis, animation_frame="Model Num", color="Race",
                     hover_name="outcome", template=theme, title="SHAP Feature Drift")
    return fig

@app.callback(
    Output("bar", "figure"),
    Input("x_axis", "value")) 
def generate_chart(x_axis):
    group_shap = read_parquet('model/demo/group_shap.parquet')
    fig = bar(group_shap, x='Model Num',
                 y=['Income', 'Credit', 'Loaning Risk', 'Travel', 'Finance', 'Health', 'SocialMedia'], template=theme,
                 title="Cumulative Feature Contribution")
    return fig

@app.callback(
    Output("pie", "figure"),
    Input('x_axis', 'value')) 
def generate_chart(x_axis):
    shap: DataFrame = read_parquet('model/demo/shap.parquet', 'pyarrow', ['Race', 'outcome'])

    total = shap.size

    b_accepted = len(where((shap['Race'] == 'Black') & (shap['outcome'] >= 0.5))[0])
    w_accepted = len(where((shap['Race'] == 'White') & (shap['outcome'] >= 0.5))[0])

    fig = pie(shap, values=[b_accepted/total, w_accepted/total], names=['Black', 'White'], template='plotly_dark', title="Loan Approval Rate by Race")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)