### Import packages for dataset visualization
import plotly as py
from plotly import graph_objs as go
from plotly.offline import plot, iplot, init_notebook_mode

###Make plotly work with Jupyter Notebook
#THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON
#NOTEBOOK WHILE KERNEL IS RUNNING
init_notebook_mode(False)

def graph_predictions(df, TARGET, predictions):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[TARGET],
        name="Real Values")
    )
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[predictions],
        name="Prediction Values")
    )
    return fig
