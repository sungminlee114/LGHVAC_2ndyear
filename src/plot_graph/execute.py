__all__ = ["plot_graph"]

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import datetime
import pandas as pd
plt.rc('font', family='NanumGothicCoding')

from src.input_to_instructions.types import InstructionG

def plot_graph(instruction:InstructionG, variables):
    graph_type, axis, plots = instruction.type, instruction.axis, instruction.plots
    xlabel, ylabel, title = axis["xlabel"], axis["ylabel"], axis["title"]
    
    locals().update(variables)

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for plot in plots:
        plot_type, data, label = plot["type"], plot["data"], plot["label"]
        x, y = eval(data['x']), eval(data['y'])
        if plot_type == "line":
            ax.plot(x, y, label=label)

        # Check if x-axis is timestamp-based
        if isinstance(x.iloc[0], (datetime.datetime, pd.Timestamp)):
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically space timestamps
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Format timestamps
            ax.autofmt_xdate(rotation=90)  # Rotate dates to avoid overlap
        else:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))  # Limit number of x ticks for numeric values
    
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)  # Explicitly rotate x-tick labels
    # fig.tight_layout()
    fig.legend()

    return fig
    