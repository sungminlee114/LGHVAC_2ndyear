__all__ = ["plot_graph"]

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import datetime
import pandas as pd
import matplotlib.font_manager as fm

#plt.rc('font', family='NanumGothicCoding')
plt.rc('font', family='NanumMyeongjo') 
import matplotlib.font_manager as fm


from src.input_to_instructions.types import InstructionG
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datetime
import matplotlib.ticker as ticker

import matplotlib.dates as mdates
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_graph(instruction, variables):
    graph_type, axis, plots = instruction.type, instruction.axis, instruction.plots
    
    # Update the local variables with any additional variables
    locals().update(variables)

    # Create subplots based on the number of axes
    num_axes = len(axis)
    fig, axes = plt.subplots(num_axes, figsize=(10, 6 * num_axes), layout="constrained")
    
    # Ensure axes is iterable
    if num_axes == 1:
        axes = [axes]
    
    # Loop through each axis and plot accordingly
    for i, ax in enumerate(axes):
        xlabel, ylabel, title = axis[i]["description"]["xlabel"], axis[i]["description"]["ylabel"], axis[i]["description"]["title"]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Iterate through the plots for this axis
        for item in axis[i]["items"]:
            plot_type = axis[i]["type"]
            label = item["label"]
            x = eval(item["x"])  # Evaluate the expression for x-axis data
            y = eval(item["y"])  # Evaluate the expression for y-axis data
            
            period_unit = "D"
            x = pd.to_datetime(x)  # Ensure x is datetime format
            
            if isinstance(x.iloc[0], (datetime.datetime, pd.Timestamp)):
                # Calculate the date range for resampling decision
                date_range = max(x) - min(x)
                
                if date_range.days > 365:
                    period_unit = "Y"
                elif date_range.days > 30:
                    period_unit = "M"
                elif date_range.days > 7:
                    period_unit = "W"
                else:
                    period_unit = "D"
                
                # Resample the data based on daily granularity but display the period unit
                df = pd.DataFrame({'Date': x, 'Y': y})
                df.set_index('Date', inplace=True)
                
                if plot_type == "line":
                    
                    df_resampled = df.resample("D").mean()

                elif plot_type == "bar":
                    df_resampled = df.resample("D").apply(lambda x: x.mode()[0] if not x.mode().empty else 0)
                
                # Drop NaN values resulting from misalignment
                df_resampled = df_resampled.dropna()

                # Ensure that x_resampled and y_resampled align
                x_resampled = df_resampled.index
                y_resampled = df_resampled['Y']
            else:
                x_resampled = x
                y_resampled = y
            
            
            # Plot the data
            if plot_type == "line":
                ax.plot(x_resampled, y_resampled, label=label)
            elif plot_type == "bar":
                ax.bar(x_resampled, y_resampled, label=label)

            # Format the x-axis for timestamp-based data based on period_unit
            if isinstance(x_resampled[0], (datetime.datetime, pd.Timestamp)):
                if period_unit == "Y":
                    ax.xaxis.set_major_locator(mdates.YearLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                elif period_unit == "M":
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                elif period_unit == "W":
                    # Use WeekdayLocator with interval and byweekday matching current week
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                elif period_unit == "D":
                    ax.xaxis.set_major_locator(mdates.DayLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

                # Rotate x-tick labels
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha='center')
            else:
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha='center')
            
            if plot_type == "bar":
                ax.set_yticks([0, 1])  # Set the y-ticks at 0 and 1
                ax.set_yticklabels(["off", "on"])  # Map 0 to "off" and 1 to "on"

        ax.legend()

    plt.tight_layout()

    return fig
