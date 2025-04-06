__all__ = ["plot_graph", "plot_graph_plotly"]

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

import io
import base64

import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import datetime


def plot_graph(instruction, variables, return_html=False):
    axes = instruction.axes
    
    # Update the local variables with any additional variables
    locals().update(variables)

    # Create subplots based on the number of axes
    num_axes = len(axes)
    fig, axes = plt.subplots(num_axes, figsize=(10, 6 * num_axes), layout="constrained")
    
    # Ensure axes is iterable
    if num_axes == 1:
        axes = [axes]
    
    # Loop through each axis and plot accordingly
    for i, ax in enumerate(axes):
        xlabel, ylabel, title = axes[i]["description"]["xlabel"], axes[i]["description"]["ylabel"], axes[i]["description"]["title"]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Iterate through the plots for this axis
        for item in axes[i]["items"]:
            plot_type = axes[i]["type"]
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

    # plt.tight_layout()

    if return_html:
        return matplotlib_to_html(fig)
    return fig

def matplotlib_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    html = base64.b64encode(buf.read()).decode('utf-8')
    html = f'<img src="data:image/png;base64,{html}"/>'
    return html

def plot_graph_plotly(instruction, variables, return_html=False):
    
    axes = instruction.axes
    
    # Update the local variables with any additional variables
    locals().update(variables)

    # Create subplots based on the number of axes
    num_axes = len(axes)
    
    # Create subplot figure
    fig = sp.make_subplots(rows=num_axes, cols=1, 
                          vertical_spacing=0.1,
                          subplot_titles=[ax["description"]["title"] for ax in axes])
    
    
    # Loop through each axis and plot accordingly
    for i, ax_info in enumerate(axes):
        row = i + 1  # Plotly rows are 1-indexed
        
        # Extract axis descriptions
        xlabel, ylabel = ax_info["description"]["xlabel"], ax_info["description"]["ylabel"]
        
        # Iterate through the plots for this axis
        for item in ax_info["items"]:
            plot_type = ax_info["type"]
            label = item["label"]
            x = eval(item["x"])  # Evaluate the expression for x-axis data
            y = eval(item["y"])  # Evaluate the expression for y-axis data
            
            period_unit = "D"
            
            # Handle datetime conversion
            if not isinstance(x, pd.Series):
                x = pd.Series(x)
            
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
                fig.add_trace(
                    go.Scatter(
                        x=x_resampled, 
                        y=y_resampled, 
                        name=label,
                        mode='lines'
                    ),
                    row=row, col=1
                )
            elif plot_type == "bar":
                fig.add_trace(
                    go.Scatter(
                        x=x_resampled, 
                        y=y_resampled, 
                        name=label,
                        mode='markers',
                        marker=dict(
                            size=12,  # 점 크기 조절
                            #color=['red' if v == 0 else 'green' for v in y_resampled],  # off=red, on=green
                            symbol=['circle-open' if v == 0 else 'circle' for v in y_resampled]  # off=open circle, on=filled
                        )
                    ),
                    row=row, col=1
                )
            
            # Format axes for bar charts showing on/off status
            if plot_type == "bar":
                fig.update_yaxes(
                tickvals=[0, 1],
                ticktext=["off", "on"],
                row=row, col=1
                )
        
        # Update axis labels
        fig.update_xaxes(title_text=xlabel, row=row, col=1)
        fig.update_yaxes(title_text=ylabel, row=row, col=1)
        
        # Format the x-axis ticks based on the period unit (for datetime data)
        if isinstance(x.iloc[0], (datetime.datetime, pd.Timestamp)):
            if period_unit == "Y":
                fig.update_xaxes(
                    tickformat="%Y",
                    dtick="M12",
                    row=row, col=1
                )
            elif period_unit == "M":
                fig.update_xaxes(
                    tickformat="%Y-%m",
                    dtick="M1",
                    row=row, col=1
                )
            elif period_unit == "W":
                fig.update_xaxes(
                    tickformat="%Y-%m-%d",
                    dtick="D7",
                    row=row, col=1
                )
            elif period_unit == "D":
                fig.update_xaxes(
                    tickformat="%Y-%m-%d",
                    dtick="D1",
                    row=row, col=1
                )
    
    # # Update layout for better appearance
    # height_per_subplot = 100

    # fig.update_layout(
    #     height=height_per_subplot * num_axes,
    #     width=300,
    #     showlegend=True,
    #     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    #     margin=dict(t=50, b=50, l=50, r=50),
    #     hovermode="x unified"
    # )


    fig.update_layout(
        height=400,
        width=600,  # 기본 너비 설정 (responsive로 오버라이드됨)
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=50, l=50, r=50),
        hovermode="x unified",
        font=dict(
            family="Pretendard, sans-serif",  # Pretendard 글꼴 설정
        ),
    )

    # # 반응형 레이아웃 설정
    # fig.update_layout(
    #     autosize=True,  # 자동 크기 조정 활성화
    # )
    
    if return_html:
        return plotly_to_html_div(fig)
    return fig

import plotly.offline as pyo
def plotly_to_html_div(fig):
    # div_string = pyo.plot(fig, output_type='div', include_plotlyjs=False)
    config = {
        'responsive': True,  # 반응형 활성화
        'displayModeBar': True,  # 모드 바 숨기기 (선택사항)
    }
    
    div_string = pyo.plot(fig, output_type='div', include_plotlyjs=False, config=config)
    
    # CSS를 사용하여 부모 컨테이너에 맞추기 위한 스타일 추가
    # styled_div = div_string.replace('<div ', '<div style="width:100%;height:100%;" ')

    # with open('plotly_div.html', 'w') as f:
    #     f.write(div_string)
    return div_string