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


def plot_graph_plotly(instruction, variables, response_function=None, return_html=False):
    axes = instruction.axes
    locals().update(variables)  
    num_axes = len(axes)
    
    # 각 subplot을 개별 figure로 생성하고 HTML로 결합
    all_html_divs = []
    
    for i, ax_info in enumerate(axes):
        # 각 ax마다 독립적인 figure 생성
        individual_fig = go.Figure()
        
        xlabel, ylabel = ax_info["description"]["xlabel"], ax_info["description"]["ylabel"]
        title = ax_info["description"]["title"]
        plot_type = ax_info["type"]

        for item in ax_info["items"]:
            label = item["label"]
            x = eval(item["x"])  
            y = eval(item["y"])

            if plot_type == "box":
                individual_fig.add_trace(
                    go.Box(
                        x=sum([[idu] * len(values) for idu, values in zip(x, y)], []),
                        y=pd.concat(y),
                        name=label,
                        boxmean=True,
                        boxpoints=False,  # outlier 점들 제거
                        marker=dict(opacity=0.6),
                        line=dict(width=1),
                        showlegend=True
                    )
                )

                individual_fig.update_xaxes(
                    title_text=xlabel,
                    type='category'
                )

                individual_fig.update_yaxes(
                    title_text=ylabel
                )
            else:
                # ✅ Box Plot이 아닐 경우 리샘플링 적용
                period_unit = ""
                if not pd.api.types.is_datetime64_any_dtype(x):
                    try:
                        x = pd.to_datetime(x)
                        if isinstance(x.iloc[0], (datetime.datetime, pd.Timestamp)):
                            date_range = max(x) - min(x)
                            if date_range.days > 365:
                                period_unit = "Y"
                            elif date_range.days > 30:
                                period_unit = "M"
                            elif date_range.days > 7:
                                period_unit = "W"
                            elif date_range.days > 0:
                                period_unit = "D"
                            else:
                                period_unit = "H"  
                            
                            period_unit_y = "D" if date_range.days > 0 else "H"
                            
                            df = pd.DataFrame({'Date': x, 'Y': y}).dropna()
                            df.set_index('Date', inplace=True)
                        
                            if plot_type == "marker":
                                df_resampled = df.resample(period_unit_y).apply(lambda x: x.mode()[0] if not x.mode().empty else 0)
                            else:
                                df_resampled = df.resample(period_unit_y).mean()
                    
                            df_resampled = df_resampled.dropna()
                            x_resampled = df_resampled.index
                            y_resampled = df_resampled['Y']
                    except Exception:
                        x_resampled = x
                        y_resampled = y

                # ✅ 리샘플링된 데이터로 그래프 추가
                if plot_type == "line":
                    # 색상과 선 스타일 속성 가져오기 (없을 경우 기본값 사용)
                    # color = item.get('color', '#1f77b4')  # 기본 파란색
                    # line_style = item.get('line_style', 'solid')  # 기본 실선
                    line = {}
                    if 'color' in item:
                        line['color'] = item['color']
                    if 'line_style' in item:
                        line['dash'] = item['line_style']

                    individual_fig.add_trace(
                        go.Scatter(
                            x=x_resampled, 
                            y=y_resampled, 
                            name=label,
                            line=line,
                            mode='lines'
                        )
                    )
                elif plot_type == "bar":
                    # 색상 속성 가져오기 (없을 경우 기본값 사용)
                    color = item.get('color', '#1f77b4')  # 기본 파란색
                    
                    individual_fig.add_trace(
                        go.Bar(
                            x=x_resampled, 
                            y=y_resampled, 
                            name=label, 
                            marker=dict(color=color)
                        )
                    )
                    individual_fig.update_xaxes(
                        title_text=xlabel,
                        type='category',  # 꼭 category로 설정
                        tickangle=-45
                    )
                elif plot_type == "marker":
                    # 색상 속성 가져오기 (없을 경우 기본값 사용)
                    color = item.get('color', '#1f77b4')  # 기본 파란색
                    
                    individual_fig.add_trace(
                        go.Scatter(
                            x=x_resampled, 
                            y=y_resampled, 
                            name=label,
                            mode='markers',
                            marker=dict(
                                size=12,  # 점 크기 조절
                                color=color,
                                symbol=['circle-open' if v == 0 else 'circle' for v in y_resampled]  # off=open circle, on=filled
                            )
                        )
                    )
                
                    individual_fig.update_yaxes(
                        tickvals=[0, 1],
                        ticktext=["off", "on"]
                    )

        # ✅ X, Y 축 업데이트
        individual_fig.update_xaxes(title_text=xlabel)
        individual_fig.update_yaxes(title_text=ylabel)
            
        if pd.api.types.is_datetime64_any_dtype(x):
            try:
                if isinstance(x.iloc[0], (datetime.datetime, pd.Timestamp)):
                    if period_unit == "Y":
                        individual_fig.update_xaxes(tickformat="%Y", dtick="M12")
                    elif period_unit == "M":
                        individual_fig.update_xaxes(tickformat="%Y-%m", dtick="M1")
                    elif period_unit == "W":
                        individual_fig.update_xaxes(tickformat="%Y-%m-%d", dtick="D7")
                    elif period_unit == "D":
                        individual_fig.update_xaxes(tickformat="%Y-%m-%d", dtick="D1")
                    elif period_unit == "H":
                        individual_fig.update_xaxes(tickformat="%H:%M", dtick="3600000")
            except Exception:
                pass
        
        # 개별 figure 레이아웃 설정
        individual_fig.update_layout(
            title=dict(
                text=title,
                y=0.95,  # 제목 위치 상단으로 이동
                font=dict(size=15)  # 제목 폰트 크기 증가
            ),
            height=500,  # 높이 증가 (legend를 위한 공간 확보)
            width=900 if plot_type == "line" else 450,
            margin=dict(t=70, b=70, l=70, r=120),  # 오른쪽 여백 증가
            template="plotly_white",
            legend=dict(
                orientation="v",  # 세로 방향 legend
                yanchor="top",
                y=1.0,           # 그래프 상단에 맞춤
                xanchor="right", 
                x=1.0,           # 그래프 오른쪽에 맞춤
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="rgba(150, 150, 150, 0.5)",
                borderwidth=1
            ),
            hovermode="closest"
        )
        
        # 각 figure를 HTML로 변환하여 저장
        fig_html = plotly_to_html_div(individual_fig)
        yield from response_function(fig_html, "graph")
        all_html_divs.append(fig_html)
    
    # 모든 HTML div를 하나로 결합
    # combined_html = "\n\n\n".join(all_html_divs)
    
    if return_html:
        return all_html_divs
    
    # HTML을 반환하지 않는 경우, 원래 방식대로 하나의 subplot figure 생성
    # 이렇게 하면 기존 인터페이스와 호환성 유지
    fig = sp.make_subplots(
        rows=num_axes, cols=1, 
        vertical_spacing=0.2,
        subplot_titles=[ax["description"]["title"] for ax in axes]
    )
    
    # 원래 코드에서 사용하던 방식으로 그래프 생성
    # (생략 - 이 부분은 실제로 사용되지 않음, return_html=True로 사용할 경우)
    
    return fig  # 이 부분은 return_html=False인 경우만 실행됨

import plotly.offline as pyo
def plotly_to_html_div(fig):
    config = {
        'responsive': True,  # 반응형 활성화
        'displayModeBar': False,  # 모드 바 표시
    }
    
    div_string = pyo.plot(fig, output_type='div', include_plotlyjs=False, config=config)
    
    return div_string