from collections.abc import Sequence
from typing import Literal

import numpy as np
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from plotly.validator_cache import ValidatorCache

from .comparison import AnnotationComparison


def hex_to_rgba(hex_color: str, alpha: float = 0.5) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_comparison(
    metrics: Sequence[AnnotationComparison],
    labels: Sequence[str] | None = None,
    zoom: bool = True,
    flip_axes: bool = False,
    missing_novel: bool = False,
    table: bool = False,
    f1_curve: bool = True,
) -> go.Figure:
    keys = ["base", "intron", "transcript", "exon", "intron_chain", "locus"]
    fig = make_subplots(
        rows=2 + (1 if missing_novel else 0) + (1 if table else 0),
        cols=3,
        subplot_titles=[
            key[0].upper()+key[1:].replace("_", "-") for key in keys
        ]
        + (
            ["Percentage of novel and missing features"]
            if missing_novel else []
        )
        + (
            ["Metrics (Sensitivity | Precision (F1))"] if table else []
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
        ]
        + ([
            [{'type': 'bar', 'colspan': 3}, None, None],
        ] if missing_novel else [])
        + ([
            [{'type': 'table', 'colspan': 3}, None, None],
        ] if table else []),
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(metrics))]

    SymbolValidator = ValidatorCache.get_validator("scatter.marker", "symbol")
    raw_symbols = SymbolValidator.values[2:19*12:12]
    while len(metrics) > len(raw_symbols):
        raw_symbols += SymbolValidator.values[2:19*12:12]
    colors = px.colors.qualitative.Plotly
    while len(metrics) > len(colors):
        colors += px.colors.qualitative.Plotly

    if table:
        table_rows = [[] for _ in range(len(keys))]
        color_cols = [[] for _ in range(len(keys))]

    bnds = {}
    # bnds[key] = [sens lower, sens upper, prec lower, prec upper]
    for i, metric in enumerate(metrics):
        for j, key in enumerate(keys):
            if i == 0:
                bnds[key] = [
                    getattr(metric, key).sensitivity,
                    getattr(metric, key).sensitivity,
                    getattr(metric, key).precision,
                    getattr(metric, key).precision,
                ]
            else:
                bnds[key] = [
                    min(bnds.get(key)[0], getattr(metric, key).sensitivity),
                    max(bnds.get(key)[1], getattr(metric, key).sensitivity),
                    min(bnds.get(key)[2], getattr(metric, key).precision),
                    max(bnds.get(key)[3], getattr(metric, key).precision),
                ]
            row = j // 3 + 1
            col = j % 3 + 1
            x, y, f1 = (
                getattr(metric, key).sensitivity,
                getattr(metric, key).precision,
                getattr(metric, key).F1,
            )
            if table: table_rows[j].append(
                f"<b>{x:.3f} | {y:.3f}<br>    ({f1:.3f})</b>"
            )
            if table: color_cols[j].append(hex_to_rgba(colors[i], alpha=0.5))
            if flip_axes:
                x, y = y, x

            hover_text = [
                f"{key[0].upper()+key[1:].replace('_', '-')}"
                f"<br>Sensitivity: {x:.3f}<br>Precision: {y:.3f}"
                f"<br>F1: {f1:.3f}"
            ]
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                name=labels[i],
                marker=dict(
                    symbol=raw_symbols[i],
                    color=colors[i],
                    size=8,
                ),
                showlegend=j==0,
                legendgroup=f"group {i}",
                text=hover_text,
                hoverinfo="text"
            ), row=row, col=col)

            if f1_curve:
                P = np.linspace(f1/2, 1, 100)
                denom = 2*P - f1
                R = np.full_like(P, np.nan)
                valid = denom > 0
                R[valid] = f1 * P[valid] / denom[valid]
                inside = (P >= 0) & (P <= 1) & (R >= 0) & (R <= 1)
                fig.add_trace(go.Scatter(
                    x=P[inside],
                    y=R[inside],
                    name=labels[i],
                    mode="lines",
                    line=dict(
                        color=colors[i],
                        dash="dot",
                        width=1,
                    ),
                    legendgroup=f"group {i}",
                    hoverinfo="skip",
                    showlegend=False,
                ), row=row, col=col)

            missed = getattr(getattr(metric, key), "missed", None)
            novel = getattr(getattr(metric, key), "novel", None)
            if missing_novel and missed is not None and novel is not None:
                fig.add_trace(go.Bar(
                    x=[key, key],
                    y=[-missed, novel],
                    name=labels[i],
                    marker_color=colors[i],
                    showlegend=False,
                    legendgroup=f"group {i}",
                ), row=3, col=1)

    if table:
        fig.add_trace(go.Table(
            header=dict(
                values=[
                    key[0].upper()+key[1:].replace('_', '-') for key in keys
                ],
                align="center",
                font=dict(size=14),
            ),
            cells=dict(
                values=table_rows,
                fill_color=color_cols,
                align="center",
            ),
        ), row=4 if missing_novel else 3, col=1)


    for j, key in enumerate(keys):
        row = j // 3 + 1
        col = j % 3 + 1
        bound_x = [
            max(0, bnds[key][0 if not flip_axes else 2]-0.1),
            min(1, bnds[key][1 if not flip_axes else 3]+0.1),
        ]
        bound_y = [
            max(0, bnds[key][2 if not flip_axes else 0]-0.1),
            min(1, bnds[key][3 if not flip_axes else 1]+0.1),
        ]
        bounds = [min(bound_x[0], bound_y[0]), max(bound_x[1], bound_y[1])]
        fig.update_xaxes(
            range=[0, 1] if not zoom else bounds,
            scaleanchor=f'y{j+1}',
            constrain='domain',
            row=row,
            col=col,
            dtick=.1 if not zoom else None,
            minor=dict(
                dtick=0.02,
                showgrid=True,
                gridcolor="rgba(0,0,0,0.05)",
            ),
        )
        fig.update_yaxes(
            range=[0, 1] if not zoom else bounds,
            constrain='domain',
            row=row,
            col=col,
            dtick=.1 if not zoom else None,
            minor=dict(
                dtick=0.02,
                showgrid=True,
                gridcolor="rgba(0,0,0,0.05)",
            ),
        )
        axis_id = "" if j == 0 else str(j+1)
        fig.add_annotation(
            text="Sensitivity" if not flip_axes else "Precision",
            xref=f"x{axis_id} domain", yref=f"y{axis_id} domain",
            x=1.0, y=0,
            showarrow=False,
            font=dict(size=14),
            opacity=0.5,
            xanchor="right", yanchor="bottom",
        )
        fig.add_annotation(
            text="Precision" if not flip_axes else "Sensitivity",
            xref=f"x{axis_id} domain", yref=f"y{axis_id} domain",
            x=0, y=1.0,
            showarrow=False,
            font=dict(size=14),
            opacity=0.5,
            textangle=-90,
            xanchor="left", yanchor="top",
        )

    if missing_novel:
        fig.add_annotation(
            text="Novel",
            xref="paper", yref="y7",
            x=0.05, y=0.5,
            showarrow=False,
            font=dict(size=14),
            opacity=0.5,
            textangle=-90,
            xanchor="left", yanchor="middle",
        )
        fig.add_annotation(
            text="Missed",
            xref="paper", yref="y7",
            x=0.05, y=-0.5,
            showarrow=False,
            font=dict(size=14),
            opacity=0.5,
            textangle=-90,
            xanchor="left", yanchor="middle",
        )

    fig.update_layout(
        width=1200,
        height=600
               + (300 if missing_novel else 0)
               + (200+50*len(metrics) if table else 0),
        margin=dict(l=30, r=0, t=30, b=30),
        title=dict(xref="container", yref="container", yanchor="bottom"),
        xaxis7=dict(domain=[0.05, 0.95]),
        yaxis7=dict(
            tickmode='array',
            tickvals=(
                [-i/10 for i in range(1, 11)] + [i/10 for i in range(1, 11)]
            ),
            ticktext=[str(i/10) for i in range(1, 11)],
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
        ),
    )
    return fig


def plot_comparison_changes(
    metrics: Sequence[Sequence[AnnotationComparison]],
    label_species: Sequence[str] | None = None,
    label_tools: Sequence[str] | None = None,
    zoom: bool = True,
    flip_axes: bool = False,
    f1_curve: bool = True,
    include: list[Literal[
        "base", "intron", "transcript", "exon", "intron_chain", "locus"
    ]] | None = None,
) -> go.Figure:
    keys = include if include is not None else [
        "base", "intron", "transcript", "exon", "intron_chain", "locus"
    ]
    fig = make_subplots(
        rows=1,
        cols=len(keys),
        subplot_titles=[
            key[0].upper()+key[1:].replace("_", "-") for key in keys
        ],
        specs=[len(keys) * [{'type': 'scatter'}]],
        horizontal_spacing=0.05,
    )
    if label_species is None:
        label_species = [f"Species {i+1}" for i in range(len(metrics))]
    if label_tools is None:
        label_tools = [f"Model {i+1}" for i in range(len(metrics[0]))]

    SymbolValidator = ValidatorCache.get_validator("scatter.marker", "symbol")
    raw_symbols = SymbolValidator.values[2:19*12:12]
    while len(metrics) > len(raw_symbols):
        raw_symbols += SymbolValidator.values[2:19*12:12]
    colors = px.colors.qualitative.Plotly
    while len(metrics) > len(colors):
        colors += px.colors.qualitative.Plotly

    bnd = {}
    # bnds[key] = [sens lower, sens upper, prec lower, prec upper]
    for k, group in enumerate(metrics):
        for i, metric in enumerate(group):
            for j, key in enumerate(keys):
                if k == 0 and i == 0:
                    bnd[key] = [
                        getattr(metric, key).sensitivity,
                        getattr(metric, key).sensitivity,
                        getattr(metric, key).precision,
                        getattr(metric, key).precision,
                    ]
                else:
                    bnd[key] = [
                        min(bnd.get(key)[0], getattr(metric, key).sensitivity),
                        max(bnd.get(key)[1], getattr(metric, key).sensitivity),
                        min(bnd.get(key)[2], getattr(metric, key).precision),
                        max(bnd.get(key)[3], getattr(metric, key).precision),
                    ]
                row = 1
                col = j + 1
                x, y, f1 = (
                    getattr(metric, key).sensitivity,
                    getattr(metric, key).precision,
                    getattr(metric, key).F1,
                )
                if flip_axes:
                    x, y = y, x

                hover_text = [
                    f"{label_species[k]}<br>{label_tools[i]}"
                    f"<br>Sensitivity: {x:.3f}<br>Precision: {y:.3f}"
                    f"<br>F1: {f1:.3f}"
                ]
                fig.add_trace(go.Scatter(
                    x=[x],
                    y=[y],
                    name=f"{label_species[k]} - {label_tools[i]}",
                    marker=dict(
                        symbol=raw_symbols[i],
                        color=colors[k],
                        size=8,
                    ),
                    showlegend=False,
                    legendgroup=f"group {k}",
                    text=hover_text,
                    hoverinfo="text"
                ), row=row, col=col)

                if f1_curve:
                    P = np.linspace(f1/2, 1, 100)
                    denom = 2*P - f1
                    R = np.full_like(P, np.nan)
                    valid = denom > 0
                    R[valid] = f1 * P[valid] / denom[valid]
                    inside = (P >= 0) & (P <= 1) & (R >= 0) & (R <= 1)
                    fig.add_trace(go.Scatter(
                        x=P[inside],
                        y=R[inside],
                        name=f"{label_species[k]} - {label_tools[i]}",
                        mode="lines",
                        line=dict(
                            color=colors[k],
                            dash="dot",
                            width=1,
                        ),
                        legendgroup=f"group {k}",
                        hoverinfo="skip",
                        showlegend=False,
                    ), row=row, col=col)

    for k, group in enumerate(metrics):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            name=label_species[k],
            mode="lines",
            line=dict(color=colors[k], width=12),
            showlegend=True,
            legendgroup=f"group",
            legendgrouptitle_text="Species",
        ), row=row, col=col)
    for i, metric in enumerate(metrics[0]):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            name=label_tools[i],
            mode="markers",
            marker=dict(
                symbol=raw_symbols[i],
                color="black",
                size=12,
            ),
            showlegend=True,
            legendgroup=f"metric",
            legendgrouptitle_text="Tool",
        ), row=row, col=col)

    for j, key in enumerate(keys):
        row = 1
        col = j + 1
        bound_x = [
            max(0, bnd[key][0 if not flip_axes else 2]-0.1),
            min(1, bnd[key][1 if not flip_axes else 3]+0.1),
        ]
        bound_y = [
            max(0, bnd[key][2 if not flip_axes else 0]-0.1),
            min(1, bnd[key][3 if not flip_axes else 1]+0.1),
        ]
        bounds = [min(bound_x[0], bound_y[0]), max(bound_x[1], bound_y[1])]
        fig.update_xaxes(
            range=[0, 1] if not zoom else bounds,
            scaleanchor=f'y{j+1}',
            constrain='domain',
            row=row,
            col=col,
            dtick=.1 if not zoom else None,
            minor=dict(
                dtick=0.02,
                showgrid=True,
                gridcolor="rgba(0,0,0,0.05)",
            ),
        )
        fig.update_yaxes(
            range=[0, 1] if not zoom else bounds,
            constrain='domain',
            row=row,
            col=col,
            dtick=.1 if not zoom else None,
            minor=dict(
                dtick=0.02,
                showgrid=True,
                gridcolor="rgba(0,0,0,0.05)",
            ),
        )
        axis_id = "" if j == 0 else str(j+1)
        fig.add_annotation(
            text="Sensitivity" if not flip_axes else "Precision",
            xref=f"x{axis_id} domain", yref=f"y{axis_id} domain",
            x=1.0, y=0,
            showarrow=False,
            font=dict(size=14),
            opacity=0.5,
            xanchor="right", yanchor="bottom",
        )
        fig.add_annotation(
            text="Precision" if not flip_axes else "Sensitivity",
            xref=f"x{axis_id} domain", yref=f"y{axis_id} domain",
            x=0, y=1.0,
            showarrow=False,
            font=dict(size=14),
            opacity=0.5,
            textangle=-90,
            xanchor="left", yanchor="top",
        )

    fig.update_layout(
        width=len(keys)*300+300,
        height=300,
        margin=dict(l=30, r=0, t=30, b=30),
        title=dict(xref="container", yref="container", yanchor="bottom"),
    )
    return fig
