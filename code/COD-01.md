---
layout: visualization
permalink: /cod-01s
---

```python
# !jupyter nbconvert --to html --TemplateExporter.exclude_input=True "D:/Python_Pruebas/Notebooks/Processum/SUF_VIH_NT_15_18_PLOTS.ipynb"
```


```python
from plotly.graph_objs import Scatter, Figure, Layout, Choropleth, Bar, Heatmap
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import pandas as pd
import numpy as np
import json
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np


# Visualizar el notebook
init_notebook_mode(connected=True)
```

# Nota Técnica VIH - Análisis por Agrupador
**Fuente:** BD Suficiencia  2015 - 2018


```python
df_grupo = pd.read_csv("D:/PROCESSUM/Analisis/Analisis_Ext/2_Resultados/VIH/15_18/NT_VIH_15_18/00_VIH_15_18_NotaTec_grupo.csv000"
                       , sep=';')
# df_grupo.dtypes
```


```python
df_grupo['agrupador'] = df_grupo['agrupador'].str.capitalize()
# df_grupo.head()
```


```python
# Generar HTML
def print_div(x):
    r = x.replace("\\u00e1","á").replace("\\u00e9","é").replace("\\u00ed","í").replace("\\u00f3","ó").replace("\\u00fa","ú").replace("\\u00c1","Á").replace("\\u00c9","É").replace("\\u00cd","Í").replace("\\u00d3","Ó").replace("\\u00da","Ú").replace("\\u00f1","ñ").replace("\\u00d1","Ñ")
    print(r)

```


```python
def generar_grafico(df_aux, var_y,columna_x1, columna_x2):
    datos = df_aux.copy().sort_values(by=[columna_x1, columna_x2], ascending = False)
    max_p = datos[columna_x1].max()

    sns.set(style="whitegrid")
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(18, 10))
    # Plot the total crashes
    sns.set_color_codes("pastel")
    sns.barplot(x=columna_x1, y= var_y
                , data=datos,
                label=columna_x1, color="b")

    sns.set_color_codes("pastel")
    sns.barplot(x=columna_x2, y= var_y
                , data=datos,
                label=columna_x2, color="g")

    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, max_p), ylabel="",
           xlabel="Top 15 ")
    sns.despine(left=True, bottom=True)
#     plt.title("APMES "+cod_eps+" - Top 15 "+title_g+"\nEn Ambulatorio, excluyendo No aprobados")
#     ruta_s = 'D:/PROCESSUM/Analisis/Analisis_Ext/2_Resultados/APMES/'+cod_eps+'/'+cod_eps
#     plt.savefig(ruta_s+"_"+var_y+""+title_g+".png",bbox_inches='tight')
```


```python
def grafico_2sub(df_aux, columna_y,columna_x1, columna_x2,titulo,nombre_g1,nombre_g2):
    df = df_aux.copy().sort_values(by=[columna_x2, columna_x1],ascending =[False,False])
    y_saving = df[columna_x2]
    y_net_worth = df[columna_x1]
    x = df[columna_y]


    # Creating two subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                        shared_yaxes=False, vertical_spacing=0.001)

    fig.append_trace(go.Bar(
        x=y_saving,
        y=x,
        marker=dict(
            color='rgba(50, 171, 96, 0.6)',
            line=dict(
                color='rgba(50, 171, 96, 1.0)',
                width=1),
        ),
        name= nombre_g2,
        orientation='h',
    ), 1, 1)

    fig.append_trace(go.Scatter(
        x=y_net_worth, y=x,
        mode='lines+markers',
        line_color='rgb(128, 0, 128)',
        name=nombre_g1,
    ), 1, 2)

    fig.update_layout(
        title=titulo,
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
             domain=[0, 0.85],
        ),
        yaxis2=dict(
            showgrid=False,
            showline=True,
            showticklabels=False,
            linecolor='rgba(102, 102, 102, 0.8)',
            linewidth=2,
             domain=[0, 0.85],
        ),
        xaxis=dict(
            zeroline=False,
            showline=False,
            showticklabels=True,
            showgrid=True,
#             domain=[0, 0.42],
        ),
        xaxis2=dict(
            zeroline=False,
            showline=False,
            showticklabels=True,
            showgrid=True,
#             domain=[0.47, 1],
            side='top',
#             dtick=25000,
        ),
        legend=dict(x=0.029, y=1.038, font_size=10),
        margin=dict(l=100, r=20, t=70, b=70),
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
    )

    annotations = []

    y_s = np.round(y_saving, decimals=0)
    y_nw = np.rint(y_net_worth)

    # Adding labels
#     for ydn, yd, xd in zip(y_nw, y_s, x):
#         # labeling the scatter savings
#         annotations.append(dict(xref='x2', yref='y2',
#                                 y=xd, x=ydn - 20000,
#                                 text='{:,}'.format(ydn) + 'M',
#                                 font=dict(family='Arial', size=12,
#                                           color='rgb(128, 0, 128)'),
#                                 showarrow=False))
        # labeling the bar net worth
#         annotations.append(dict(xref='x1', yref='y1',
#                                 y=xd, x=yd + 3,
#                                 text=str(yd) + '%',
#                                 font=dict(family='Arial', size=12,
#                                           color='rgb(50, 171, 96)'),
#                                 showarrow=False))
    # Source
#     annotations.append(dict(xref='paper', yref='paper',
#                             x=-0.2, y=-0.109,
#                             text='OECD "' +
#                                  '(2015), Household savings (indicator), ' +
#                                  'Household net worth (indicator). doi: ' +
#                                  '10.1787/cfc6f499-en (Accessed on 05 June 2015)',
#                             font=dict(family='Arial', size=10, color='rgb(150,150,150)'),
#                             showarrow=False))

    fig.update_layout(annotations=annotations)
    print_div(fig.show())
#     fig.show()
```


```python
import plotly.graph_objects as go

def crear_gf_barras2(df_aux, columna_y,columna_x1, columna_x2):
    df = df_aux.copy().sort_values(by=[columna_x1, columna_x2])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df[columna_y],
        x=df[columna_x2],
        name=columna_x2,
        orientation='h',
        marker=dict(
            color='rgba(246, 78, 139, 0.6)',
            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
        )
    ))
    fig.add_trace(go.Bar(
        y=df[columna_y],
        x=df[columna_x1],
        name=columna_x1,
        orientation='h',
        marker=dict(
            color='rgba(58, 71, 80, 0.6)',
            line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
        )
    ))

#     fig.update_layout(barmode='stack')
    #Comando para exportar a HTML
#     print_div(plot(fig, include_plotlyjs=False, output_type='div'))
    fig.show()
```


```python
from plotly.subplots import make_subplots

def horizontal_bar_labels(categories):
    subplots = make_subplots(
        rows=len(categories),
        cols=1,
        subplot_titles=[x["name"] for x in categories],
        shared_xaxes=True,
        print_grid=False,
        vertical_spacing=(0.45 / len(categories)),
    )
    subplots['layout'].update(
        width=550,
        plot_bgcolor='#fff',
    )

    # add bars for the categories
    for k, x in enumerate(categories):
        subplots.add_trace(dict(
            type='bar',
            orientation='h',
            y=[x["name"]],
            x=[x["value"]],
            text=["{:,.0f}".format(x["value"])],
            hoverinfo='text',
            textposition='auto',
            marker=dict(
                color="#7030a0",
            ),
        ), k+1, 1)

    # update the layout
    subplots['layout'].update(
        showlegend=False,
    )
    for x in subplots["layout"]['annotations']:
        x['x'] = 0
        x['xanchor'] = 'left'
        x['align'] = 'left'
        x['font'] = dict(
            size=12,
        )

    # update the margins and size
    subplots['layout']['margin'] = {
        'l': 0,
        'r': 0,
        't': 20,
        'b': 1,
    }
    height_calc = 45 * len(categories)
    height_calc = max([height_calc, 350])
    subplots['layout']['height'] = height_calc
    subplots['layout']['width'] = height_calc

    return subplots
```

### Consumos por grupo de nota técnica en pacientes con VIH


```python
grafico_2sub(df_grupo, 'agrupador','costo_total','nro_personas_vih'
             ,titulo ="Número de Personas y Costo asociados a pacientes con VIH" 
            ,nombre_g1 = "Costo"
            ,nombre_g2= "Personas")
```

### Personas y Costo con registro de CIE10 de VIH


```python
grafico_2sub(df_grupo, 'agrupador','costo_total_vih','nro_personas_consumo_cie10_vih'
             ,titulo ="Número de Personas y Costo relacionado con CIE10 de VIH" 
            ,nombre_g1 = " Costo asociado a CIE10 de VIH"
            ,nombre_g2= "Personas con registro de CIE10 de VIH")
```

##  Costo Percapita


```python
import plotly.graph_objects as go

def grafico_barra_linea(df_aux, columna_y,columna_x1, columna_x2,titulo,nombre_g1,nombre_g2):
    df = df_aux.copy().sort_values(by=[columna_x1, columna_x2])
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[columna_y],
            y=df[columna_x1],
         mode='lines+markers',
        line_color='rgb(128, 0, 128)',
        name=nombre_g1,
        ))

    fig.add_trace(
        go.Bar(
            x=df[columna_y],
            y=df[columna_x2],name=nombre_g2,
#         orientation='h',
        marker=dict(
            color='rgba(0, 178, 169, 0.6)',
            line=dict(color='rgba(0, 178, 169, 1.0)', width=3)
        )
        ))
    fig.update_layout(
        title=titulo,legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))

    fig.show()
```


```python
grafico_barra_linea(df_grupo, 'agrupador','costo_percapita','costo_percapita_cie10_vih'
             ,titulo ="Comparación de Costo Pércapita" 
            ,nombre_g1 = "Costo Pércapita en pacientes con VIH"
            ,nombre_g2= "Costo Pércapita asociado a CIE10 VIH")
```

### Costo y Número de Personas por Agrupador


```python
df_grupo_eps = pd.read_csv("D:/PROCESSUM/Analisis/Analisis_Ext/2_Resultados/VIH/15_18/NT_VIH_15_18/02_VIH_15_18_EPS_NotaTec_grupo.csv000"
                       , sep=';')

df_grupo_eps['agrupador'] = df_grupo_eps['agrupador'].str.capitalize()
```


```python
for n_eps in  df_grupo_eps['eps'].unique():
    list(df[df['eps']== 'EPS017']['costo_total'])
```


```python
import plotly.express as px
df = df_grupo_eps.copy()

fig = px.scatter(df, x="nro_personas_vih", y="costo_total",
	         size="costo_percapita", color="eps",
                 hover_name="agrupador", log_x=False, size_max=58)

fig.update_layout(
        title="Costo y # de Personas por EPS con pacientes de VIH "
    ,legend=dict(
    yanchor="top",
#     y=0.99,
    xanchor="left",
#     x=0.01
))
    
fig.show()
```


```python
import plotly.express as px
df = df_grupo_eps.copy()

fig = px.scatter(df, x="nro_personas_vih", y="costo_total",
	         size="costo_percapita", color="eps",
                 hover_name="agrupador", log_x=True, size_max=58)

fig.update_layout(
        title="Costo y # de Personas por EPS con pacientes de VIH - Escala Logaritmica"
    ,legend=dict(
    yanchor="top",
#     y=0.99,
    xanchor="left",
#     x=0.01
))
    
fig.show()
```


```python
import plotly.express as px
df = df_grupo_eps.copy()

fig = px.scatter(df, x="nro_personas_vih", y="costo_total",
	         size="costo_percapita", color="agrupador",
                 hover_name="eps", log_x=True, size_max=58)

fig.update_layout(
        title="Costo y # de Personas por agrupador y EPS con pacientes de VIH - Escala Logaritmica"
    ,legend=dict(
    yanchor="top",
#     y=0.99,
    xanchor="left",
#     x=0.01
))
    
fig.show()
```


```python

```


```python
def grafico_agrupado1(df_aux, columna_y,columna_x1, columna_x2,titulo,nombre_g1,nombre_g2):
    df = df_aux.copy().sort_values(by=[columna_x1, columna_x2])
    x=df[columna_y].unique()
    fig = go.Figure(go.Bar(x=x, y=[2,5,1,9], name='Montreal'))
    
    fig.add_trace(go.Bar(x=x, y=[1, 4, 9, 16], name='Ottawa'))
    fig.add_trace(go.Bar(x=x, y=[6, 8, 4.5, 8], name='Toronto'))

    fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})
    fig.show()
```


```python
# grafico_agrupado1(df_grupo_eps, 'agrupador','costo_percapita','costo_percapita_cie10_vih'
#              ,titulo ="Comparación de Costo Pércapita" 
#             ,nombre_g1 = "Costo Pércapita en pacientes con VIH"
#             ,nombre_g2= "Costo Pércapita asociado a CIE10 VIH")
```


```python
# # dicc_eps =
# # for n_eps in  df_grupo_eps['eps'].unique():
# #     n_eps
# df_grupo_eps['eps'].unique()
```


```python
# list(df_grupo_eps[df_grupo_eps['eps']== 'EPS017']['costo_total'])
```


```python
import plotly.graph_objects as go


def grafico_grupo(df_aux, columna_y,columna_x1, columna_x2,titulo,nombre_g1,nombre_g2):
    df = df_aux.copy().sort_values(by=[columna_x1, columna_x2])


    labels = df[columna_y]
    widths = np.array([10,20,20,50])

    data = {
        "EPS017": list(df[df['eps']== 'EPS017']['costo_total']),
        "EPS023": list(df[df['eps']== 'EPS023']['costo_total'])
    }

    fig = go.Figure()
    for key in data:
        fig.add_trace(go.Bar(
            name=key,
            y=data[key],
            x=np.cumsum(widths)-widths,
            width=widths,
            offset=0,
            customdata=np.transpose([labels, widths*data[key]]),
            texttemplate="%{y} x %{width} =<br>%{customdata[1]}",
            textposition="inside",
            textangle=0,
            textfont_color="white",
            hovertemplate="<br>".join([
                "label: %{customdata[0]}",
                "width: %{width}",
                "height: %{y}",
                "area: %{customdata[1]}",
            ])
        ))

    fig.update_xaxes( 
        tickvals=np.cumsum(widths)-widths/2, 
        ticktext= ["%s<br>%d" % (l, w) for l, w in zip(labels, widths)]
    )

    fig.update_xaxes(range=[0,100])
    fig.update_yaxes(range=[0,100])

    fig.update_layout(
        title_text="Marimekko Chart",
        barmode="stack",
        uniformtext=dict(mode="hide", minsize=10),
    )
    fig.show()
```


```python
# grafico_grupo(df_grupo_eps, 'agrupador','costo_percapita','costo_percapita_cie10_vih'
#              ,titulo ="Comparación de Costo Pércapita" 
#             ,nombre_g1 = "Costo Pércapita en pacientes con VIH"
#             ,nombre_g2= "Costo Pércapita asociado a CIE10 VIH")
```


```python

```


```python
# !jupyter nbconvert --to html --TemplateExporter.exclude_input=True "D:/Python_Pruebas/Notebooks/Processum/SUF_VIH_NT_15_18_PLOTS.ipynb"
```
