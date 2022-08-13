from sys import flags
import streamlit as st
#import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly as py
import matplotlib.pyplot as plt 
from statsmodels.graphics.gofplots import qqplot

def remove_outlier(dataset_in, col_name, value_alpha):
    q1 = dataset_in[col_name].quantile(0.25)
    q3 = dataset_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    lower_limit  = q1-value_alpha*iqr
    upper_limit = q3+value_alpha*iqr
    dataset_out = dataset_in.loc[(dataset_in[col_name]> lower_limit)&(dataset_in[col_name]< upper_limit)]
    return dataset_out

def generate_graphics(dataset, column, graph, tittle_graph):
    if graph == 'Histograma':
        fig = px.histogram(dataset, x=column, title=tittle_graph)
        st.plotly_chart(fig)
    elif graph == "Cajas y Bigotes":
        fig = px.box(dataset, x=column, title=tittle_graph)
        st.plotly_chart(fig)
    elif graph == "Regresion Lineal":
        pass

    else:
        qqplot_data = qqplot(dataset[column], line='s').gca().lines

        fig = go.Figure()

        fig.add_trace({
            'type': 'scatter',
            'x': qqplot_data[0].get_xdata(),
            'y': qqplot_data[0].get_ydata(),
            'mode': 'markers',
            'marker': {
                'color': '#19d3f3'
            }
        })

        fig.add_trace({
            'type': 'scatter',
            'x': qqplot_data[1].get_xdata(),
            'y': qqplot_data[1].get_ydata(),
            'mode': 'lines',
            'line': {
                'color': '#636efa'
            }
        })

        fig['layout'].update({
            'title': tittle_graph,
            'xaxis': {
                'title': 'Theoritical Quantities',
                'zeroline': False
            },
            'yaxis': {
                'title': 'Sample Quantities'
            },
            'showlegend': False,
        })
        st.plotly_chart(fig)
            
def graph_two_columns(dataset, column_input, column_output, graph, tittle_graph):
    if graph == 'Dispersión':
        fig = px.scatter(dataset, x=column_input, y=column_output, title=tittle_graph)
        st.plotly_chart(fig)
    # elif graph == "Cajas y Bigotes":
    #     fig = px.box(dataset, x=column, title=tittle_graph)
    #     st.plotly_chart(fig)

def graph_regression(options_column, column):
    st.warning("Recuerda que para usar **Regresión Lineal** debes incluir al menos una variable de entrada y sólo una de salida")
    col3, col4 = st.columns(2)

    inputs_selection = col3.multiselect('Variable(s) de entrada:', options_column, default=column)

    column_input = ['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
    column_to_compare = col4.selectbox("Variable de salida", column_input)




def user_input():
    col1, col2 = st.columns(2)

    options_column = ['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
    column = col1.selectbox("¿Sobre cuál columna quieres ver las gráficas?", options_column)

    options_graph = ["Histograma", "Cajas y Bigotes", "Probabilidad Normal", "Regresion Lineal", "Dispersión"]
    graph = col2.selectbox("¿Qué gráfica quieres ver?", options_graph)

    alpha_selection = None
    output_column = None

    if graph == "Regresion Lineal":
        graph_regression(options_column, column)
    
    elif graph == 'Dispersión':
        st.warning("Recuerda que para la **Dispersión** debes incluir una variable de salida")

        output_column = st.selectbox("Variable de salida", options_column)
    
    flag = st.checkbox("Eliminar datos atípicos")
    if flag:
        alpha_selection = st.slider("Factor de alfa para los atípicos:", min_value=0.0, max_value=3.0, value=1.5)

    #st.write(st.session_state["my_dataset"])
    if st.button('Generar Gráfica'):
        try:
            dataset = st.session_state["my_dataset"]
            if flag:
                if output_column != None:
                    title_with = "Gráfica "+graph+" CON Atipicos"
                    graph_two_columns(dataset, column, output_column, graph, title_with)
                    title_without  = "Gráfica "+graph+" SIN Atipicos"
                    dataset_clean = remove_outlier(dataset, column, alpha_selection)
                    graph_two_columns(dataset_clean, column, output_column, graph, title_without)
                else:
                    dataset_clean = remove_outlier(dataset, column, alpha_selection)
                    title_with = "Gráfica "+graph+" CON Atipicos"
                    generate_graphics(dataset, column, graph, title_with)
                    title_without  = "Gráfica "+graph+" SIN Atipicos"
                    generate_graphics(dataset_clean, column, graph, title_without)
            else:
                if output_column != None:
                    title_without  = "Gráfica "+graph
                    graph_two_columns(dataset, column, output_column, graph, title_without)
                else:
                    title_without  = "Gráfica "+graph+" SIN Atipicos"
                    generate_graphics(dataset, column, graph, title_without)
        except:
            st.error("Primero debes ir a **HomePage** y generar el dataset 😒")

           
        

        
       


def main():
    st.title("Dataset Abalone con Datos Atípicos")

    
    user_input()


if __name__ == '__main__':
    main()