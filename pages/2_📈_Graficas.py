##### Paquete Interfaz Grafica
import streamlit as st
#### Paquetes importacion y manejo datos
import pandas as pd
import numpy as np
import scipy.stats.stats as stats
#### Paquete graficas
import plotly.express as px
import plotly.graph_objs as go
from statsmodels.graphics.gofplots import qqplot
######  paquetes de analitica de datos
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def remove_outlier(dataset_in, col_name, value_alpha):
    q1 = dataset_in[col_name].quantile(0.25)
    q3 = dataset_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    lower_limit  = q1-value_alpha*iqr
    upper_limit = q3+value_alpha*iqr
    dataset_out = dataset_in.loc[(dataset_in[col_name]> lower_limit)&(dataset_in[col_name]< upper_limit)]
    return dataset_out

def generateStatics(dataframe, column_name):
    mean = dataframe[column_name].mean()
    median = dataframe[column_name].median()
    mode = dataframe[column_name].mode(dropna=False)

    kurtosis = stats.kurtosis(dataframe[column_name])
    return mean, median, mode, kurtosis

def getSimetryKurtosis(mean, median, mode, kurtosis, position):
    simetry = ""
    ret_kurtosis = ""
    if (mean == median and mean == mode):
        simetry =  "Distribución *Simétrica*"
    elif (mean > median):
        simetry =  "Distribución *Asímetrica* con cola a la derecha (sesgada a la derecha)"
    elif (mean < median):
        simetry = "Distribución *Asímetrica* con cola a la izquierda (sesgada a la izquierda)"

    if (kurtosis == 0):
        ret_kurtosis = "Distribución *Mesocúrtica*"
    elif (kurtosis > 0):
        ret_kurtosis = "Distribución *Leptocúrtica*"
    elif (kurtosis < 0):
        ret_kurtosis = "Distribución *Platicúrtica*"
    position.markdown("### Simetría")
    position.write(simetry)
    position.markdown("### Curtosis")
    position.write(ret_kurtosis)

def generate_graphics(dataset, column, graph, tittle_graph, position):
    mean, median, mode, kurtosis = generateStatics(dataset,column)
    dict_statics = {'Media':mean,'Mediana':median,'Moda':mode, "Curtosis":kurtosis}
    pandas_static = pd.DataFrame(dict_statics).head(1)
    if graph == 'Histograma':
        fig = px.histogram(dataset, x=column, title=tittle_graph)
        position.plotly_chart(fig, use_container_width=True)
        position.write(pandas_static)
        getSimetryKurtosis(mean,median,mode,kurtosis, position)
        
    elif graph == "Cajas y Bigotes":
        fig = px.box(dataset, x=column, title=tittle_graph)
        position.plotly_chart(fig, use_container_width=True)
        position.write(pandas_static)
        getSimetryKurtosis(mean,median,mode,kurtosis, position)
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
        position.plotly_chart(fig, use_container_width=True)
        position.write(pandas_static)
        getSimetryKurtosis(mean,median,mode,kurtosis, position)
            
def graph_dispersion(dataset, column_input, column_output, tittle_graph, position):
    mean, median, mode, kurtosis = generateStatics(dataset,column_input)
    dict_statics = {'Media':mean,'Mediana':median,'Moda':mode, "Curtosis":kurtosis}
    pandas_static = pd.DataFrame(dict_statics).head(1)
    fig = px.scatter(dataset, x=column_input, y=column_output, title=tittle_graph)
    position.plotly_chart(fig, use_container_width=True)
    position.write(pandas_static)
    getSimetryKurtosis(mean,median,mode,kurtosis, position)

def graph_regression(dataframe, inputs_selection, column_to_compare, position):
    mean, median, mode, kurtosis = generateStatics(dataframe,column_to_compare)
    dict_statics = {'Media':mean,'Mediana':median,'Moda':mode, "Curtosis":kurtosis}
    pandas_static = pd.DataFrame(dict_statics).head(1)
    model = LinearRegression()
    fig = None
    r2 = 0
    rmse = 0
    if (len(inputs_selection) == 1):
        X = dataframe[inputs_selection[0]]
        Y = dataframe[column_to_compare]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.5)

        model.fit(X = np.array(X_train).reshape(-1, 1), y = y_train)
        y_predict = model.predict(X = np.array(X_test).reshape(-1,1))

        fig = px.scatter(dataframe, x=inputs_selection[0], y=column_to_compare, opacity=0.50)
        r2 = r2_score(y_true=y_test,y_pred=y_predict)
        rmse = mean_squared_error(y_true=y_test, y_pred=y_predict)

    else:
        X = dataframe[inputs_selection]
        Y = dataframe[column_to_compare]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.5)

        model.fit(X = X_train, y = y_train)

        y_predict = model.predict(X = np.array(X_test))
        fig = px.scatter(dataframe, x=(inputs_selection), y=column_to_compare, opacity=0.50)
        r2 = r2_score(y_true=y_test,y_pred=y_predict)
        rmse = mean_squared_error(y_true=y_test, y_pred=y_predict)

    fig.add_traces(go.Scatter(x=X_test, y=y_predict, name='Regression Fit'))
    position.plotly_chart(fig)
    position.write(pandas_static)
    getSimetryKurtosis(mean,median,mode,kurtosis, position)
    position.markdown("### Métricas de Regresión ")
    dict_metrics = {'RMSE':rmse, 'R^2':r2}
    #dict_statics = {'Media':mean,'Mediana':median,'Moda':mode, "Curtosis":kurtosis}
    pandas_metrics = pd.DataFrame(dict_metrics,index=[0])
    position.write(pandas_metrics)



def main():
    st.title("📈 Gráficas")

    col1, col2 = st.columns(2)

    options_column = ['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
    column = col1.selectbox("¿Sobre cuál columna quieres ver las gráficas?", options_column)

    options_graph = ["Histograma", "Cajas y Bigotes", "Probabilidad Normal", "Regresion Lineal", "Dispersión"]
    graph = col2.selectbox("¿Qué gráfica quieres ver?", options_graph)

    output_column = None
    input_column = None

    if graph == "Regresion Lineal":
        st.warning("Recuerda que para usar **Regresión Lineal** debes incluir al menos una variable de entrada y sólo una de salida")
        col3, col4 = st.columns(2)

        input_column = col3.multiselect('Variable(s) de entrada:', options_column, default=column)

        column_input = ['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
        output_column = col4.selectbox("Variable de salida", column_input)
    
    elif graph == 'Dispersión':
        st.warning("Recuerda que para la **Dispersión** debes incluir una variable de salida")

        output_column = st.selectbox("Variable de salida", options_column)
    
    flag = st.checkbox("Eliminar datos atípicos")
    if flag:
        alpha_selection = st.slider("Factor de alfa para los atípicos:", min_value=0.0, max_value=3.0, value=1.5)

    if st.button('Generar Gráfica'):
        try:
            dataset = st.session_state["my_dataset"]
            if flag:
                col1, col2 = st.columns(2)
                if output_column != None:
                    if input_column != None:
                        
                        graph_regression(dataset, input_column, output_column, st)

                        dataset_clean = remove_outlier(dataset, column, alpha_selection)
                        graph_regression(dataset_clean, input_column, output_column, st)

                    else:
                        title_with = "Gráfica "+graph+" CON Atipicos"
                        graph_dispersion(dataset, column, output_column, title_with, col1)

                        title_without  = "Gráfica "+graph+" SIN Atipicos"
                        dataset_clean = remove_outlier(dataset, column, alpha_selection)

                        graph_dispersion(dataset_clean, column, output_column, title_without, col2)
                        st.write("Se eliminaron ", (len(dataset)-len(dataset_clean)), "registros. Actualmente hay",len(dataset_clean))
                else:
                    dataset_clean = remove_outlier(dataset, column, alpha_selection)
                    title_with = "Gráfica "+graph+" CON Atipicos"
                    generate_graphics(dataset, column, graph, title_with, col1)

                    title_without  = "Gráfica "+graph+" SIN Atipicos"
                    generate_graphics(dataset_clean, column, graph, title_without, col2)
                    #output = "Se eliminaron ", (len(dataset)-len(dataset_clean)),"registros. Actualmente hay", len(dataset_clean)
                    st.write("Se eliminaron ", (len(dataset)-len(dataset_clean)),"registros. Actualmente hay", len(dataset_clean))
            else:
                if output_column != None:
                    if input_column!=None : graph_regression(dataset, input_column, output_column, st)
                    title_without  = "Gráfica "+graph
                    graph_dispersion(dataset, column, output_column, title_without, st)
                else:
                    title_without  = "Gráfica "+graph+" CON Atipicos"
                    generate_graphics(dataset, column, graph, title_without, st)
        except:
            st.error("Primero debes ir a **HomePage** y generar el dataset 😒")

if __name__ == '__main__':
    main()