import streamlit as st
import pandas as pd
import plotly.express as px


def get_dataset(data):
    dataset = pd.read_csv(data)

    name_columns = ['sex','Length','Diameter','Height','Whole weight',
           'Shucked weight','Viscera weight','Shell weight','Rings']  #Columnas del dataset
    dataset.columns = name_columns #Asigna nombre a las columnas
    return dataset

def main():
    st.title('🤓 Visualización de los datos Abalone')

    st.write("Si quieres ver las gráficas del dataset conservando o eliminando los datos atípicos, selecciona la página 📈 **Gráficos** en el sidebar de la izquierda, recuerda que debes ingresar los datos correctamente.")
    
    st.warning("Recuerda que **¡Debes generar el Dataset para ver las gráficas!**", icon="⚠️")

    st.subheader("Dataset Abalone")
    st.write("El dataset [Abalone](https://archive.ics.uci.edu/ml/datasets/Abalone) consisten en datos de ``4176`` abalones. Los datos consisten en mediciones del tipo (macho, hembra y cría), la medida más larga de la concha, el diámetro, la altura y varios pesos (entero, descascarillado, vísceras y concha). El resultado es el número de anillos. La edad del abalón es el número de anillos mas 1,5.")

    submit = st.button("Generar Dataset")

    if submit:
        dataset = get_dataset('abalone.csv')

        if "my_dataset" not in st.session_state:
            st.session_state["my_dataset"] = None

        st.session_state["my_dataset"] = dataset
        st.write("El dataset **Abalone** tiene un total de", len(dataset), "registros, contando los datos atípicos de cada columna")

        st.dataframe(dataset)

        

if __name__ == '__main__':
    main()