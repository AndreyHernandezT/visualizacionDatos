# Visualización de datos
_Aplicación web para la visualización de datos estadísticos del dataset [Abalone](https://archive.ics.uci.edu/ml/datasets/Abalone) con posibilidad de eliminar datos atípicos y observar diversas gráficas._

## 🚀 Pruébalo

Visita [este link](https://visualizacion-datos.herokuapp.com/) para probar la aplicación web en **Heroku**, recuerda seguir las instrucciones de la página de inicio.

Si no deseas ver el proyecto en heroku, puedes clonar el proyecto. Recuerda que para ejecutar el programa en tu computador, debes escribir el siguiente comando en la terminal.

```
streamlit run 1_🤓_HomePage
```

## 📌 ¿Qué encontrarás?
### 📋 Conjunto de datos
En el proyecto _visualización de datos_ encontrarás el dataset **Abalone**, el cual consiste en ``4176`` registros de [abalones](https://es.wikipedia.org/wiki/Haliotis) (orejas de mar). Los datos consisten en mediciones del tipo (macho, hembra y cría), la medida más larga de la concha, el diámetro, la altura y varios pesos (entero, descascarillado, vísceras y concha). El resultado es el número de anillos. La edad del abalón es el número de anillos mas 1,5. 

Así mismo, el proyecto dispone de un apartado para **eliminar datos atípicos** a través del **Rango Intercuartílico**, dado como parámetro el valor ``alpha``.

### 📈 Gráficas
Para una mejor visualización de los datos, el proyecto dispone un apartado de gráficas, las cuales son:
* Histograma
* Cajas y Bigotes (Bloxplot)
* Probabilidad normal (Normalización)
* Regresión Lineal simple y multiparamétrica
* Dispersión


## 🙇🏼‍♂️ Disclaimer
Por alguna razón (que realmente no entiendo), la gráfica _Probabilidad Normal_ no funciona en Heroku, pero sí localmente.  



