import streamlit as st


def main():
    st.title("📓 Contacto")

    st.info("Proyecto desarrollado por [Andrey Hernández](https://github.com/AndreyHernandezT/) y Yireth Aldana, para la materia Analítica de Datos.",icon="ℹ️")

    st.header("🚀 Información sobre los desarrolladores")
    st.markdown("### 💻 Andrey Hernandez")
    col1, mid, col2 = st.columns([5,1,10])
    with col1:
        st.image('img/andrey.jpg')
    with col2:
        st.write("Estudiante de Ingeniería de Sistemas e Informática de la Universidad Pontificia Bolivariana Seccional Bucaramanga. Becado por el Ministerio de Tecnologías de la Información y las Comunicaciones de Colombia (MinTIC). Diplomado en Habilidades de Programación, becado por MinTIC. Conocimientos en Programación con múltiples lenguajes, Seguridad de la Información, Base de Datos Relacionales, Desarrollo de Software con Metodología Scrum y algunos Frameworks de Desarrollo. [Visitar Linkedin](https://www.linkedin.com/in/andreyhdez/)")

    
    st.markdown("### 💻 Yireth Aldana")
    col1, mid, col2 = st.columns([5,1,10])
    with col1:
        st.image('img/yira.jpeg')
    with col2:
        st.write("Estudiante de Ingeniería de Sistemas e Informática de la Universidad Pontificia Bolivariana Seccional Bucaramanga.")


if __name__ == '__main__':
    main()