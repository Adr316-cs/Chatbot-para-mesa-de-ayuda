import streamlit as st
from chatbot import predict_class, get_response, intents

st.title("🤖 Chatbot MAI")

# Inicializar el estado de la sesión para almacenar los mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Inicializar el estado de la sesión para controlar si se ha mostrado el mensaje de bienvenida
if "first_message" not in st.session_state:
    st.session_state.first_message = True

# Mostrar los mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Mostrar el mensaje de bienvenida solo la primera vez
if st.session_state.first_message:
    with st.chat_message("assistant"):
        st.markdown("Hola, ¿como puedo ayudarte?")

    st.session_state.first_message = False

# Manejar la entrada del usuario
if prompt := st.chat_input("Escribe tu mensaje aquí..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

# Aquí es donde se procesaría el mensaje del usuario y se generaría la respuesta del chatbot
    intents_list = predict_class(prompt)
    response = get_response(intents_list, intents)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    