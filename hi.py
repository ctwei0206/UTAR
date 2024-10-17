import streamlit as st
#set the app title
st.title('My first streamlit app')
#streamlit run [filename].py
#Display text output
st.write('Welcome to my first streamlit app')
#display a button
st.button("Reset", type="primary")
if st.button("Say hello"):
  st.write("Why hello there")
else:
  st.write("Goodbye")
