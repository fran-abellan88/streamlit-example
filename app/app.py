
import streamlit as st
import pandas as pd
import os
import sys

# -- Change path to Phase2 folder --
if os.path.basename(os.path.normpath(os.getcwd())) != 'streamlit-example':
    sys.path.append('..')
    print(os.getcwd())

st.write("My First Streamlit Web App")
st.write(os.getcwd())

df = pd.DataFrame({"one": [1, 2, 3], "two": [4, 5, 6], "three": [7, 8, 9]})

st.write(df)