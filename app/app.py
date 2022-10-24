import streamlit as st
import pandas as pd
import os
import joblib
import sys

# -- Change path to Phase2 folder --
if os.path.basename(os.path.normpath(os.getcwd())) != 'streamlit-example':
    sys.path.append('..')
    # os.chdir('..')
    print(os.getcwd())

# # TODO: Leer esto de un fichero externo
# features = ['state',
#             'industry',
#             'subindustry',
#             'year_established',
#             'annual_revenue',
#             'total_payroll',
#             'business_structure',
#             'num_employees',
#             'product',
#             'premium',
#             'carrier_id']

st.set_page_config(layout="wide")
st.write(os.getcwd())

col1, col2  = st.columns(2)
col2.title('Predict Account Value')

st.header('Cargar tablón de rechazos')

with st.expander("Load file accounts_test.csv"):
    accounts_test = pd.DataFrame()
    uploaded_file = st.file_uploader("", type="csv", key="accounts_test")

if uploaded_file:
    try:
        accounts_test = pd.read_csv(uploaded_file, sep=",", encoding='latin-1')
        st.dataframe(data=accounts_test.head(5), width=2000, height=300)
    except Exception as e:
        print(e)
        pass

    with st.expander("Load file quotes_test.csv"):
        quotes_test = pd.DataFrame()
        uploaded_file = st.file_uploader("", type="csv", key="quotes_test")

    if uploaded_file:
        try:
            quotes_test = pd.read_csv(uploaded_file, sep=",", encoding='latin-1')
            st.dataframe(data=quotes_test.head(5), width=2000, height=300)
        except Exception as e:
            print(e)
            pass

        df = accounts_test.merge(quotes_test, how='left', on='account_uuid')
        # TODO: Modificar esto
        df.dropna(inplace=True)
        X_test = df.drop(columns='account_uuid')

        st.write('Input data')
        st.dataframe(data=X_test.head(5), width=2000, height=300)

        clf = joblib.load('../models/catboost_model.joblib')

        y_pred = clf.predict(X_test)
        y_pred_proba_test = clf.predict_proba(X_test)
        y_scores = y_pred_proba_test[:, 1]

        df_results = X_test.copy()
        df_results['account_uuid'] = df['account_uuid']
        df_results['convert'] = y_pred

        st.write('Predictions')
        st.dataframe(data=df_results.head(5), width=3000, height=250)
