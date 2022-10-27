import streamlit as st
import pandas as pd
import os
import joblib
import sys
import base64

# -- Change path to Phase2 folder --
if os.path.basename(os.path.normpath(os.getcwd())) != 'streamlit-example':
    sys.path.append('..')
    # os.chdir('..')


@st.cache(suppress_st_warning=True)
def get_table_download_link_csv(df, filename):
    """
    Generates a link allowing the data in a given pandas dataframe to be downloaded
    :param df:
    :param filename:
    :return: href to csv
    """
    # csv = df.to_csv(index=False)
    csv = df.to_csv(sep=";", decimal=",", index=False, encoding='utf-8').encode('utf-8')
    b64 = base64.b64encode(csv).decode('utf-8')
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv" target="_blank">Download csv file</a>'

    return href


st.set_page_config(layout="wide")

# -- START OF WEB APP --
col1, col2, col3 = st.columns(3)
col2.title('Predict Accounts Value')

st.header('Load files')

with st.expander("Load file 'accounts_test.csv'"):
    accounts_test = pd.DataFrame()
    uploaded_file = st.file_uploader("", type="csv", key="accounts_test")

if uploaded_file:
    try:
        accounts_test = pd.read_csv(uploaded_file, sep=",", encoding='latin-1')
        st.success('**File loaded successfully**')
        # st.dataframe(data=accounts_test.head(5), width=2000, height=300)
    except Exception as e:
        print(e)
        pass

    with st.expander("Load file 'quotes_test.csv'"):
        quotes_test = pd.DataFrame()
        uploaded_file = st.file_uploader("", type="csv", key="quotes_test")

    if uploaded_file:
        try:
            quotes_test = pd.read_csv(uploaded_file, sep=",", encoding='latin-1')
            st.success('**File loaded successfully**')
            # st.dataframe(data=quotes_test.head(5), width=2000, height=300)
        except Exception as e:
            print(e)
            pass

        df = accounts_test.merge(quotes_test, how='left', on='account_uuid')
        # TODO: Modificar esto
        df.dropna(inplace=True)

        # -- CAST COLUMNS --
        df['carrier_id'] = 'carrier_id_' + df['carrier_id'].astype(str)
        df['year_established'] = df['year_established'].astype(int)
        df['num_employees'] = df['num_employees'].astype(int)

        # -- CREATE X_test --
        X_test = df.drop(columns='account_uuid')

        st.write('**Input data (Top 5 rows)**')
        st.dataframe(data=X_test.head(5), width=2000, height=210)

        # -- LOCAL --
        # clf = joblib.load('../models/catboost_model.joblib')

        # -- REMOTE --
        clf = joblib.load('models/catboost_model.joblib')

        y_pred = clf.predict(X_test)
        y_pred_proba_test = clf.predict_proba(X_test)
        y_scores = y_pred_proba_test[:, 1]

        df_results = X_test.copy()
        df_results['account_uuid'] = df['account_uuid']
        df_results['convert'] = y_pred

        st.write('**Input data with predictions (Top 5 rows)**')
        st.dataframe(data=df_results.head(5), width=3000, height=210)

        st.write('**Accounts value (Top 5 rows)**')
        df_results['account_value_by_product'] = df_results['premium'] * df_results['convert']
        df_accounts_value = df_results.groupby('account_uuid')['account_value_by_product'].sum().to_frame().rename(columns={'account_value_by_product': 'account_value'})
        df_accounts_value.reset_index(inplace=True)
        st.dataframe(data=df_accounts_value.head(5), width=3000, height=210)

        st.header('Download predictions')

        filename = st.text_input("Filename", "predictions")
        if st.button('Download predictions'):
            st.markdown(get_table_download_link_csv(df_results, filename), unsafe_allow_html=True)
        filename = st.text_input("Filename", "accounts_value")
        if st.button('Download accounts value'):
            st.markdown(get_table_download_link_csv(df_accounts_value, filename), unsafe_allow_html=True)
