import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import build_transformer, preprocess_features, encode_target, scale_target, only_numeric_check
from model_training import train_classifiers, train_regressors, run_unsupervised

st.set_page_config(layout='wide', page_title='ML Model Explorer')


page = st.sidebar.radio(
    "Navigate to:",
    ["Train a Model", "About"]
)

st.sidebar.title('ML Model Explorer')
mode = st.sidebar.selectbox('Select Learning Type', ['Supervised', 'Unsupervised'])

st.title('Machine Learning Model Explorer')


if page == 'Train a Model':

    if mode == 'Supervised':
        st.header('Supervised Learning')
        uploaded = st.file_uploader('Upload CSV for supervised', type=['csv'], key='sup')
        if uploaded is None:
            st.info('Upload a CSV file to get started (the dataset must include a target column).')
            st.stop()

        df = pd.read_csv(uploaded)
        st.subheader('Dataset preview')
        st.dataframe(df.head())


        st.subheader("Data Distribution Overview")

        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(num_cols) > 0:
            st.write("### Numerical Feature Distributions")
            fig, axes = plt.subplots(nrows=(len(num_cols) + 2)//3, ncols=3, figsize=(15, 10))
            axes = axes.flatten()
            for i, col in enumerate(num_cols):
                sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
                axes[i].set_title(col)
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            st.pyplot(fig)

        if len(cat_cols) > 0:
            st.write("### Categorical Feature Distributions")
            fig, axes = plt.subplots(nrows=(len(cat_cols) + 2)//3, ncols=3, figsize=(15, 10))
            axes = axes.flatten()
            for i, col in enumerate(cat_cols):
                sns.countplot(y=col, data=df, ax=axes[i], palette='pastel')
                axes[i].set_title(col)
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            st.pyplot(fig)

        # Sidebar model selection and params
        st.sidebar.subheader('Supervision Type')
        sup_type = st.sidebar.selectbox('Choose model', ['Classification', 'Regression'])

        if sup_type == 'Classification':
            st.header('Supervised Learning (Classification)')

            target_col = st.selectbox('Select target column', df.columns)
            if not target_col:
                st.warning('Select a target column.')
                st.stop()

            X = df.drop(columns=[target_col])
            y = df[target_col]

            st.write('Detected numeric columns before preprocessing:', X.select_dtypes(include=[np.number]).columns.tolist())
            st.write('Detected categorical columns before preprocessing:', X.select_dtypes(exclude=[np.number]).columns.tolist())

            # preprocessing
            transformer = build_transformer(df)
            X_proc = preprocess_features(transformer, X)

            non_numeric_after = only_numeric_check(X_proc)
            if non_numeric_after:
                st.error('After preprocessing non-numeric columns found: ' + str(non_numeric_after))
                st.stop()

            y_enc, label_enc = encode_target(y)

            # Sidebar model selection and params
            st.sidebar.subheader('Model selection')
            model_name = st.sidebar.selectbox('Choose model', ['Decision Tree', 'Random Forest', 'SVM'])

            st.sidebar.subheader('Train settings')
            test_size = st.sidebar.slider('Test set size (%)', 5, 50, 20)

            params = {}
            if model_name == 'Decision Tree':
                params['max_depth'] = st.sidebar.number_input('max_depth', min_value=1, max_value=50, value=5)
                params['min_samples_split'] = st.sidebar.number_input('min_samples_split', min_value=2, max_value=20, value=2)
                params['min_samples_leaf'] = st.sidebar.number_input('min_samples_leaf', min_value=2, max_value=25, value=3)
                params['max_features'] = st.sidebar.selectbox('max_features', ['sqrt', 'log2'])
            elif model_name == 'Random Forest':
                params['n_estimators'] = st.sidebar.number_input('n_estimators', min_value=10, max_value=500, value=100)
                params['max_depth'] = st.sidebar.number_input('max_depth', min_value=1, max_value=50, value=10)
                params['min_samples_leaf'] = st.sidebar.number_input('min_samples_leaf', min_value=2, max_value=25, value=3)
                params['max_features'] = st.sidebar.selectbox('max_features', ['sqrt', 'log2'])
            else:
                params['C'] = st.sidebar.number_input('C', min_value=0.01, max_value=100.0, value=1.0)
                params['kernel'] = st.sidebar.selectbox('kernel', ['rbf', 'linear', 'poly'])

            if st.sidebar.button('Train model'):
                result = train_classifiers(X_proc, y_enc, model_name, params, test_size=test_size/100)

                st.subheader('Metrics')
                st.write('Accuracy:', f"{result['accuracy']:.3f}")
                st.write('Precision (weighted):', f"{result['precision']:.3f}")
                st.write('Recall (weighted):', f"{result['recall']:.3f}")
                st.write('F1 (weighted):', f"{result['f1']:.3f}")

                st.subheader('Classification report')
                st.text(result['classification_report'])

                st.subheader('Confusion Matrix')
                st.pyplot(result['confusion_fig'])

                if result.get('feature_importances') is not None:
                    st.subheader('Top feature importances')
                    st.bar_chart(result['feature_importances'].head(20))

        if sup_type == 'Regression':
            st.header('Supervised Learning (Regression)')

            target_col = st.selectbox('Select target column', df.columns, key='reg_target')
            if not target_col:
                st.warning('Select a target column.')
                st.stop()

            X = df.drop(columns=[target_col])
            y = df[target_col]

            # preprocessing
            transformer = build_transformer(df)
            X_proc = preprocess_features(transformer, X)

            non_numeric_after = only_numeric_check(X_proc)
            if non_numeric_after:
                st.error('After preprocessing non-numeric columns found: ' + str(non_numeric_after))
                st.stop()

            y_enc = scale_target(y)

            # Sidebar model selection and params
            st.sidebar.subheader('Model selection')
            model_name = st.sidebar.selectbox('Choose model', ['Decision Tree', 'Random Forest', 'SVM'], key='reg_model')

            st.sidebar.subheader('Train settings')
            test_size = st.sidebar.slider('Test set size (%)', 5, 50, 20, key='reg_testsize')

            params = {}
            if model_name == 'Decision Tree':
                params['max_depth'] = st.sidebar.number_input('max_depth', min_value=1, max_value=50, value=5, key='reg_depth')
                params['min_samples_split'] = st.sidebar.number_input('min_samples_split', min_value=2, max_value=20, value=2, key='reg_split')
                params['min_samples_leaf'] = st.sidebar.number_input('min_samples_leaf', min_value=2, max_value=25, value=3)
                params['max_features'] = st.sidebar.selectbox('max_features', ['sqrt', 'log2'])
            elif model_name == 'Random Forest':
                params['n_estimators'] = st.sidebar.number_input('n_estimators', min_value=10, max_value=500, value=100, key='reg_estimators')
                params['max_depth'] = st.sidebar.number_input('max_depth', min_value=1, max_value=50, value=10, key='reg_maxdepth')
                params['min_samples_leaf'] = st.sidebar.number_input('min_samples_leaf', min_value=2, max_value=25, value=3)
                params['max_features'] = st.sidebar.selectbox('max_features', ['sqrt', 'log2'])
            else:
                params['C'] = st.sidebar.number_input('C', min_value=0.01, max_value=100.0, value=1.0, key='reg_C')
                params['kernel'] = st.sidebar.selectbox('kernel', ['rbf', 'linear', 'poly'], key='reg_kernel')

            if st.sidebar.button('Train model', key='reg_train'):
                result = train_regressors(X_proc, y_enc, model_name, params, test_size=test_size/100)

                st.subheader('Metrics')
                st.write('RMSE:', f"{result['rmse']:.3f}")
                st.write('MAE:', f"{result['mae']:.3f}")
                st.write('R² Score:', f"{result['r2']:.3f}")
                st.subheader('Predicted vs Actual')
                st.pyplot(result['fig'])

                st.subheader('Residual Plot')
                st.pyplot(result['fig2'])

                if result.get('feature_importances') is not None:
                    st.subheader('Top feature importances')
                    st.bar_chart(result['feature_importances'].head(20))

    if mode == 'Unsupervised':
        st.header('Unsupervised Learning')
        uploaded = st.file_uploader('Upload CSV for unsupervised', type=['csv'], key='unsup')
        if uploaded is None:
            st.info('Upload a CSV file to run clustering.')
            st.stop()

        df = pd.read_csv(uploaded)
        st.subheader('Dataset preview')
        st.dataframe(df.head())

        st.subheader("Data Distribution Overview")

        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(num_cols) > 0:
            st.write("### Numerical Feature Distributions")
            fig, axes = plt.subplots(nrows=(len(num_cols) + 2)//3, ncols=3, figsize=(15, 10))
            axes = axes.flatten()
            for i, col in enumerate(num_cols):
                sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
                axes[i].set_title(col)
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            st.pyplot(fig)

        if len(cat_cols) > 0:
            st.write("### Categorical Feature Distributions")
            fig, axes = plt.subplots(nrows=(len(cat_cols) + 2)//3, ncols=3, figsize=(15, 10))
            axes = axes.flatten()
            for i, col in enumerate(cat_cols):
                sns.countplot(y=col, data=df, ax=axes[i], palette='pastel')
                axes[i].set_title(col)
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            st.pyplot(fig)

        transformer = build_transformer(df)
        X_proc = preprocess_features(transformer, df)

        non_numeric_after = only_numeric_check(X_proc)
        if non_numeric_after:
            st.error('After preprocessing non-numeric columns found: ' + str(non_numeric_after))
            st.stop()

        st.sidebar.subheader('Clustering')
        cluster_algo = st.sidebar.selectbox('Algorithm', ['KMeans', 'Agglomerative', 'DBSCAN'])

        params = {}
        if cluster_algo == 'KMeans':
            params['n_clusters'] = st.sidebar.number_input('n_clusters', min_value=2, max_value=20, value=3)
        elif cluster_algo == 'Agglomerative':
            params['n_clusters'] = st.sidebar.number_input('n_clusters', min_value=2, max_value=20, value=3, key='agg')
            params['linkage'] = st.sidebar.selectbox('linkage', ['ward','complete','average','single'])
        else:
            params['eps'] = st.sidebar.number_input('eps', min_value=0.1, max_value=10.0, value=0.5)
            params['min_samples'] = st.sidebar.number_input('min_samples', min_value=1, max_value=50, value=5)

        if st.sidebar.button('Run clustering'):
            out = run_unsupervised(X_proc, cluster_algo, params)
            st.metric('Silhouette score', f"{out['silhouette']:.3f}" if out['silhouette'] is not None else 'N/A')
            st.pyplot(out['cluster_fig'])

if page == 'About':
    st.header("About the Application")
    st.write('''
Machine Learning Model Explorer is an interactive Streamlit web app that allows you to experiment with machine learning techniques in an intuitive way. It provides a simple interface for uploading datasets, preprocessing data automatically, training models, and visualizing results—all without writing code.

The app includes both Supervised and Unsupervised learning modules. In the supervised section, you can train models for classification and regression using algorithms like Decision Tree, Random Forest, and Support Vector Machine (SVM). In the unsupervised section, you can explore clustering methods such as KMeans, Agglomerative Clustering, and DBSCAN.

Each module automatically handles missing values, encodes categorical data, and scales numerical features. After training, the app displays key metrics and visualizations like confusion matrices, residual plots, and cluster maps to help you understand model performance.

This project was created to make learning and experimenting with machine learning concepts easier and more practical—ideal for students, educators, and anyone exploring data-driven insights.''')