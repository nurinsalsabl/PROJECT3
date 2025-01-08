import streamlit as st
import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from random_forest_manual import RandomForest, StandardScaler


# Set page configuration
st.set_page_config(
    page_title="Student Dropout Prediction App",  # Page title
    page_icon="📊",  # Page icon (emoji or file)
    layout="wide",  # Wide layout for the app
    initial_sidebar_state="expanded"  # Keep sidebar expanded by default
)

# Sidebar for color theme and settings
st.sidebar.title("Visualization Settings")
selected_color_theme = st.sidebar.selectbox(
    "Choose color theme for visualizations:",
    ["viridis", "inferno", "plasma", "magma"]
)
st.sidebar.write(f"Selected Color Theme: {selected_color_theme}")

# Membaca data bersih dari file CSV
df_clean = pd.read_csv("cleaned_data.csv")
X = df_clean.drop("Target", axis=1)
y = df_clean["Target"]


# Memuat model dari file pickle
model = pickle.load(open("rf_model_7742.pkl", "rb"))


# Memprediksi data yang sama dengan yang digunakan untuk training
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
accuracy = round((accuracy * 100), 2)

# Menyiapkan DataFrame untuk user input
df_final = X
df_final["Target"] = y


# Mapping dictionaries for descriptive dropdown labels

application_mode_mapping = {
        1: '1st phase - general contingent',
        2: 'Ordinance No. 612/93',
        5: '1st phase - special contingent (Azores Island)',
        7: 'Holders of other higher courses',
        10: 'Ordinance No. 854-B/99',
        15: 'International student (bachelor)',
        16: '1st phase - special contingent (Madeira Island)',
        17: '2nd phase - general contingent',
        18: '3rd phase - general contingent',
        26: 'Ordinance No. 533-A/99, item b2) (Different Plan)',
        27: 'Ordinance No. 533-A/99, item b3 (Other Institution)',
        39: 'Over 23 years old',
        42: 'Transfer',
        43: 'Change of course',
        44: 'Technological specialization diploma holders',
        51: 'Change of institution/course',
        53: 'Short cycle diploma holders',
        57: 'Change of institution/course (International)'
    }
course_mapping = {
        33: 'Biofuel Production Technologies',
        171: 'Animation and Multimedia Design',
        8014: 'Social Service (evening attendance)',
        9003: 'Agronomy',
        9070: 'Communication Design',
        9085: 'Veterinary Nursing',
        9119: 'Informatics Engineering',
        9130: 'Equinculture',
        9147: 'Management',
        9238: 'Social Service',
        9254: 'Tourism',
        9500: 'Nursing',
        9556: 'Oral Hygiene',
        9670: 'Advertising and Marketing Management',
        9773: 'Journalism and Communication',
        9853: 'Basic Education',
        9991: 'Management (evening attendance)'
    }
displaced_mapping = {
    1: 'Yes',
    2: 'No'
    }
debtor_mapping = {
    1: 'Yes',
    2: 'No'
    }
tuition_fees_up_to_date_mapping = {
    1: 'Yes',
    2: 'No'
    }
gender_mapping = {
    1: 'Male',
    2: 'Female'
    }
scholarship_holder_mapping = {
    1: 'Yes',
    2: 'No'
    }

# Define a function for univariate analysis
def univariate_analysis(df):
    st.subheader("Univariate Analysis")
    st.write("### Histograms")
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_columns:
        st.write(f"**{col}**")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col], ax=ax[0], kde=True)
        ax[0].set_title(f"Histogram of {col}")
        sns.boxplot(y=df[col], ax=ax[1])
        ax[1].set_title(f"Boxplot of {col}")
        st.pyplot(fig)

    st.write("### Bar Charts for Categorical Variables")
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        st.write(f"**{col}**")
        st.bar_chart(df[col].value_counts())

# Define a function for bivariate analysis
# def bivariate_analysis(df):
#     st.subheader("Bivariate Analysis")
#     st.write("### Correlation Matrix")
#     numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
#     correlation_matrix = df[numerical_columns].corr()
#     fig, ax = plt.subplots(figsize=(10, 8))
#     sns.heatmap(correlation_matrix, annot=True, cmap=selected_color_theme, ax=ax)
#     st.pyplot(fig)

#     st.write("### Scatter Plot Pairs (Limited Features)")
#     selected_columns = numerical_columns[:5]
#     pair_plot = sns.pairplot(df[selected_columns], diag_kind="kde", palette=selected_color_theme)
#     st.pyplot(pair_plot)

# # Define a function for multivariate analysis (PCA)
# def multivariate_analysis(df, selected_color_theme):
#     st.subheader("Multivariate Analysis")
#     st.write("### Principal Component Analysis (PCA)")

#     numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
#     # Use the original numerical columns without scaling
#     df_scaled = df[numerical_columns]

#     pca = PCA(n_components=2)
#     pca_data = pca.fit_transform(df_scaled)

#     target_mapping = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
#     df['Target_Code'] = df['Target'].map(target_mapping)

#     pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
#     pca_df['Target_Code'] = df['Target_Code']

#     fig, ax = plt.subplots()
#     scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Target_Code'], cmap=selected_color_theme)
#     legend_labels = {v: k for k, v in target_mapping.items()}
#     handles, _ = scatter.legend_elements()
#     legend = ax.legend(handles, [legend_labels[i] for i in range(len(handles))], title="Target")
#     ax.add_artist(legend)

#     ax.set_xlabel('Principal Component 1')
#     ax.set_ylabel('Principal Component 2')
#     ax.set_title('PCA of Dataset')
#     st.pyplot(fig)

import pickle

def load_rf_model():
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "C:\\Users\\acer\\OneDrive\\SALSA'S STUDY\\SEM 3\\AI\\Student-Dropout-Prediction-main\\models"))
    randomforest_model_path = os.path.join(model_dir, 'rf_model_7742.pkl')
    with open(randomforest_model_path, 'rb') as file:
        model = pickle.load(file)
    return model
# Manual mapping for numeric predictions to human-readable labels
prediction_mapping = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
# Load the trained model

def main():
    st.title("Student Dropout Prediction")
    model = RandomForest(n_trees=100)


    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "cleaned_data.csv")  
    df = pd.read_csv(data_path) 

    if df is not None:
        st.success("Data loaded successfully!")

        feature_cols = df.drop(columns=['Target']).columns.tolist() if 'Target' in df.columns else df.columns.tolist()

        if st.checkbox("Show raw data"):
            st.write(df.head())

        # if st.checkbox("Perform Univariate Analysis"):
        #     univariate_analysis(df)

        # if st.checkbox("Perform Bivariate Analysis"):
        #     bivariate_analysis(df)

        # if st.checkbox("Perform Multivariate Analysis"):
        #     multivariate_analysis(df, selected_color_theme)

        if st.checkbox("Show initial data exploration"):
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            st.write(df.dtypes)
            st.write(df.describe())

        if st.checkbox("Check for missing values"):
            missing_values = df.isnull().sum()
            st.write(missing_values[missing_values > 0])
            if missing_values.sum() > 0:
                plt.figure(figsize=(10, 6))
                sns.heatmap(df.isnull(), cbar=False, cmap=selected_color_theme)
                st.pyplot(plt.gcf())

        num_cols = ['Admission grade', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']

        if st.checkbox("Show box plots for numerical columns"):
            fig, axes = plt.subplots(1, len(num_cols), figsize=(12, 8))
            for i, col in enumerate(num_cols):
                sns.boxplot(y=df[col], ax=axes[i])
                axes[i].set_title(col)
            st.pyplot(fig)

         # Persiapan Data untuk Prediksi
        if st.checkbox("Show preprocessed data"):
            st.write(df.head())
        
        X = df.drop(columns='Target')
        y = df['Target']

        # Inisialisasi StandardScaler dan lakukan scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  

        # Latih model RandomForest
        rf_model = RandomForest(n_trees=100, max_depth=10) # Sesuaikan parameter jika diperlukan
        rf_model.fit(X_scaled, y)

        if st.checkbox("Show feature importance"):
            # model.fit(X_train, y_train)
            importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
            st.write(feature_importance_df)

            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
            st.pyplot(fig)

        # User inputs with descriptive dropdowns
        st.header("Enter details for prediction:")
        selected_application_mode = st.selectbox(
            "Application Mode",
            options=list(application_mode_mapping.keys()),
            format_func=lambda x: application_mode_mapping[x]
        )
        select_displaced_mapping = st.selectbox(
            "Displaced",
            options=list(displaced_mapping.keys()),
            format_func=lambda x: displaced_mapping[x]
        )
        select_debtor_mapping = st.selectbox(
            "Debtor",
            options=list(debtor_mapping.keys()),
            format_func=lambda x: debtor_mapping[x]
        )
        select_tuition_fees_up_to_date_mapping = st.selectbox(
            "Tuition Fees Up To Date",
            options=list(tuition_fees_up_to_date_mapping.keys()),
            format_func=lambda x: tuition_fees_up_to_date_mapping[x]
        )
        select_gender_mapping = st.selectbox(
            "Gender",
            options=list(gender_mapping.keys()),
            format_func=lambda x:gender_mapping[x]
        )
        select_scholarship_holder_mapping = st.selectbox(
            "Scholarship Holder",
            options=list(scholarship_holder_mapping.keys()),
            format_func=lambda x: scholarship_holder_mapping[x]
        )

        st.write("### Model Prediction")
        st.write(f"- Application Mode: {application_mode_mapping[selected_application_mode]}")
        st.write(f"- Displaced: {displaced_mapping[select_displaced_mapping]}")
        st.write(f"- Debtor: {debtor_mapping[select_debtor_mapping]}")
        st.write(f"- Tuition Fees Up To Date: {tuition_fees_up_to_date_mapping[select_tuition_fees_up_to_date_mapping]}")
        st.write(f"- Gender: {gender_mapping[select_gender_mapping]}")
        st.write(f"- Scholarship Holder: {scholarship_holder_mapping[select_scholarship_holder_mapping]}")

        numerical_cols = [
            'Age at enrollment',
            'Curricular units 1st sem (enrolled)',
            'Curricular units 1st sem (approved)',
            'Curricular units 1st sem (grade)',
            'Curricular units 2nd sem (enrolled)',
            'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
            'Curricular units 2nd sem (grade)'
        ]

        with st.form(key='prediction_form'):
            st.write("Enter details for prediction:")
            user_input = {}
            for col in numerical_cols:
                user_input[col] = st.number_input(f"{col}")
            submit_button = st.form_submit_button("Predict")

            if submit_button:
                try:
                    input_data = pd.DataFrame([user_input])
                    input_data = input_data.reindex(columns=feature_cols, fill_value=0)

                    # Scaling data input pengguna
                    input_data_scaled = scaler.transform(input_data)  

                    numeric_prediction = rf_model.predict(input_data_scaled)[0]
                    label_prediction = prediction_mapping.get(numeric_prediction, "Unknown")
                    st.success(f"Prediction: {label_prediction}")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        st.write("### Insights, Recommendations, and Conclusions")
        st.subheader("Insights")
        st.write("""
        - The dataset contains 4,424 rows and 38 columns.
        - Outliers were detected in key numerical variables such as Admission grade and semester grades.
        - Gender shows a significant impact on dropout rates according to the Chi-square test.
        """)

        st.subheader("Recommendations")
        st.write("""
        - Investigate missing values and handle them appropriately to improve model performance.
        - Consider outlier treatment techniques like transformations to avoid skewing model predictions.
        """)

        st.subheader("Conclusions")
        st.write("""
        - Admission grades alone may not be a strong predictor of dropouts.
        - Gender has a notable influence on dropout rates, indicating a need for gender-targeted interventions.
        """)

    else:
        st.error("Failed to load data.")

if __name__ == '__main__':
    main()
