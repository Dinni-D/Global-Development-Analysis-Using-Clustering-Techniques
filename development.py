import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm

st.title("🌍 Country Development Cluster Analyzer")

# Load dataset
df = pd.read_csv(r"C:\Users\ADMIN\Documents\Data Science\Projects\clustering\cleaned_development_dataset.csv")

# ---- CHANGE THIS if your column name is different ----
country_col = "Country"   # e.g., "Country" or "Country Name"

st.subheader("Dataset Preview")
st.write(df.head())

# Numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Feature selection
features = st.multiselect(
    "Select 2 Features",
    numeric_cols,
    default=numeric_cols[:2]
)

k = 3  # Developed / Developing / Underdeveloped

if len(features) == 2:

    X = df[features]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # -----------------------------
    # 🔥 Cluster → Development Mapping
    # -----------------------------
    cluster_means = df.groupby('Cluster')[features[0]].mean()
    sorted_clusters = cluster_means.sort_values().index.tolist()

    cluster_labels = {
        sorted_clusters[0]: "Underdeveloped",
        sorted_clusters[1]: "Developing",
        sorted_clusters[2]: "Developed"
    }

    df['Development'] = df['Cluster'].map(cluster_labels)

    # -----------------------------
    # 🔹 COUNTRY DROPDOWN
    # -----------------------------
    st.subheader("Select Existing Country")

    selected_country = st.selectbox(
        "Choose a country",
        df[country_col].unique()
    )

    country_data = df[df[country_col] == selected_country].iloc[0]

    st.write("### Selected Country Data")
    st.write(country_data[features])

    # Predict selected country
    selected_scaled = scaler.transform([country_data[features]])
    selected_cluster = kmeans.predict(selected_scaled)[0]
    selected_dev = cluster_labels[selected_cluster]

    st.success(f"🌍 {selected_country} is **{selected_dev}** (Cluster {selected_cluster})")

    # -----------------------------
    # 🔹 OPTIONAL: MANUAL INPUT
    # -----------------------------
    st.subheader("Or Enter Custom Country")

    custom_name = st.text_input("Custom Country Name", "New Country")

    val1 = st.number_input(features[0], value=float(X[features[0]].mean()))
    val2 = st.number_input(features[1], value=float(X[features[1]].mean()))

    predict_custom = st.button("Predict Custom Country")

    if predict_custom:
        new_point = np.array([[val1, val2]])
        new_scaled = scaler.transform(new_point)
        custom_cluster = kmeans.predict(new_scaled)[0]
        custom_dev = cluster_labels[custom_cluster]

        st.info(f"🌍 {custom_name} is **{custom_dev}** (Cluster {custom_cluster})")

    # -----------------------------
    # 🔹 VISUALIZATION
    # -----------------------------
    colors = cm.get_cmap('tab10', k)

    plt.figure(figsize=(8,6))

    # Plot clusters
    for cluster in range(k):
        subset = df[df['Cluster'] == cluster]
        plt.scatter(
            subset[features[0]],
            subset[features[1]],
            label=f"{cluster_labels[cluster]}",
            color=colors(cluster)
        )

    # Highlight selected country
    plt.scatter(
        country_data[features[0]],
        country_data[features[1]],
        color='black',
        marker='X',
        s=200
    )

    plt.text(
        country_data[features[0]],
        country_data[features[1]],
        selected_country,
        fontsize=10
    )

    # Highlight custom country if used
    if predict_custom:
        plt.scatter(val1, val2, color='red', marker='D', s=150)
        plt.text(val1, val2, custom_name, fontsize=10)

    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title("Country Development Clusters")
    plt.legend()
    plt.grid(True)

    st.pyplot(plt)

else:
    st.warning("Please select exactly 2 features.")