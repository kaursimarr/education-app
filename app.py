# Step 0: Imports
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px

# Full-screen layout
st.set_page_config(page_title="Favourable Abroad Education", layout="wide")

# Step 1: Load datasets
@st.cache_data
def load_data():
    indian_students = pd.read_csv("IndianStudentsAbroad.csv")
    cost_living = pd.read_csv("Cost_of_Living_Index_by_Country_2024.csv")
    tuition = pd.read_csv("International_Education_Costs.csv")
    reputation = pd.read_csv("top 100 world university 2024 new.csv")
    return indian_students, cost_living, tuition, reputation

indian_students, cost_living, tuition, reputation = load_data()

# Step 2: Clean column names
for df_ in [indian_students, cost_living, tuition, reputation]:
    df_.columns = df_.columns.str.strip().str.lower().str.replace(" ", "_")

# Step 3: Merge datasets
df = tuition.merge(cost_living, on="country", how="left") \
            .merge(indian_students, on="country", how="left") \
            .merge(reputation, on="university", how="left")

# Step 4: Fill missing data
for col in ["visa_fee_usd", "insurance_usd"]:
    if col not in df.columns:
        df[col] = 0
    else:
        df[col] = df[col].fillna(0)

df.fillna(df.median(numeric_only=True), inplace=True)

# Step 5: Feature Engineering
df["total_cost"] = (df["tuition_usd"] * df["duration_years"]) + \
                   (df["rent_usd"] * df["duration_years"]) + \
                   df["visa_fee_usd"] + df["insurance_usd"]

df["roi_score"] = (df["employer_reputation"] * df["employment_outcomes"]) / (df["total_cost"] + 1)

# Normalize
df["roi_norm"] = df["roi_score"] / df["roi_score"].max()
df["acad_norm"] = df["academic_reputation"] / df["academic_reputation"].max()
df["cost_norm"] = 1 - (df["total_cost"] / df["total_cost"].max())

# Step 6: Streamlit UI
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🌍 Favourable Education Abroad for Indian Students</h1>", unsafe_allow_html=True)

# Step 7: Inputs (Left) & Outputs (Right)
col1, col2 = st.columns([1, 2])

with col1:
    # Program filter
    program_options = df["program"].unique()
    selected_program = st.selectbox("Select your program/degree", program_options)

    # Student preference
    choice = st.radio("What matters most to you?", ["Affordability", "Reputation", "Both"])

    if choice == "Affordability":
        cost_w, acad_w, roi_w = 0.7, 0.2, 0.1
    elif choice == "Reputation":
        cost_w, acad_w, roi_w = 0.1, 0.6, 0.3
    else:  # Both
        cost_w, acad_w, roi_w = 0.3, 0.3, 0.4

# Step 7B: Compute Scores
df["student_score"] = (
    roi_w * df["roi_norm"] +
    acad_w * df["acad_norm"] +
    cost_w * df["cost_norm"]
)

df_filtered = df[df["program"] == selected_program]
df_filtered = df_filtered.sort_values(by="student_score", ascending=False).reset_index(drop=True)

with col2:
    st.subheader("🏫 Top Universities for this Program")
    st.dataframe(df_filtered[["country", "university", "student_score"]].head(20))

# Step 7C: Country-level results
country_scores = df_filtered.groupby("country")["student_score"].mean().reset_index()
country_scores["edge_weight"] = 1 - country_scores["student_score"]
country_scores = country_scores.sort_values(by="edge_weight").reset_index(drop=True)

with col2:
    st.subheader("🌍 Top Countries for this Program")
    st.dataframe(country_scores[["country", "student_score"]].head(20))

    # Bar Chart
    fig = px.bar(country_scores.head(10),
                 x="country", y="student_score",
                 title="Top Countries by Student Score",
                 text_auto=".2f")
    st.plotly_chart(fig, use_container_width=True)

        # Pie Chart (cost breakdown)# Step 7C: Pie Chart (cost breakdown including cost of living)
    cost_breakdown = df_filtered[["tuition_usd", "rent_usd", "visa_fee_usd", "insurance_usd", "cost_of_living_index"]].mean()

    cost_breakdown["cost_of_living_index"] = cost_breakdown["cost_of_living_index"] * 1000  # scale as needed

    fig_pie = px.pie(
        values=cost_breakdown.values,
        names=cost_breakdown.index,
        title="Average Cost Distribution (Including Cost of Living)"
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# Step 8: Graph (NetworkX)
G = nx.DiGraph()
for _, row in country_scores.iterrows():
    G.add_edge("India", row["country"], weight=row["edge_weight"])

st.subheader("🔗 Country Graph Connections from India")
st.write("Nodes:", list(G.nodes))
st.write("Edges with weights:")
for u, v, w in G.edges(data=True):
    st.write(f"{u} -> {v}, weight = {w['weight']:.3f}")

# Best country
best_country = min(G["India"], key=lambda x: G["India"][x]["weight"])
st.metric(label="✅ Most Favourable Country", value=best_country)
