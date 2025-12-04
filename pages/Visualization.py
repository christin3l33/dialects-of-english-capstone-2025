import streamlit as st
import pandas as pd
import gdown
import re
from pathlib import Path
import plotly.express as px
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Dialect Change Over Time", layout="wide")
st.markdown("<h1 style='text-align: center;'>Visualization Page</h1>", unsafe_allow_html=True)
st.write("<p style='text-align: center; font-size: 1.3rem;'>Scroll to see interactive visualizations.", unsafe_allow_html=True)

@st.cache_data(show_spinner="Fetching data from Google Drive…")
def load_from_drive(_file_map):
    dfs = {}
    data_folder = Path("data")
    data_folder.mkdir(exist_ok=True)

    # Google Drive file ID pattern (handles any share link)
    file_id_pattern = r"(?:id=|/d/|open\?id=|file/d/)([A-Za-z0-9_-]{25,})"

    for name, link in _file_map.items():

        match = re.search(file_id_pattern, link)
        if match:
            file_id = match.group(1)
        else:
            file_id = link.strip()

        # Direct download URL
        url = f"https://drive.google.com/uc?id={file_id}"

        output = data_folder / f"{name}.csv"

        if not output.exists():
            gdown.download(url, str(output), quiet=False)
        try:
            dfs[name] = pd.read_csv(output, low_memory=False, on_bad_lines='skip')
        except pd.errors.ParserError:
            st.error(f"Could not parse {name}.csv. Make sure it is a valid CSV.")
            st.stop()
    return dfs


try:
    data = load_from_drive(st.secrets["drive_files"])
except KeyError:
    st.error("❌ Missing `drive_files` in secrets.toml! Add it under `[drive_files]`.")
    st.stop()

# Unpack data
questions = data["questions"]
choices = data["choices"]
users = data["users"]
responses = data["responses"]

raw_responses = responses.copy()

st.success("✅ All four datasets loaded successfully!")

st.markdown("---")
st.subheader("U.S. Dialect Word Usage Over Time")

soda_qid = 2
responses_soda = responses[responses["question_id"] == soda_qid].copy()

responses_soda = responses_soda.merge(
    users[["id", "year"]], left_on="user_id", right_on="id", how="left"
)

responses_soda = responses_soda.merge(
    choices[["id", "value"]], left_on="choice_id", right_on="id", how="left"
)

responses_soda["term"] = responses_soda["value"].combine_first(responses_soda["other"])

# Clean & Transform
responses_soda = responses_soda.dropna(subset=["year", "term"])
responses_soda["decade"] = (responses_soda["year"] // 10 * 10).astype(int)
responses_soda["term"] = responses_soda["term"].str.strip().str.lower()

# Aggregate
counts = (
    responses_soda.groupby(["decade", "term"])
    .size()
    .reset_index(name="count")
)

totals = counts.groupby("decade")["count"].transform("sum")
counts["percent"] = (counts["count"] / totals * 100).round(1)

# Get Top 5 Terms Overall 
top_terms = (
    counts.groupby("term")["count"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .index
    .tolist()
)

st.sidebar.header("Filter Controls")
selected_terms = st.sidebar.multiselect(
    "Select terms to display:",
    options=counts["term"].unique(),
    default=top_terms
)

filtered = counts[counts["term"].isin(selected_terms)]
fig = px.line(
    filtered,
    x="decade",
    y="percent",
    color="term",
    markers=True,
    labels={"percent": "% Respondents Using Term", "decade": "Birth Decade"},
    hover_data={
        "term": True,
        "percent": True,
        "decade": True,
    },
    title="Change in Word Usage Over Birth Decades (Top 5 Terms, Normalized by Birth Year)"
)

fig.update_traces(
    mode="lines+markers",
    hovertemplate="<b>%{customdata[0]}</b><br>Decade: %{x}<br>% Respondents: %{y:.1f}%<extra></extra>"
)
fig.update_layout(
    hovermode="closest",
    legend_title_text="Term",
    plot_bgcolor="white"
)

st.plotly_chart(fig, width='stretch')


QUESTION_ID = 2  # Only the "sweetened carbonated beverage" question
responses = responses[responses["question_id"] == QUESTION_ID].copy()

responses = responses.merge(
    users[["id", "year", "gender", "state"]],
    left_on="user_id",
    right_on="id",
    how="left"
)

responses = responses.merge(
    choices[["id", "value"]],
    left_on="choice_id",
    right_on="id",
    how="left"
)

responses["term"] = responses["value"].combine_first(responses["other"]).str.lower().str.strip()
responses = responses[responses["term"].isin(["soda", "pop"])]
responses = responses.dropna(subset=["state", "term"])

# Shannon Entropy Function
def shannon_entropy(series):
    counts = series.value_counts(normalize=True)
    return -np.sum(counts * np.log2(counts))

# Compute lexical diversity per state
entropy_by_state = (
    responses.groupby("state")["term"]
    .apply(shannon_entropy)
    .reset_index(name="entropy")
)

st.sidebar.header("Filter Controls")

min_year = int(responses["year"].min())
max_year = int(responses["year"].max())
year_range = st.sidebar.slider(
    "Filter by birth year:",
    min_year,
    max_year,
    value=(min_year, max_year)
)

# Gender filter
existing_genders = responses["gender"].dropna().unique().tolist()
all_genders = sorted(set(existing_genders + ["f","m","o","x"]))
gender_filter = st.sidebar.multiselect(
    "Filter by gender:",
    options=all_genders,
    default=all_genders
)

filtered = responses[
    (responses["year"].between(year_range[0], year_range[1])) &
    (responses["gender"].isin(gender_filter))
]
entropy_by_state = (
    filtered.groupby("state")["term"]
    .apply(shannon_entropy)
    .reset_index(name="entropy")
)

fig = px.choropleth(
    entropy_by_state,
    locations="state",
    locationmode="USA-states",
    color="entropy",
    color_continuous_scale="plasma",
    scope="usa",
    labels={"entropy": "Lexical Diversity (Shannon Entropy)"},
    hover_data={"state": True, "entropy": True}
)

fig.update_layout(
    geo=dict(bgcolor="rgba(0,0,0,0)"),
    coloraxis_colorbar=dict(title="Lexical Diversity"),
    margin=dict(l=10, r=10, t=60, b=10),
)

st.subheader("Lexical Diversity (Shannon Entropy) by U.S. State — Soda vs. Pop")
st.plotly_chart(fig, width='stretch')
st.markdown("""
**Shannon entropy** measures how diverse word choices are within each state  
(high = high diversity, no single dominant response; low = low diversity, one response dominates).  

👉 *This example is **ONLY SODA VS. POP!***
""")

# Roly Poly Question
st.markdown("---")
st.subheader("Roly Poly Question: Age Group Analysis")
st.write("What do you call a creature that rolls up into a ball when when you touch it?")

ROLY_POLY_QID = 21
responses_roly = raw_responses[raw_responses["question_id"] == ROLY_POLY_QID].copy()

# Merge with users to get age
responses_roly = responses_roly.merge(
    users[["id", "year"]],
    left_on="user_id",
    right_on="id",
    how="left"
)

# Merge with choices to get values
responses_roly = responses_roly.merge(
    choices[["id", "value"]],
    left_on="choice_id",
    right_on="id",
    how="left"
)

current_year = datetime.now().year
responses_roly["age"] = current_year - responses_roly["year"]

def categorize_age(age):
    if pd.isna(age):
        return None
    if age < 18:
        return 'Gen Z (Under 18)'
    elif age < 25:
        return 'Gen Z (18-24)'
    elif age < 35:
        return 'Millennial (25-34)'
    elif age < 45:
        return 'Millennial (35-44)'
    elif age < 55:
        return 'Gen X (45-54)'
    elif age < 65:
        return 'Boomer (55-64)'
    else:
        return 'Boomer (65+)'

responses_roly["age_group"] = responses_roly["age"].apply(categorize_age)

# Clean the term values
responses_roly["term"] = responses_roly["value"].combine_first(responses_roly["other"])
responses_roly["term"] = responses_roly["term"].str.lower().str.strip()

# Regex to match differnet variations of "roly poly"
responses_roly["term"] = responses_roly["term"].replace(
    to_replace=r'(?i)^(roly|rollie|rolly|roley)[\s\-]*poly.*$',
    value='roly poly',
    regex=True
)

responses_roly = responses_roly.dropna(subset=["age_group", "term"])

# Top choices
choice_counts = responses_roly["term"].value_counts()
top_choices = choice_counts[choice_counts >= len(responses_roly) * 0.05].index
responses_roly = responses_roly[responses_roly["term"].isin(top_choices)]

contingency = pd.crosstab(responses_roly["age_group"], responses_roly["term"])

# Youngest to oldest
age_order = [
    'Gen Z (Under 18)',
    'Gen Z (18-24)',
    'Millennial (25-34)',
    'Millennial (35-44)',
    'Gen X (45-54)',
    'Boomer (55-64)',
    'Boomer (65+)'
]
age_order_present = [age for age in age_order if age in contingency.index]
contingency = contingency.reindex(age_order_present)

contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Usage by Age Group")
    bar_data = contingency_pct.reset_index().melt(
        id_vars="age_group",
        var_name="term",
        value_name="percentage"
    )
    
    fig_bar = px.bar(
        bar_data,
        x="term",
        y="percentage",
        color="age_group",
        barmode="group",
        labels={
            "percentage": "% Using Term",
            "term": "Dialect Term",
            "age_group": "Age Group"
        },
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig_bar.update_layout(
        xaxis_title="Dialect Term",
        yaxis_title="Percentage Using Term (%)",
        legend_title="Age Group",
        hovermode="closest",
        yaxis_range=[0, 100],
        height=500
    )
    
    st.plotly_chart(fig_bar, width='stretch')

with col2:
    st.markdown("#### Overall Response Distribution")
    total_counts = responses_roly["term"].value_counts()
    
    fig_pie = px.pie(
        values=total_counts.values,
        names=total_counts.index,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig_pie.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig_pie.update_layout(
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig_pie, width='stretch')