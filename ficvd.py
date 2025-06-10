import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Cardiovascular Disease Frailty Index Calculator",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

df = pd.read_csv('./data.csv')
fis = df['FICVD'].values

# --- 2. Data Definition (All English) ---
# A list of dictionaries defining each feature, its coefficient, and UI widget properties.
features_data = [
    {
        "label": "Self-rated health",
        "coefficient": 0.717533359,
        "widget": "selectbox",
        "options": {"Excellent": 0, "Good": 0.25, "Fair": 0.5, "Poor": 1}
    },
    {
        "label": "Frequency of tiredness / lethargy in last two weeks",
        "coefficient": 0.082581976,
        "widget": "selectbox",
        "options": {"Not at all": 0, "Several days": 0.25, "More than half the days": 0.5, "Nearly every day": 1}
    },
    {
        "label": "Falls in last year",
        "coefficient": 0.069505305,
        "widget": "selectbox",
        "options": {"No falls": 0, "Only one fall": 0.5, "More than one fall": 1}
    },
    {
        "label": "Frequency of depressed feelings in last two weeks",
        "coefficient": 0.00802942,
        "widget": "selectbox",
        "options": {"Not at all": 0, "Several days": 0.5, "More than half the days": 0.75, "Nearly every day": 1}
    },
    # --- Binary (Yes/No) questions ---
    {"label": "High blood pressure", "coefficient": 0.357163883, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Chest pain", "coefficient": 0.243212157, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Diabetes", "coefficient": 0.199196198, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Wheezing or whistling in the chest in last year", "coefficient": 0.184625756, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Long-standing illness, disability or infirmity", "coefficient": 0.161556333, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Chronic bronchitis or emphysema", "coefficient": 0.157243385, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Hiatus hernia", "coefficient": 0.153806021, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Deep-vein thrombosis (DVT)", "coefficient": 0.1329178, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "High cholesterol", "coefficient": 0.108453366, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Osteoarthritis", "coefficient": 0.068945754, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Pain all over the body", "coefficient": 0.062731266, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Hip pain", "coefficient": 0.04856814, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Cataracts", "coefficient": 0.038964908, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Dental problems", "coefficient": 0.037521928, "widget": "radio", "options": {"None": 0, "Any": 1}},
    {"label": "Gastric reflux / heartburn", "coefficient": 0.030780686, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Back pain", "coefficient": 0.024900588, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Feelings of loneliness", "coefficient": 0.01792041, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Any cancer diagnosis", "coefficient": 0.017803166, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Fractures or broken bones in last five years", "coefficient": 0.017730483, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Knee pain", "coefficient": 0.016566061, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Gout", "coefficient": 0.010983089, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Rheumatoid arthritis", "coefficient": 0.007961687, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Stomach or abdominal pain", "coefficient": 0.003288348, "widget": "radio", "options": {"No": 0, "Yes": 1}},
    {"label": "Hayfever, allergic rhinitis or eczema", "coefficient": -0.031680303, "widget": "radio", "options": {"No": 0, "Yes": 1}},
]



# --- 4. App Layout and User Interface ---
st.title("Cardiovascular Disease-Specific Frailty Index (CVD-FI) Calculator")

with st.expander("ℹ️ About this Calculator"):
    st.markdown("""
        This tool calculates a frailty index based on the model from the manuscript **"Development and evaluation of a cardiovascular disease-specific frailty index: A machine learning-based analysis of the UK Biobank"**.

        - **How to use:** Please answer the questions below based on your current health status.
        - **Calculation:** The final score is the sum of the weighted values of your selected answers. A higher score indicates a greater accumulation of health deficits.
        - **Disclaimer:** This is a research tool and is **not a substitute for professional medical advice, diagnosis, or treatment.**
    """)

st.subheader("Please fill out the following health questionnaire:")

# Use a form to group inputs and have a single submission button
with st.form(key='health_form'):
    user_inputs = {}
    # Create a two-column layout for the questions
    col1, col2 = st.columns(2, gap="large")

    for i, feature in enumerate(features_data):
        target_col = col1 if i < len(features_data) / 2 else col2
        with target_col:
            if feature["widget"] == "selectbox":
                user_inputs[feature["label"]] = st.selectbox(
                    label=feature["label"],
                    options=list(feature["options"].keys()),
                    key=f"sb_{i}" # Unique key for each widget
                )
            elif feature["widget"] == "radio":
                user_inputs[feature["label"]] = st.radio(
                    label=feature["label"],
                    options=list(feature["options"].keys()),
                    horizontal=True,
                    key=f"rd_{i}" # Unique key for each widget
                )

    # Form submission button
    submitted = st.form_submit_button(
        label="**Calculate My Frailty Index Score**",
        type="primary",
        use_container_width=True
    )


# --- 3. Visualization Function ---
def create_gauge_chart(all_scores):
    # 得到 最小和最大分数
    min_score = np.min(all_scores)
    max_score = np.max(all_scores)
    # 计算三分位数，1/3, 2/3
    q1 = np.quantile(all_scores, 1/3)
    q2 = np.quantile(all_scores, 2/3)

    # 最终分数是 all_scores 的最后一个元素
    score = all_scores[-1]

    # """Creates a Plotly gauge chart to visualize the score."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "CVD Frailty Index Score", 'font': {'size': 20}},

        gauge = {
            'axis': {'range': [min_score, max_score], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#262730"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_score, q1], 'color': 'lightgreen'},
                {'range': [q1, q2], 'color': 'yellow'},
                {'range': [q2, max_score], 'color': 'orange'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 5},
                'thickness': 0.85,
                'value': score
            }
        }
    ))
    fig.update_layout(
        font = {'color': "darkblue", 'family': "Arial"},
        height=350
    )
    return fig

# --- 5. Calculation and Results Display ---
def quantile_normalization(x, target_distribution=norm.ppf):
    x = np.asarray(x)
    ranks = np.argsort(np.argsort(x)) + 1  # 得到秩（从1开始）
    quantiles = ranks / (len(x) + 1)       # 计算分位数
    normalized = target_distribution(quantiles)  # 映射到目标分布
    return normalized


if submitted:
    total_score = 0
    calculation_details = []

    for feature in features_data:
        label = feature["label"]
        coefficient = feature["coefficient"]
        user_selection = user_inputs[label]
        numerical_value = feature["options"][user_selection]
        item_score = numerical_value * coefficient
        total_score += item_score

        calculation_details.append({
            "Feature": label,
            "Your Answer": user_selection,
            "Encoded Value": numerical_value,
            "Coefficient": coefficient,
            "Sub-score": item_score
        })

    # Normalize the scores using quantile normalization
    combined_scores = np.concatenate([fis, [total_score]])
    normalized_scores = quantile_normalization(combined_scores)
    total_score = normalized_scores[-1]

    print(f"total_score: {total_score}")

    st.markdown("---")
    st.header("Results", anchor=False)

    # Display results in two columns: one for the gauge, one for the interpretation
    res_col1, res_col2 = st.columns([1, 1], gap="large")

    with res_col1:
        st.metric(label="Calculated CVD-Frailty Index Score", value=f"{total_score:.4f}")
        st.markdown("""
        **Interpretation:**
        The CVD-FI score represents a cumulative tally of health deficits.

        - A **lower score** (green zone, bottom 1/3) indicates fewer deficits and lower risk of cardiovascular disease (reference group).
        - A **medium score** (yellow zone, middle 1/3) is associated with a **27% higher risk** of incident CVD compared to the low group (HR=1.27, 95% CI: 1.23–1.31).
        - A **higher score** (orange zone, top 1/3) is associated with an **81% higher risk** of incident CVD compared to the low group (HR=1.81, 95% CI: 1.74–1.86).

        *Please discuss this result and its implications with a qualified healthcare professional.*
        """)

    with res_col2:
        gauge_chart = create_gauge_chart(all_scores=normalized_scores)
        st.plotly_chart(gauge_chart, use_container_width=True)
        # 这里我希望说明的是 不同颜色代表的是不同风险级别，分别是 低风险（FI指数低1/3区间），中风险，高风险
        # , individuals were divided into low (bottom 1/3), median (1/3-2/3), and high (top 1/3) FI/FICVD groups based on the distribution of FI/FICVD
        st.caption("Note: The gauge chart visualizes your CVD-FI score relative to the distribution of scores from the UK Biobank dataset. Green, yellow, and orange segments represent low (bottom 1/3), medium (middle 1/3), and high (top 1/3) risk levels, respectively.")
        # st.caption("Note: The color-coded risk levels (green, yellow, orange, red) on the gauge are for illustrative purposes only.")

    # with st.expander("View Detailed Calculation Breakdown"):
    #     details_df = pd.DataFrame(calculation_details)
    #     st.dataframe(details_df.style.format({
    #         "Encoded Value": "{:.2f}",
    #         "Coefficient": "{:.7f}",
    #         "Sub-score": "{:.7f}"
    #     }))