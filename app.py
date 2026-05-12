import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Student Analytics Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI';
}

.main {
    background-color: #f4f7fb;
}

.hero-title {
    font-size: 52px;
    font-weight: 800;
    color: #132043;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 20px;
    color: #4b5563;
    margin-bottom: 30px;
}

.metric-box {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.08);
}

.prediction-card {
    background: linear-gradient(135deg,#132043,#1f3b73);
    padding: 30px;
    border-radius: 20px;
    color: white;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.15);
}

.recommend-box {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.08);
}

.sidebar .sidebar-content {
    background-color: #ffffff;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="hero-title">🎓 AI Student Analytics Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">Advanced Machine Learning Based Student Performance Prediction System</div>',
    unsafe_allow_html=True
)

# =========================================================
# DATA GENERATION
# =========================================================
np.random.seed(42)

rows = 1000

study_hours = np.random.randint(1, 12, rows)
attendance = np.random.randint(45, 100, rows)
previous_marks = np.random.randint(35, 100, rows)
sleep_hours = np.random.randint(4, 10, rows)
assignment_score = np.random.randint(30, 100, rows)
internet_usage = np.random.randint(1, 10, rows)
participation = np.random.randint(1, 10, rows)
mental_wellness = np.random.randint(1, 10, rows)
extra_activities = np.random.randint(1, 10, rows)

final_marks = (
    study_hours * 3.2 +
    attendance * 0.28 +
    previous_marks * 0.42 +
    sleep_hours * 1.4 +
    assignment_score * 0.25 -
    internet_usage * 1.3 +
    participation * 2.1 +
    mental_wellness * 1.5 +
    extra_activities * 0.9 +
    np.random.normal(0, 5, rows)
)

final_marks = np.clip(final_marks, 35, 100)

# =========================================================
# DATAFRAME
# =========================================================
data = pd.DataFrame({

    "Study Hours": study_hours,
    "Attendance": attendance,
    "Previous Marks": previous_marks,
    "Sleep Hours": sleep_hours,
    "Assignment Score": assignment_score,
    "Internet Usage": internet_usage,
    "Participation": participation,
    "Mental Wellness": mental_wellness,
    "Extra Activities": extra_activities,
    "Final Marks": final_marks

})

# =========================================================
# MACHINE LEARNING
# =========================================================
X = data.drop("Final Marks", axis=1)
y = data["Final Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Dashboard",
        "🎯 Prediction",
        "📊 Analytics",
        "📁 Dataset",
        "🤖 AI Insights",
        "📘 About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success(f"Model Accuracy: {accuracy:.2f}")
st.sidebar.info(f"MAE: {mae:.2f}")
st.sidebar.warning(f"RMSE: {rmse:.2f}")

# =========================================================
# DASHBOARD
# =========================================================
if page == "🏠 Dashboard":

    st.subheader("📈 Overall Student Analytics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Students",
        len(data)
    )

    c2.metric(
        "Average Marks",
        round(data["Final Marks"].mean(), 2)
    )

    c3.metric(
        "Highest Score",
        round(data["Final Marks"].max(), 2)
    )

    c4.metric(
        "Lowest Score",
        round(data["Final Marks"].min(), 2)
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            data,
            x="Final Marks",
            nbins=20,
            title="Distribution of Final Marks"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig2 = px.scatter(
            data,
            x="Study Hours",
            y="Final Marks",
            color="Attendance",
            size="Participation",
            title="Study Hours vs Final Marks"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

# =========================================================
# PREDICTION PAGE
# =========================================================
elif page == "🎯 Prediction":

    st.subheader("🧠 Student Performance Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:

        study_input = st.slider(
            "Study Hours",
            1,
            12,
            5
        )

        attendance_input = st.slider(
            "Attendance (%)",
            40,
            100,
            80
        )

        previous_marks_input = st.slider(
            "Previous Marks",
            0,
            100,
            70
        )

    with col2:

        sleep_input = st.slider(
            "Sleep Hours",
            3,
            10,
            7
        )

        assignment_input = st.slider(
            "Assignment Score",
            0,
            100,
            75
        )

        internet_input = st.slider(
            "Internet Usage",
            0,
            12,
            4
        )

    with col3:

        participation_input = st.slider(
            "Participation",
            1,
            10,
            6
        )

        wellness_input = st.slider(
            "Mental Wellness",
            1,
            10,
            7
        )

        activity_input = st.slider(
            "Extra Activities",
            1,
            10,
            5
        )

    if st.button("🚀 Predict Performance"):

        input_data = np.array([[
            study_input,
            attendance_input,
            previous_marks_input,
            sleep_input,
            assignment_input,
            internet_input,
            participation_input,
            wellness_input,
            activity_input
        ]])

        predicted_marks = model.predict(input_data)[0]

        predicted_marks = round(predicted_marks, 2)

        # PERFORMANCE LEVEL
        if predicted_marks >= 85:

            performance = "Excellent 🌟"

            recommendation = """
            Student is performing exceptionally well.
            Maintain consistency and continue advanced learning.
            """

        elif predicted_marks >= 70:

            performance = "Good 👍"

            recommendation = """
            Performance is strong.
            Focus more on revision and time management.
            """

        elif predicted_marks >= 50:

            performance = "Average ⚠️"

            recommendation = """
            Student needs improvement in study consistency.
            Reduce distractions and improve attendance.
            """

        else:

            performance = "Needs Improvement ❌"

            recommendation = """
            Immediate academic support recommended.
            Increase study hours and assignment completion.
            """

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted Score",
            f"{predicted_marks}%"
        )

        col2.metric(
            "Performance",
            performance
        )

        col3.metric(
            "Attendance",
            f"{attendance_input}%"
        )

        st.markdown("")

        st.markdown(
            f'''
            <div class="prediction-card">
            <h2>📌 AI Recommendation</h2>
            <p style="font-size:18px;">
            {recommendation}
            </p>
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.markdown("---")

        # GAUGE CHART
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_marks,
            title={'text': "Performance Meter"},
            gauge={
                'axis': {'range': [0, 100]}
            }
        ))

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

# =========================================================
# ANALYTICS
# =========================================================
elif page == "📊 Analytics":

    st.subheader("📊 Advanced Analytics Dashboard")

    feature_importance = pd.DataFrame({

        "Feature": X.columns,
        "Importance": model.feature_importances_

    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    col1, col2 = st.columns(2)

    with col1:

        fig1 = px.bar(
            feature_importance,
            x="Feature",
            y="Importance",
            title="Feature Importance"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with col2:

        corr = data.corr()

        fig2 = px.imshow(
            corr,
            text_auto=True,
            title="Correlation Heatmap"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    fig3 = px.box(
        data,
        y="Final Marks",
        title="Marks Distribution Analysis"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# =========================================================
# DATASET
# =========================================================
elif page == "📁 Dataset":

    st.subheader("📁 Student Dataset")

    st.dataframe(
        data,
        use_container_width=True
    )

    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Dataset",
        data=csv,
        file_name="student_dataset.csv",
        mime="text/csv"
    )

# =========================================================
# AI INSIGHTS
# =========================================================
elif page == "🤖 AI Insights":

    st.subheader("🤖 AI Generated Insights")

    avg_marks = data["Final Marks"].mean()

    top_feature = feature_importance.iloc[0]["Feature"]

    st.markdown(f"""

    ### 📌 Key Insights

    - Average student marks are **{avg_marks:.2f}%**
    - Most influential factor is **{top_feature}**
    - Higher attendance generally improves academic performance
    - Excessive internet usage negatively impacts marks
    - Students with higher participation tend to score better

    ### 🎯 Recommendations

    ✅ Improve attendance tracking  
    ✅ Encourage assignment completion  
    ✅ Focus on mental wellness  
    ✅ Promote balanced internet usage  
    ✅ Increase classroom participation

    """)

# =========================================================
# ABOUT
# =========================================================
else:

    st.subheader("📘 About This Project")

    st.write("""

    ## AI Student Analytics Platform

    This project predicts student academic performance using
    Machine Learning and Data Analytics.

    ### Technologies Used
    - Python
    - Streamlit
    - Random Forest Regressor
    - Plotly Dashboard
    - Scikit-learn

    ### Features
    ✅ Professional Dashboard  
    ✅ AI-Based Prediction  
    ✅ Analytics & Visualization  
    ✅ Downloadable Dataset  
    ✅ AI Recommendations  
    ✅ Interactive UI  
    ✅ Performance Meter  
    ✅ Correlation Analysis  

    ### Objective
    To analyze student behavior and predict academic performance
    using Artificial Intelligence techniques.

    """)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.markdown(
    """
    <center>
    <h4>Made with ❤️ using Streamlit, AI & Machine Learning</h4>
    </center>
    """,
    unsafe_allow_html=True
)
