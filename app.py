import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Page Config (Must be the very first Streamlit command)
st.set_page_config(page_title="EduPro Dashboard",page_icon="🎓", layout="wide",
    initial_sidebar_state="expanded")

# 2. Inject Custom CSS for "Neo-Tech Minimalism" Theme
st.markdown(
    """
    <style>
    /* Global background gradient mimicking the screenshot */
    .stApp {
        background: radial-gradient(circle at 20% 30%, #0d1127 0%, #070913 100%);
        color: #e0e6ed;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ─── FIXES FOR THE WHITE TOP BAR ─── */
    /* Hides the default Streamlit header bar completely */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    /* Removes the colored decoration line at the very top of the page */
    div[data-testid="stDecoration"] {
        background: transparent !important;
        display: none !important;
    }

    /* Adjust main content padding so it doesn't get cut off by the transparent header */
    div[data-testid="stMainBlockContainer"] {
        padding-top: 2rem !important;
    }
    /* ─────────────────────────────────── */
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(13, 17, 39, 0.7) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(0, 242, 254, 0.15);
    }
    
    /* Headers with vibrant cyber colors */
    h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        text-shadow: 0 0 20px rgba(0, 242, 254, 0.3);
        margin-bottom: 2rem !important;
    }
    h2, h3 {
        color: #00f2fe !important;
        font-weight: 600 !important;
    }
    
    /* Translucent Glassmorphism Card Info Box */
    .stInfo {
        background-color: rgba(20, 24, 52, 0.4) !important;
        border: 1px solid rgba(138, 43, 226, 0.3) !important;
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        color: #d1d5db !important;
    }
    
    /* Customization for upload wrapper area */
    div[data-testid="stFileUploader"] {
        background-color: rgba(25, 30, 62, 0.5);
        border: 1px dashed rgba(0, 242, 254, 0.4);
        border-radius: 8px;
        padding: 10px;
    }

    /* Style Streamlit Tabs to match the dark sci-fi aesthetic */
    button[data-baseweb="tab"] {
        color: #8a99ad !important;
        background-color: transparent !important;
        font-weight: 500;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00f2fe !important;
        border-bottom-color: #00f2fe !important;
    }
    
    /* Metrics display adjustments */
    div[data-testid="stMetric"] {
        background-color: rgba(20, 24, 52, 0.6);
        border: 1px solid rgba(0, 242, 254, 0.1);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricValue"] {
        color: #00f2fe !important;
    }

    /* Metric label — multiple selectors to guarantee override */
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label p,
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] div,
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        -webkit-text-fill-color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Student Segmentation & Recommendation System")

# ─────────────────────────────────────────────
# Helper: run the full ML pipeline on raw data
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Running segmentation model…")
def run_pipeline(file_bytes: bytes):
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    users        = xls.parse("Users")
    courses      = xls.parse("Courses")
    transactions = xls.parse("Transactions")

    trans_courses = transactions.merge(courses, on="CourseID", how="left")
    df            = trans_courses.merge(users,  on="UserID",   how="left")

    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    courses_enrolled       = df.groupby("UserID")["CourseID"].count()
    total_courses_catgories = df.groupby("UserID")["CourseCategory"].nunique()
    avg_courses_per_category = courses_enrolled / total_courses_catgories
    frequency = df.groupby("UserID")["TransactionDate"].nunique()

    preferred_category = df.groupby("UserID")["CourseCategory"].agg(lambda x: x.mode()[0])
    preferred_level = df.groupby("UserID")["CourseLevel"].agg(lambda x: x.mode()[0])
    avg_rating  = df.groupby("UserID")["CourseRating"].mean()
    avg_spending = df.groupby("UserID")["Amount"].mean()
    diversity   = df.groupby("UserID")["CourseCategory"].nunique()

    advanced_ratio = df[df["CourseLevel"] == "Advanced"].groupby("UserID")["CourseID"].count()
    beginner_ratio = df[df["CourseLevel"] == "Beginner"].groupby("UserID")["CourseID"].count()
    learning_depth = advanced_ratio / (beginner_ratio + 1)

    learner_profiles = pd.DataFrame({
        "TotalCourses":          courses_enrolled,
        "AvgCoursesPerCategory": avg_courses_per_category,
        "EnrollmentFrequency":   frequency,
        "PreferredCategory":     preferred_category,
        "PreferredLevel":        preferred_level,
        "AvgRating":             avg_rating,
        "AvgSpending":           avg_spending,
        "DiversityScore":        diversity,
        "LearningDepth":         learning_depth,
    }).reset_index()

    learner_profiles.fillna(0, inplace=True)
    le = LabelEncoder()
    learner_profiles["PreferredCategory"] = le.fit_transform(learner_profiles["PreferredCategory"].astype(str))
    learner_profiles["PreferredLevel"] = le.fit_transform(learner_profiles["PreferredLevel"].astype(str))
    learner_profiles["TotalCourses"] = pd.to_numeric(learner_profiles["TotalCourses"])
    learner_profiles = learner_profiles[learner_profiles["TotalCourses"] >= 3]

    features = learner_profiles.drop("UserID", axis=1)
    scaler   = StandardScaler()
    X        = scaler.fit_transform(features)

    inertia = []
    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X)
        inertia.append(km.inertia_)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
    learner_profiles["Cluster"] = kmeans.fit_predict(X).astype(str)
    score = silhouette_score(X, learner_profiles["Cluster"])

    pca_components = PCA(n_components=2).fit_transform(X)
    learner_profiles["PCA1"] = pca_components[:, 0]
    learner_profiles["PCA2"] = pca_components[:, 1]

    return learner_profiles, df, score, inertia, features, X, kmeans


# ─────────────────────────────────────────────
# File uploader
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=60)
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload EduPro_Dataset.xlsx",
    type=["xlsx"],
    help="File must contain sheets: Users, Courses, Transactions",
)

if uploaded_file is None:
    st.info(
        "👈 **Upload your EduPro dataset (.xlsx)** from the sidebar to get started.\n\n"
        "The workbook must contain three sheets with these exact names:\n"
        "- **Users** —  UserID,Age,Gender\n"
        "- **Courses** — CourseID,CourseCategory,CourseType,CourseLevel,CourseRating\n"
        "- **Transactions** — UserID,CourseID,TransactionDate,Amount"
    )
    st.stop()

# ── Run pipeline (cached on file content) ────
try:
    learner_profiles, full_data, score, inertia, features, X, kmeans = run_pipeline(
        uploaded_file.getvalue()
    )
except Exception as e:
    st.error(
        f"**Could not process the uploaded file.**\n\n"
        f"Make sure the sheet names are exactly `Users`, `Courses`, and `Transactions` "
        f"and all required columns are present.\n\n`{e}`"
    )
    st.stop()

st.sidebar.success(f"✅ Loaded {learner_profiles.shape[0]:,} learner profiles")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🧩 Cluster Dashboard",
    "🔍 Learner Explorer",
    "🎯 Recommendations",
    "📈 Model Validation",
])

# ── Tab 1 – Overview ─────────────────────────
with tab1:
    st.header("Overview Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Learners",    learner_profiles.shape[0])
    col2.metric("Total Transactions", full_data.shape[0])
    col3.metric("Silhouette Score",  round(score, 3))

    category_counts = full_data["CourseCategory"].value_counts()
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Course Category Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    spending = full_data.groupby("CourseCategory")["Amount"].sum()
    fig = px.bar(
        x=spending.index, y=spending.values,
        title="Category-wise Spending",
        labels={"x": "Category", "y": "Total Spending"},
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2 – Cluster Dashboard ─────────────────
with tab2:
    st.header("Cluster Dashboard")

    fig = px.scatter(
        learner_profiles,
        x="PCA1", y="PCA2",
        color="Cluster",
        hover_data=["UserID"],
        title="Learner Segments (PCA)",
    )
    st.plotly_chart(fig, use_container_width=True)

    cluster_summary = learner_profiles.groupby("Cluster").mean(numeric_only=True)
    st.subheader("Cluster Summary")
    st.dataframe(cluster_summary)

    fig = px.box(
        learner_profiles,
        x="Cluster", y="AvgSpending",
        color="Cluster",
        title="Cluster Spending Comparison",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 3 – Learner Explorer ──────────────────
with tab3:
    st.header("Learner Explorer")

    selected_user = st.selectbox("Select User", learner_profiles["UserID"])

    user_profile = learner_profiles[learner_profiles["UserID"] == selected_user]
    st.subheader("Learner Profile")
    st.dataframe(user_profile)

    history = full_data[full_data["UserID"] == selected_user]
    st.subheader("Enrollment History")
    st.dataframe(history)

    if not history.empty:
        timeline = (
            history.groupby("TransactionDate")["Amount"].sum().reset_index()
        )
        fig = px.line(
            timeline,
            x="TransactionDate", y="Amount",
            title="Spending Timeline",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No transaction history for this user.")

# ── Tab 4 – Recommendations ───────────────────
with tab4:
    st.header("Course Recommendations")

    def recommend_courses(user_id):
        user_cluster = learner_profiles.loc[
            learner_profiles["UserID"] == user_id, "Cluster"
        ].values[0]

        cluster_users = learner_profiles[
            learner_profiles["Cluster"] == user_cluster
        ]["UserID"]

        cluster_transactions = full_data[full_data["UserID"].isin(cluster_users)]

        recommendations = (
            cluster_transactions.groupby(["CourseID", "CourseCategory", "CourseLevel"])
            .size()
            .reset_index(name="Popularity")
            .sort_values("Popularity", ascending=False)
            .head(5)
        )
        return recommendations

    st.subheader(f"Top 5 recommendations for user **{selected_user}**")
    st.dataframe(recommend_courses(selected_user))

# ── Tab 5 – Model Validation ──────────────────
with tab5:
    st.header("Model Validation")

    fig = px.line(
        x=list(range(2, 10)), y=inertia,
        markers=True,
        title="Elbow Curve",
        labels={"x": "Number of Clusters (k)", "y": "Inertia"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Silhouette Score", round(score, 3))

    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(features.corr(), annot=True, ax=ax, fmt=".2f")
    st.pyplot(fig)

    csv = learner_profiles.to_csv(index=False)
    st.download_button(
        label="⬇ Download Learner Profiles",
        data=csv,
        file_name="learner_profiles.csv",
        mime="text/csv",
    )