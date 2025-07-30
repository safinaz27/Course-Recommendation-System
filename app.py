import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ------------- Page Config -------------
st.set_page_config(page_title="📚 Smart Course Recommender", layout="wide")
st.markdown("<h1 style='color:#4CAF50; text-align:center;'>🎯 Smart Course Recommender</h1>", unsafe_allow_html=True)

# ------------- Background Style -------------
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    
    /* Force Light Theme */
    html, body, [class*="st-"] {
        background-color: white !important;
        color: black !important;
    }
    .stApp {
        background-color: White !important;
    }
    .css-1d391kg, .css-1v0mbdj {
        background-color: white !important;
        color: black !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f6f6f6 !important;
    }
    [data-testid="stSidebar"] * {
        background-color: transparent !important;
        color: gray !important;
    }
    
    /* Clickable card styles */
    .clickable-card {
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 15px;
        background-color: white;
        box-shadow: 2px 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 15px;
        height: 330px;
        overflow: auto;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .clickable-card:hover {
        transform: translateY(-5px);
        box-shadow: 4px 8px 20px rgba(0,0,0,0.15);
        border-color: #4CAF50;
        background-color: #f8fffe;
    }
    
    .recommendation-card {
        border: 1px solid #ccc;
        border-radius: 12px;
        padding: 15px;
        background-color: white;
        box-shadow: 2px 2px 12px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        height: 390px;
        overflow: auto;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: 4px 8px 20px rgba(0,0,0,0.15);
        border-color: #006400;
        background-color: #fffef8;
    }
    
    .click-indicator {
        position: absolute;
        top: 10px;
        right: 10px;
        background: #4CAF50;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 10px;
    }
    
    .rec-click-indicator {
        position: absolute;
        top: 10px;
        right: 10px;
        background: #006400;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 10px;
    }
    
    /* Style the select buttons */
    .select-btn {
        background: linear-gradient(135deg, #4CAF50, #45a049) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        padding: 8px 16px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 4px rgba(76, 175, 80, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin-top: 10px !important;
    }
    
    .select-btn:hover {
        background: linear-gradient(135deg, #45a049, #3d8b40) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.4) !important;
    }
    
    .rec-select-btn {
        background: linear-gradient(135deg, #006400, #004d00) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        padding: 8px 16px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 4px rgba(0, 100, 0, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin-top: 10px !important;
    }
    
    .rec-select-btn:hover {
        background: linear-gradient(135deg, #004d00, #003300) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 100, 0, 0.4) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------------- Load Data -------------
@st.cache_data
def load_data():
    return pd.read_csv("udemy_courses.csv")

df = load_data()

# ------------- Load Models -------------
@st.cache_resource
def load_models():
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    scaler = joblib.load('scaler.pkl')
    return tfidf, scaler

tfidf, scaler = load_models()

# ------------- Prepare Weighted Feature Matrix -------------
def prepare_weighted_features(df, tfidf, scaler):
    df_encoded = pd.get_dummies(df, columns=["subject", "level"])
    num_features = ['price', 'num_subscribers', 'num_reviews', 'num_lectures', 'content_duration']
    subject_cols = [col for col in df_encoded.columns if col.startswith("subject_")]
    level_cols = [col for col in df_encoded.columns if col.startswith("level_")]
    X_numeric = scaler.transform(df_encoded[num_features])
    X_subject = df_encoded[subject_cols].values
    X_level = df_encoded[level_cols].values
    X_title = tfidf.transform(df['course_title']).toarray()
    W_num = np.array([1.5, 1.0, 1.0, 1.0, 1.3])
    W_subject = 3.0
    W_level = 2.0
    W_title = 1.5
    X_weighted = np.hstack([
        X_numeric * W_num,
        X_subject * W_subject,
        X_level * W_level,
        X_title * W_title
    ])
    return X_weighted

features_matrix = prepare_weighted_features(df, tfidf, scaler)

# ------------- Session State -------------
if 'selected_course' not in st.session_state:
    st.session_state.selected_course = None

# Initialize click tracking
if 'card_clicked' not in st.session_state:
    st.session_state.card_clicked = {}

# ------------- Sidebar Filters -------------
with st.sidebar:
    st.markdown("## 🎛️ Customize Your Preferences")
    subject_filter = st.selectbox("🎓 Choose Subject:", sorted(df["subject"].unique()))
    level_filter = st.selectbox("📊 Choose Level:", ["All"] + sorted(df["level"].dropna().unique().tolist()))
    search_title = st.text_input("🔍 Search by Title:")

# ------------- Apply Filters -------------
filtered_df = df[df["subject"] == subject_filter]
if level_filter != "All":
    filtered_df = filtered_df[filtered_df["level"] == level_filter]
if search_title:
    filtered_df = filtered_df[filtered_df['course_title'].str.contains(search_title, case=False, na=False)]
filtered_df = filtered_df.reset_index(drop=True)

# ------------- Pagination -------------
page_size = 9
total_pages = max(1, (len(filtered_df) - 1) // page_size + 1)
page = st.number_input("📄 Choose Page:", 1, total_pages, 1)
start_idx = (page - 1) * page_size
end_idx = start_idx + page_size
paged_df = filtered_df.iloc[start_idx:end_idx]

# ------------- Display Courses with Clickable Cards -------------
num_cols = 3
rows = len(paged_df) // num_cols + 1

for i in range(rows):
    cols = st.columns(num_cols)
    for j in range(num_cols):
        idx = i * num_cols + j
        if idx >= len(paged_df):
            break
        course = paged_df.iloc[idx]
        card_id = f"card_{start_idx + idx}"

        with cols[j]:
            # Use st.container with click handling
            container = st.container()
            
            with container:
                # Create the card HTML
                card_html = f"""
                <div class="clickable-card">
                    <h4 style="color:#2E8B57; margin-bottom: 10px;">{course['course_title'][:50]}</h4>
                    <p style="margin: 5px 0;">📚 <b>{course['subject']}</b></p>
                    <p style="margin: 5px 0;">⏱ <b>Duration:</b> {course['content_duration']} hours</p>
                    <p style="margin: 5px 0;">👥 <b>Subscribers:</b> {course['num_subscribers']:,}</p>
                    <p style="margin: 5px 0;">⭐ <b>Reviews:</b> {course['num_reviews']:,}</p>
                    <p style="margin: 5px 0;">💰 <b>Price:</b> {"Free" if not course['is_paid'] else course['price']}</p>
                    <p style="margin: 5px 0;">📈 <b>Level:</b> {course['level']}</p>
                    <p style="margin: 5px 0;"><a href="{course['url']}" target="_blank">🔗 Course Link</a></p>
                </div>
                """
                
                # Display the card
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Add CSS class to the button
                st.markdown(f"""
                <style>
                div[data-testid="stButton"] > button[key="{card_id}"] {{
                    background: linear-gradient(135deg, #4CAF50, #45a049) !important;
                    border: none !important;
                    border-radius: 8px !important;
                    color: white !important;
                    padding: 8px 16px !important;
                    font-size: 14px !important;
                    font-weight: 600 !important;
                    box-shadow: 0 2px 4px rgba(76, 175, 80, 0.3) !important;
                    transition: all 0.3s ease !important;
                    width: 100% !important;
                    margin-top: 10px !important;
                }}
                
                div[data-testid="stButton"] > button[key="{card_id}"]:hover {{
                    background: linear-gradient(135deg, #45a049, #3d8b40) !important;
                    transform: translateY(-2px) !important;
                    box-shadow: 0 4px 8px rgba(76, 175, 80, 0.4) !important;
                }}
                </style>
                """, unsafe_allow_html=True)
                
                # Select button
                if st.button("📌 Select Course", key=card_id):
                    st.session_state.selected_course = course['course_title']
                    st.rerun()

# ---------- Course Details and Recommendations ----------
if st.session_state.selected_course:
    selected_course = df[df['course_title'] == st.session_state.selected_course]
    if selected_course.empty:
        st.error("The selected course is not found.")
    else:
        course = selected_course.iloc[0]
        
        st.markdown("## 🎯 Choosed Course Details")
        st.markdown(f"### 📘 {course['course_title']} ({course['subject']})")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"📈 **Level**: {course['level']}  |  ⏱ **Duration**: {course['content_duration']} hours  |  👥 **Subscribers**: {course['num_subscribers']:,}  |  ⭐ **Reviews**: {course['num_reviews']:,}")
            st.write(f"💰 **Price**: {'Free' if not course['is_paid'] else course['price']}")
        
        with col2:
            if st.button("❌ Clear Selection"):
                st.session_state.selected_course = None
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 🔍 Recommended Courses")

        # Calculate recommendations
        selected_index = df.index[df['course_title'] == st.session_state.selected_course][0]
        sims = cosine_similarity([features_matrix[selected_index]], features_matrix)[0]
        sims[selected_index] = -1
        top_indices = sims.argsort()[::-1][:3]
        recs = df.iloc[top_indices].copy()
        recs['similarity'] = sims[top_indices]

        # Display recommendation cards
        num_cols = 3
        rows = len(recs) // num_cols + 1

        for i in range(rows):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                idx = i * num_cols + j
                if idx >= len(recs):
                    break
                rec_course = recs.iloc[idx]
                rec_card_id = f"rec_card_{idx}"

                with cols[j]:
                    container = st.container()
                    
                    with container:
                        # Recommendation card HTML
                        card_html = f"""
                        <div class="recommendation-card">
                            <h4 style="color:#006400; margin-bottom: 10px;">{rec_course['course_title'][:50]}</h4>
                            <p style="margin: 5px 0;">📏 <b>Similarity:</b> {rec_course['similarity']:.2f}</p>
                            <p style="margin: 5px 0;">📚 <b>{rec_course['subject']}</b></p>
                            <p style="margin: 5px 0;">📈 <b>Level:</b> {rec_course['level']}</p>
                            <p style="margin: 5px 0;">💰 <b>Price:</b> {"Free" if not rec_course['is_paid'] else rec_course['price']}</p>
                            <p style="margin: 5px 0;">⏱ <b>Duration:</b> {rec_course['content_duration']:.2f} hours</p>
                            <p style="margin: 5px 0;">👥 <b>Subscribers:</b> {rec_course['num_subscribers']:,}</p>
                            <p style="margin: 5px 0;">⭐ <b>Reviews:</b> {rec_course['num_reviews']:,}</p>
                            <p style="margin: 5px 0;"><a href="{rec_course['url']}" target="_blank">🔗 Course Link</a></p>
                        </div>
                        """
                        
                        st.markdown(card_html, unsafe_allow_html=True)
                        
                        # Style the recommendation button
                        st.markdown(f"""
                        <style>
                        div[data-testid="stButton"] > button[key="{rec_card_id}"] {{
                            background: linear-gradient(135deg, #006400, #004d00) !important;
                            border: none !important;
                            border-radius: 8px !important;
                            color: white !important;
                            padding: 8px 16px !important;
                            font-size: 14px !important;
                            font-weight: 600 !important;
                            box-shadow: 0 2px 4px rgba(0, 100, 0, 0.3) !important;
                            transition: all 0.3s ease !important;
                            width: 100% !important;
                            margin-top: 10px !important;
                        }}
                        
                        div[data-testid="stButton"] > button[key="{rec_card_id}"]:hover {{
                            background: linear-gradient(135deg, #004d00, #003300) !important;
                            transform: translateY(-2px) !important;
                            box-shadow: 0 4px 8px rgba(0, 100, 0, 0.4) !important;
                        }}
                        </style>
                        """, unsafe_allow_html=True)
                        
                        if st.button("📌 Select Recommendation", key=rec_card_id):
                            st.session_state.selected_course = rec_course['course_title']
                            st.rerun()