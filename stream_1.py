import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from streamlit.components.v1 import html


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spam Detection App",
    layout="wide",
    page_icon="🔍"
)

# Load external CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
model = joblib.load("spam_clf_2.pkl")


# --------------- SIDEBAR ------------------

sbar = st.sidebar

# ---- TOP IMAGE ----
try:
    sbar.image("images/logo.png", width=250)
except:
    sbar.warning("Logo image not found. Please place logo.png in images/ folder.")

# ---- TITLE ----
sbar.markdown(
    "<div class='sidebar-title' style='font-size:26px; font-weight:700; margin-top:10px;'>📘 About</div>",
    unsafe_allow_html=True
)

sbar.info(
    "**📨 Spam Detection App**\n\n"
    "• Quickly analyze any single message for spam probability\n\n"
    "• Upload text or CSV files for bulk spam classification\n\n"
    "• Fast and scalable ML pipeline using HashingVectorizer + SGDClassifier\n\n"
    "• Designed for high-speed processing and accurate prediction results\n\n"
)


# ---- FEATURES ----
sbar.markdown("### ⭐ Features")
sbar.markdown(
    """
    - Real-time message classification  
    - Bulk CSV/TXT upload  
    - Fast & accurate predictions  
    - Professional UI with custom styling  
    """
)

# ---- CONTACT SECTION ----
sbar.markdown("### 📞 Contact")
sbar.markdown(
    """
    **Developer:** MEHUL RASTOGI  
    **Email:** mehulrastogi@gmail.com 
    **Number:** 9876543210      

    **GitHub:** github.com/username  
    """
)

# ---- FOOTER ----
sbar.markdown(
    """
    <hr>
    <div style='font-size:13px; text-align:center; opacity:0.7;'>
        <p style='margin:0;'>Made with ❤️ using Streamlit </p>
     <p style='margin:0; font-size:12px; color:#7A8A9A;'>© 2025 All Rights Reserved</p>
 </div>
    """,
    unsafe_allow_html=True
)



# ---------------- MAIN TITLE + ADMIN BUTTON ----------------
col1, col2 = st.columns([4,1])

with col1:
    st.markdown("<div class='main-title'>AI-Powered Spam Detection Engine</div>", unsafe_allow_html=True)

with col2:
    if st.button("Admin Login"):
        st.session_state['show_admin'] = True

# ---------------- ADMIN LOGIN PAGE ----------------
if st.session_state.get('show_admin', False):

    st.markdown("### 🔒 Admin Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login", key="login_admin"):
        # Replace below with your secure credentials
        if username == "admin" and password == "1234":
            st.session_state['admin_logged_in'] = True
            st.success("✅ Logged in successfully!")
        else:
            st.error("❌ Invalid credentials")

# ---------------- ADMIN MODEL UPDATE ----------------
if st.session_state.get('admin_logged_in', False):
    st.markdown("### ⚡ Update Spam Detection Model")

    file = st.file_uploader("Upload new training data (.csv)", type=["csv"])

    if file:
        df_new = pd.read_csv(file)
        
        
        X_new = df_new.iloc[:,0]
        y_new = df_new.iloc[:,1]

        if st.button("Update Model", key="update_model"):
            # Update the existing model with new data
            model['sgd'].partial_fit(model['hv'].transform(X_new), y_new,classes=['ham','spam'])

            
            # Save updated model
            joblib.dump(model, "spam_clf_2.pkl")
            st.success("✅ Model updated successfully!")

            st.balloons()  

st.markdown("<div class='sub'>Quick, reliable, and smart spam classification for your messages</div>", unsafe_allow_html=True)

# ---------------- COLUMNS ----------------
col1, col2 = st.columns([1, 2], gap='large')



# ============= LEFT COLUMN (SINGLE MESSAGE) =============
with col1:

    st.markdown("<div class='left-block'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Single Message Check</div>", unsafe_allow_html=True)

    text = st.text_input("Enter a message")

    # Initialize session states
    if "single_result" not in st.session_state:
        st.session_state.single_result = None
    if "single_proba" not in st.session_state:
        st.session_state.single_proba = None

    # Predict button
    if st.button("Predict"):
        result = model.predict([text])[0]
        proba = model.predict_proba([text])[0]

        st.session_state.single_result = result
        st.session_state.single_proba = proba

        # Display main result
        if result == "spam":
            st.error("🛑 This message is **SPAM**")
        else:
            st.success("✅ This message is **HAM**")

    # -------------------------------
    # MORE INFO
    # -------------------------------
    if st.session_state.single_result is not None:

        if st.button("More Info"):

            spam_prob = st.session_state.single_proba[1] * 100
            ham_prob = st.session_state.single_proba[0] * 100

            if st.session_state.single_result == "spam":
                prob = spam_prob
                label = "SPAM"
            else:
                prob = ham_prob
                label = "HAM"

            intensity = "Might be" if prob <= 75 else "Sure"

            # ---- PIE CHART ----
            chart_df = pd.DataFrame({
                "Type": ["Ham", "Spam"],
                "Confidence": [ham_prob, spam_prob]
            })

            fig = px.pie(
                chart_df,
                names="Type",
                values="Confidence",
                hole=0.45,
                color="Type",
                color_discrete_map={"Ham": "#2DBF73", "Spam": "#FF4B4B"}
            )

            fig.update_layout(
                showlegend=False,
                height=240,
                margin=dict(l=0, r=0, t=10, b=10)
            )

            chart_html = fig.to_html(include_plotlyjs="cdn")

            # ---- FULL CUSTOM BLOCK (NO STREAMLIT SPACING) ----
            html_block = f"""
            <div style="
                background: linear-gradient(135deg, #F1F5FF, #FFFFFF);
                padding: 22px 26px;
                border-radius: 16px;
                margin-top: 2px;
                border: 1px solid rgba(120,150,255,0.25);
                box-shadow: 0 8px 20px rgba(0,0,0,0.08);
            ">
                <div style="display:flex; gap:25px; align-items:center;">

                    <!-- LEFT INFO -->
                    <div style="
                        flex:1;
                        background:#F0F4FF;
                        padding:18px;
                        border-radius:12px;
                        border-left:6px solid #557CFF;
                        font-size:17px;
                        color:#3d8903;
                        line-height:1.6;
                        font-weight:500;
                        box-shadow:0 2px 7px rgba(0,0,0,0.07);
                    ">
                        📊 <b>Confidence:</b> {prob:.2f}% <br>
                        <b>{intensity} it's {label}</b>
                    </div>

                    <!-- PIE CHART -->
                    <div style="flex:1.2;">
                        {chart_html}
                    </div>

                </div>
            </div>
            """

            html(html_block, height=420)

    st.markdown("</div>", unsafe_allow_html=True)

# ============= RIGHT COLUMN (BULK CHECK) =============
with col2:

    st.markdown("<div class='section-title'>Bulk Message Check</div>", unsafe_allow_html=True)

    # --- File Upload ---
    file = st.file_uploader("Upload your file (.txt or .csv)", type=["txt", "csv"])

    if file:
        if 'bulk_df' not in st.session_state or st.session_state.get('file_changed', True):
            # Read uploaded file once
            st.session_state.bulk_df = pd.read_csv(file, header=None, names=["mgs"], sep='$')
            st.session_state.predicted = False
            st.session_state.show_spam = False
            st.session_state.show_ham = False
            st.session_state.search_keyword = ''
            st.session_state.file_changed = False
        place = st.empty()
        place.dataframe(st.session_state.bulk_df, use_container_width=True)

    # --- Buttons Row ---
    r_col1, r_col2 = st.columns([1,1])
    
    with r_col1:
        if st.button("Predict Bulk", key="b2") and file:
            # Predict and save in session state
            st.session_state.bulk_df['result'] = model.predict(st.session_state.bulk_df["mgs"])
            st.session_state.predicted = True

    # --- Only show summary/filtering if prediction done ---
    if file and st.session_state.predicted:

        # --- Checkboxes Row ---
        c1, c2 = st.columns([1,1])
        with c1:
            st.session_state.show_spam = st.checkbox("🛑 Spam", value=st.session_state.show_spam)
        with c2:
            st.session_state.show_ham = st.checkbox("✅ Ham", value=st.session_state.show_ham)

        # --- Keyword Search with Clear Button ---
        s_col1, s_col2 = st.columns([4,1])
        with s_col1:
            st.session_state.search_keyword = st.text_input(
                "🔍 Search Keyword", 
                value=st.session_state.search_keyword
            )
        with s_col2:
            st.text('')    #-------> this is just for gap for push down clear button
            if st.button("❌ Clear", key="clear_search"):
                st.session_state.search_keyword = ''
                

        keyword = st.session_state.search_keyword

        # --- Counts ---
        spam_count = (st.session_state.bulk_df['result'] == 'spam').sum()
        ham_count = (st.session_state.bulk_df['result'] == 'ham').sum()
        total = spam_count + ham_count

        spam_perc = spam_count * 100 / total
        ham_perc = ham_count * 100 / total

        # --- Summary Box ---
        summary_text = (
            f"📌 Spam Count: {spam_count}\n\n"
            f"📌 Ham Count: {ham_count}\n\n"
            f"📌 Total Messages: {total}\n\n"
            f"📌 Spam %: {spam_perc:.2f}%\n\n"
            f"📌 Ham %: {ham_perc:.2f}%"
        )
        st.info(summary_text)

        # --- Filtered DataFrame based on checkboxes + keyword ---
        filtered_df = st.session_state.bulk_df.copy()
        if st.session_state.show_spam and not st.session_state.show_ham:
            filtered_df = filtered_df[filtered_df["result"] == "spam"]
        elif st.session_state.show_ham and not st.session_state.show_spam:
            filtered_df = filtered_df[filtered_df["result"] == "ham"]

        if keyword.strip() != "":
            filtered_df = filtered_df[filtered_df['mgs'].str.contains(keyword, case=False, na=False)]

        # --- Highlight function ---
        def highlight_spam_ham(row):
            if row['result'] == 'spam':
                return ['background-color: #FFCCCC'] * len(row)  # light red
            elif row['result'] == 'ham':
                return ['background-color: #CCFFCC'] * len(row)  # light green
            else:
                return [''] * len(row)

        # --- Display filtered DataFrame with styling ---
        place.dataframe(filtered_df.style.apply(highlight_spam_ham, axis=1), use_container_width=True)

        # --- CSV Export ---
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        with r_col2:
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_data,
                file_name="spam_ham_results.csv",
                mime="text/csv"
            )
