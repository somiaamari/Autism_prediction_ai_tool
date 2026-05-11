
import streamlit as st
import pandas as pd
import time

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Screening Questionnaire",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for navigation if not already done
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Questionnaire"

# ===== HERO SECTION =====
st.html("""
<div style="
    text-align: center; 
    padding: 50px 20px; 
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    border-radius: 20px; 
    margin-bottom: 40px;
    box-shadow: 0 10px 30px rgba(76, 175, 80, 0.1);
    border: 2px solid rgba(76, 175, 80, 0.2);
    position: relative;
    overflow: hidden;
">
    <div style="
        position: absolute;
        top: 10px;
        right: 20px;
        font-size: 40px;
        opacity: 0.1;
    ">❓❓❓</div>
    
    <h1 style="
        font-size: 42px; 
        color: #1b5e20; 
        margin-bottom: 15px;
        font-weight: 700;
    ">
        📋 Screening Questionnaire
    </h1>
    
    <p style="
        font-size: 20px; 
        color: #388e3c; 
        max-width: 800px; 
        margin: 0 auto 20px;
        line-height: 1.5;
    ">
        Answer these 10 simple questions about your child's behavior
    </p>
    
    <div style="
        margin-top: 20px;
        padding: 12px 25px;
        background: rgba(255, 255, 255, 0.7);
        border-radius: 50px;
        display: inline-block;
        font-weight: 600;
        color: #2e7d32;
    ">
        ⏱️ Takes only 3-5 minutes
    </div>
</div>
""")

# ===== INSTRUCTIONS =====
with st.container():
    st.html("""
    <div style="
        background: linear-gradient(to right, #f1f8e9, #e8f5e9);
        padding: 25px 30px;
        border-radius: 15px;
        margin-bottom: 40px;
        border-left: 6px solid #4caf50;
    ">
        <h2 style="color: #1b5e20; margin-top: 0; margin-bottom: 15px;">📝 How to Answer</h2>
        <div style="display: flex; align-items: center; gap: 15px; color: #555;">
            <div style="font-size: 24px;">1️⃣</div>
            <div style="flex: 1;">Read each question carefully about your child's behavior</div>
        </div>
        <div style="display: flex; align-items: center; gap: 15px; color: #555; margin-top: 15px;">
            <div style="font-size: 24px;">2️⃣</div>
            <div style="flex: 1;">Select <strong>"Yes"</strong> if the behavior is <strong>typically present</strong></div>
        </div>
        <div style="display: flex; align-items: center; gap: 15px; color: #555; margin-top: 15px;">
            <div style="font-size: 24px;">3️⃣</div>
            <div style="flex: 1;">Select <strong>"No"</strong> if the behavior is <strong>rarely or never seen</strong></div>
        </div>
    </div>
    """)

# ===== DEMOGRAPHIC INFORMATION =====
st.html("""
<div style="
    background: linear-gradient(to right, #f1f8e9, #e8f5e9);
    padding: 25px 30px;
    border-radius: 15px;
    margin: 40px 0 20px 0;
    border-left: 6px solid #4caf50;
">
    <h2 style="color: #0d47a1; margin-top: 0;">👶 Demographic Information</h2>
    <p style="color: #1565c0; margin-bottom: 0;">
        Basic information about your child
    </p>
</div>
""")

with st.container():
    # st.html("""
    # <div style="
    #     background: white;
    #     border-radius: 15px;
    #     padding: 30px;
    #     box-shadow: 0 4px 15px rgba(33, 150, 243, 0.08);
    #     border: 2px solid #4caf50;
    #     margin-bottom: 40px;
    # ">
    # """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Custom CSS for the number input
        st.html("""
    <style>
    div[data-baseweb="input"] input {
        background-color: #f1f8e9 !important;
        border-color: #4caf50 !important;
        border-width: 2px !important;
        border-radius: 7px 0 0 0 !important; 
        padding: 8px 12px !important;
    }
    div[data-baseweb="input"] input:focus {
        border-color: #4caf50 !important;
        box-shadow: 0 0 0 3px rgba(56, 142, 60, 0.2) !important;
        background-color: #e8f5e9 !important;
    }
    </style>
    """)
        age = st.number_input(
            "Child's Age",
            # min_value=12,
            # max_value=60,
            value=1,
            step=1,
            help="Enter child's age"
        )
    
    with col2:
        gender = st.radio(
            "Child's Gender",
            ["Male", "Female"],
            horizontal=True
        )
    
    st.html("</div>")

# ===== MEDICAL HISTORY =====
st.html("""
<div style="
    background: linear-gradient(to right, #f1f8e9, #e8f5e9);
    padding: 25px 30px;
    border-radius: 15px;
    margin: 40px 0 20px 0;
    border-left: 6px solid #4caf50;
">
    <h2 style="color: #880e4f; margin-top: 0;">🏥 Medical History</h2>
    <p style="color: #c2185b; margin-bottom: 0;">
        Medical and family background information
    </p>
</div>
""")

with st.container():
    # st.html("""
    # <div style="
    #     background: white;
    #     border-radius: 15px;
    #     padding: 30px;
    #     box-shadow: 0 4px 15px rgba(233, 30, 99, 0.08);
    #     border: 2px solid #fce4ec;
    #     margin-bottom: 40px;
    # ">
    # """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        jaundice = st.radio(
            "Was your child born with jaundice?",
            ["Yes", "No"],
            horizontal=True,
            key="jaundice",
            help="Jaundice at birth"
        )
    
    with col2:
        family_asd = st.radio(
            "Family history of ASD?",
            ["Yes", "No"],
            horizontal=True,
            key="family_asd",
            help="Does anyone in the immediate family have an Autism Spectrum Disorder diagnosis?"
        )
    
    st.html("</div>")

# ===== DEFINE QUESTIONS BY CATEGORY =====
question_categories = [
    {
        "category": "Social Interaction",
        "icon": "👥",
        "color": "#2e7d32",
        "bg_color": "#e8f5e9",
        "border_color": "#4caf50",
        "questions": [
            {
                "id": "A1_Score",
                "question": "Does your child look at you when you call his or her name?",
                "options": ["Yes", "No"],
                "description": "Responds to name being called"
            },
            {
                "id": "A2_Score",
                "question": "How easy is it for you to get eye contact with your child?",
                "options": ["Easy", "Difficult"],
                "description": "Makes appropriate eye contact"
            },
            {
                "id": "A4_Score",
                "question": "Does your child point to something to share interest with you?",
                "options": ["Yes", "No"],
                "description": "Shares enjoyment with others"
            },
            {
                "id": "A3_Score",
                "question": "Does your child point to something to indicate that he or she wants something?",
                "options": ["Yes", "No"],
                "description": "Responds to others' smiles"
            }
        ]
    },
    {
        "category": "Communication",
        "icon": "💬",
        "color": "#2e7d32",
        "bg_color": "#e8f5e9",
        "border_color": "#4caf50",
        "questions": [
            {
                "id": "A9_Score",
                "question": "Does your child use simple gestures? (example: waving for goodbye)",
                "options": ["Yes", "No"],
                "description": "Uses gestures (pointing, waving)"
            },
            {
                "id": "A8_Score",
                "question": "How would you describe your child's first word?",
                "options": ["Typical", "Atypical"],
                "description": "Uses words meaningfully"
            },
            {
                "id": "A5_Score",
                "question": "Does your child pretend? (e.g., talk on the toy phone)",
                "options": ["Yes", "No"],
                "description": "Engages in pretend play"
            }
        ]
    },
    {
        "category": "Behavior Patterns",
        "icon": "🔄",
        "color": "#2e7d32",
        "bg_color": "#e8f5e9",
        "border_color": "#4caf50",
        "questions": [
            {
                "id": "A6_Score",
                "question": "Does your child follow where you are looking at?",
                "options": ["Yes", "No"],
                "description": "Follows visual attention"
            },
            {
                "id": "A7_Score",
                "question": "If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them?",
                "options": ["Yes", "No"],
                "description": "Shows empathy and comfort"
            },
            {
                "id": "A10_Score",
                "question": "Does your child stare at nothing with no apparent purpose?",
                "options": ["Yes", "No"],
                "description": "Intense, focused interests"
            }
        ]
    }
]

# Track responses
responses = {}

# ===== DISPLAY QUESTIONS BY CATEGORY =====
question_number = 1

for category_data in question_categories:
    # Category Header
    st.html(f"""
    <div style="
        background: linear-gradient(to right, {category_data['bg_color']}, {category_data['bg_color']});
        padding: 25px 30px;
        border-radius: 15px;
        margin: 40px 0 20px 0;
        border-left: 6px solid {category_data['color']};
    ">
        <h2 style="color: {category_data['color']}; margin-top: 0;">
            {category_data['icon']} {category_data['category']}
        </h2>
        <p style="color: #555; margin-bottom: 0; font-size: 14px;">
            Questions about {category_data['category'].lower()}
        </p>
    </div>
    """)
    
    # Questions in this category
    for q in category_data['questions']:
        with st.container():
            st.html(f"""
            <div style="
                background: white;
                padding: 25px;
                border-radius: 15px;
                margin-bottom: 20px;
                border: 2px solid {category_data['border_color']};
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            ">
                <div style="
                    display: flex;
                    align-items: center;
                    margin-bottom: 15px;
                ">
                    <div style="
                        background: {category_data['color']};
                        color: white;
                        width: 36px;
                        height: 36px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: bold;
                        margin-right: 12px;
                        font-size: 16px;
                    ">{question_number}</div>
                    <div>
                        <h3 style="color: {category_data['color']}; margin: 0; font-size: 18px;">Question {question_number}</h3>
                        <p style="color: #666; margin: 3px 0 0 0; font-size: 13px; font-style: italic;">{q['description']}</p>
                    </div>
                </div>
                <p style="color: #2e7d32; font-size: 16px; margin-bottom: 20px; font-weight: 500;">
                    {q['question']}
                </p>
            </div>
            """)
            
            # Radio buttons for each question
            response = st.radio(
                f"Select answer for Question {question_number}",
                q['options'],
                key=f"q_{q['id']}",
                horizontal=True,
                label_visibility="collapsed",
                index=0 
            )
            
# ===== CORRECTED MAPPING - Based on model test results =====
# 1 = concerning behavior, 0 = typical behavior

            if q['id'] in ["A1_Score", "A3_Score", "A4_Score", "A5_Score", "A6_Score", "A7_Score", "A9_Score"]:
    # For these: "No" indicates concerning behavior in the question
    # So if user says "No" (concerning), we store as 1
    # If user says "Yes" (typical), we store as 0
              responses[q['id']] = 1 if response == "No" else 0  # REVERSED!
              print(f"DEBUG: {q['id']}: '{response}' → {responses[q['id']]} (No=1=concerning, Yes=0=typical)")
    
            elif q['id'] == "A2_Score":
    # "Difficult" eye contact = concerning = 1
    # "Easy" eye contact = typical = 0
              responses[q['id']] = 1 if response == "Difficult" else 0  # CORRECT (no change)
              print(f"DEBUG: {q['id']}: '{response}' → {responses[q['id']]} (Difficult=1=concerning, Easy=0=typical)")
    
            elif q['id'] == "A8_Score":
    # "Atypical" first words = concerning = 1
    # "Typical" first words = typical = 0
              responses[q['id']] = 1 if response == "Atypical" else 0  # CORRECT (no change)
              print(f"DEBUG: {q['id']}: '{response}' → {responses[q['id']]} (Atypical=1=concerning, Typical=0=typical)")
    
            elif q['id'] == "A10_Score":
    # "Yes" staring = concerning = 1
    # "No" staring = typical = 0
              responses[q['id']] = 1 if response == "Yes" else 0  # CORRECT (no change)
              print(f"DEBUG: {q['id']}: '{response}' → {responses[q['id']]} (Yes=1=concerning, No=0=typical)")
    
            else:
              responses[q['id']] = 0
              print(f"DEBUG: {q['id']}: No mapping, set to 0")
        
        question_number += 1

# ===== ADDITIONAL INFORMATION =====
st.html("""
<div style="
    background: linear-gradient(to right, #f1f8e9, #e8f5e9);
    padding: 25px 30px;
    border-radius: 15px;
    margin: 40px 0 20px 0;
    border-left: 6px solid #4caf50;
">
    <h2 style="color: #f57f17; margin-top: 0;">📝 Additional Information</h2>
</div>
""")

with st.container():
    # st.html("""
    # <div style="
    #     background: white;
    #     border-radius: 15px;
    #     padding: 30px;
    #     box-shadow: 0 4px 15px rgba(251, 192, 45, 0.08);
    #     border: 2px solid #fff9c4;
    #     margin-bottom: 40px;
    # ">
    # """)
    
    used_app_before = st.radio(
        "Have you used a screening app before?",
        ["Yes", "No"],
        horizontal=True,
        key="used_app_before",
        help="This helps us understand your experience level"
    )
    
    st.html("</div>")

# ===== SUBMIT BUTTON SECTION =====
st.html("""
<div style="
    text-align: center;
    padding: 40px 20px;
    background: linear-gradient(135deg, #f1f8e9, #e8f5e9);
    border-radius: 20px;
    margin: 40px 0;
    border: 2px dashed #81c784;
">
    <h3 style="color: #1b5e20; margin-bottom: 20px;">Ready to Submit?</h3>
    <p style="color: #555; max-width: 600px; margin: 0 auto 30px;">
        Once you submit, we'll analyze the responses and show you personalized results.
    </p>
</div>
""")

# Submit button - CENTERED
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    submit_button = st.button(
        "🚀 Submit Screening",
        type="primary",
        use_container_width=True,
        key="submit_screening_btn",
        help="Click to analyze your responses"
    )

# ===== HANDLE SUBMISSION =====
if submit_button:
    # Prepare complete data dictionary
    screening_data = {
        **responses,
        "age": age,
        "gender": gender,
        "jundice": jaundice,
        "family_asd": family_asd
        # "used_app_before": used_app_before
    }
    
    # Store in session state
    st.session_state.screening_data = screening_data
    st.session_state.screening_submitted = True
    
    # Show success message
    st.success("✅ All questions answered! Processing your responses...")
    
    # Add a progress bar
    with st.spinner("Analyzing responses..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
    
    # Success message with animation
    st.html("""
    <div style="
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border-radius: 15px;
        margin: 20px 0;
        border: 2px solid #4caf50;
    ">
        <div style="font-size: 48px; margin-bottom: 15px;">🎉</div>
        <h3 style="color: #1b5e20; margin-bottom: 10px;">Screening Complete!</h3>
        <p style="color: #2e7d32;">
            Your responses have been successfully recorded.
        </p>
    </div>
    """)

# Show "View Results" button if screening was submitted
if st.session_state.get('screening_submitted', False):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📊 View Results", key="view_results_btn", type="primary", use_container_width=True):
            st.session_state.current_page = "Results"
            st.rerun()


# ===== CUSTOM CSS - FIXED FOR GREEN BUTTONS =====
st.html("""
<style>
/* ===== FIX FOR GREEN BUTTONS IN MAIN CONTENT ===== */
div[data-testid="stAppViewContainer"] button[kind="primary"],
div[data-testid="column"] button[kind="primary"],
.main button[kind="primary"] {
    background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(46, 125, 50, 0.2) !important;
}

div[data-testid="stAppViewContainer"] button[kind="primary"]:hover,
div[data-testid="column"] button[kind="primary"]:hover,
.main button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(46, 125, 50, 0.3) !important;
    filter: brightness(1.1) !important;
}

/* Style for the radio buttons */
.stRadio > div {
    flex-direction: row !important;
    gap: 20px !important;
}

.stRadio > div[role="radiogroup"] > label {
    background: white !important;
    border: 2px solid #c8e6c9 !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    margin-right: 10px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.stRadio > div[role="radiogroup"] > label:hover {
    border-color: #81c784 !important;
    transform: translateY(-1px) !important;
}

.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {
    background: #e8f5e9 !important;
    border-color: #4caf50 !important;
    color: #1b5e20 !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(76, 175, 80, 0.2) !important;
}
</style>
""")

# ===== FOOTER =====
st.html("""
<div style="
    text-align: center;
    padding: 30px 20px;
    margin-top: 50px;
    background: linear-gradient(135deg, #f1f8e9, #e8f5e9);
    border-radius: 15px;
    border: 2px dashed #81c784;
">
    <div style="display: flex; justify-content: center; gap: 30px; margin-bottom: 15px;">
        <div style="text-align: center;">
            <div style="font-size: 24px;">🔒</div>
            <div style="color: #2e7d32; font-size: 14px; font-weight: 600;">Private</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 24px;">💡</div>
            <div style="color: #2e7d32; font-size: 14px; font-weight: 600;">Evidence-Based</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 24px;">🤗</div>
            <div style="color: #2e7d32; font-size: 14px; font-weight: 600;">Supportive</div>
        </div>
    </div>
    
    <p style="color: #4caf50; margin: 0; font-size: 14px; font-weight: 600;">
        🧩 Autism Screening Tool • Your Privacy is Protected • Results are Temporary
    </p>
</div>
""")