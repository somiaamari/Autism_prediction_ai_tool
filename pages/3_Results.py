import streamlit as st
from core.predictor import predictor  # Import l'instance du prédicteur
from core.utils import generate_pdf_report, get_resources

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Screening Results",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS FOR LIGHT MODE =====
st.html("""
<style>
/* ===== BACKGROUND ===== */
div[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 50%, #c8e6c9 100%) !important;
}

/* ===== MAIN CONTAINER ===== */
section[data-testid="stVerticalBlock"] > div:first-child {
    background: white !important;
    border-radius: 24px !important;
    padding: 50px 40px !important;
    box-shadow: 0 15px 50px rgba(46, 125, 50, 0.15) !important;
    border: 3px solid rgba(76, 175, 80, 0.2) !important;
    margin: 30px auto !important;
    max-width: 1200px !important;
    position: relative !important;
    overflow: hidden !important;
}

/* Decorative corner accent */
section[data-testid="stVerticalBlock"] > div:first-child::before {
    content: "📊" !important;
    position: absolute !important;
    top: 20px !important;
    right: 30px !important;
    font-size: 60px !important;
    opacity: 0.08 !important;
}

/* ===== TITLES ===== */
h1 {
    color: #1b5e20 !important;
    text-align: center !important;
    margin-bottom: 15px !important;
    font-weight: 800 !important;
    font-size: 42px !important;
    text-shadow: 0 2px 10px rgba(27, 94, 32, 0.1) !important;
}

h2 {
    color: #2e7d32 !important;
    margin: 40px 0 25px 0 !important;
    font-weight: 700 !important;
    font-size: 28px !important;
    position: relative !important;
    padding-bottom: 12px !important;
}

h2::after {
    content: "" !important;
    position: absolute !important;
    bottom: 0 !important;
    left: 0 !important;
    width: 80px !important;
    height: 4px !important;
    background: linear-gradient(90deg, #4caf50, #81c784) !important;
    border-radius: 2px !important;
}

h3 {
    color: #388e3c !important;
    margin: 30px 0 20px 0 !important;
    font-weight: 600 !important;
    font-size: 22px !important;
}

/* ===== BUTTONS ===== */
.stButton > button {
    border-radius: 14px !important;
    padding: 16px 36px !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    transition: all 0.3s ease !important;
    border: none !important;
    margin-top: 35px !important;
    box-shadow: 0 6px 20px rgba(76, 175, 80, 0.2) !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2e7d32 0%, #4caf50 50%, #66bb6a 100%) !important;
    color: white !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 30px rgba(76, 175, 80, 0.35) !important;
}

.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%) !important;
    color: #1b5e20 !important;
    border: 2px solid #81c784 !important;
}

/* ===== PROGRESS BAR ===== */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #66bb6a, #4caf50, #2e7d32) !important;
    border-radius: 10px !important;
    height: 12px !important;
    box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3) !important;
}

.stProgress > div > div {
    background: #e8f5e9 !important;
    border-radius: 10px !important;
    height: 12px !important;
}

/* ===== CARDS/CONTAINERS ===== */
div[data-testid="column"] > div {
    background: linear-gradient(135deg, #f8fdf8 0%, #e8f5e9 100%) !important;
    border-radius: 18px !important;
    padding: 30px 25px !important;
    border: 2px solid rgba(129, 199, 132, 0.3) !important;
    height: 100% !important;
    box-shadow: 0 8px 20px rgba(76, 175, 80, 0.08) !important;
    transition: all 0.3s ease !important;
}

div[data-testid="column"] > div:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 30px rgba(76, 175, 80, 0.15) !important;
    border-color: rgba(76, 175, 80, 0.5) !important;
}

/* ===== INFO BOXES ===== */
.stAlert {
    border-radius: 14px !important;
    border: 2px solid #e8f5e9 !important;
    background: linear-gradient(135deg, #f8fdf8 0%, #e8f5e9 100%) !important;
}

/* ===== LISTS ===== */
ul {
    color: #1b5e20 !important;
    font-size: 16px !important;
    line-height: 1.6 !important;
}

li {
    margin-bottom: 12px !important;
}

/* ===== EXPANDERS ===== */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #f8fdf8 0%, #e8f5e9 100%) !important;
    border: 2px solid #c8e6c9 !important;
    border-radius: 12px !important;
    color: #1b5e20 !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}

.streamlit-expanderContent {
    background: white !important;
    border: 2px solid #e8f5e9 !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 20px !important;
}

/* ===== DIVIDER ===== */
hr {
    border: none !important;
    height: 3px !important;
    background: linear-gradient(90deg, #e8f5e9, #c8e6c9, #e8f5e9) !important;
    margin: 40px 0 !important;
    border-radius: 2px !important;
}

/* ===== METRIC CARDS ===== */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f8fdf8 0%, #e8f5e9 100%) !important;
    border: 2px solid #c8e6c9 !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

[data-testid="stMetricLabel"] {
    color: #2e7d32 !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}

[data-testid="stMetricValue"] {
    color: #1b5e20 !important;
    font-weight: 700 !important;
    font-size: 28px !important;
}

/* ===== RESPONSIVE ===== */
@media (max-width: 768px) {
    section[data-testid="stVerticalBlock"] > div:first-child {
        padding: 30px 20px !important;
        margin: 15px !important;
    }
    
    h1 {
        font-size: 32px !important;
    }
    
    h2 {
        font-size: 24px !important;
    }
    
    div[data-testid="column"] {
        margin-bottom: 20px !important;
    }
}

/* ===== ANIMATION ===== */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

section[data-testid="stVerticalBlock"] > div:first-child {
    animation: fadeIn 0.6s ease-out !important;
}
</style>
""")

# ===== HERO SECTION =====
st.html("""
<div style="
    text-align: center; 
    padding: 30px 20px 20px 20px; 
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    border-radius: 18px; 
    margin-bottom: 40px;
    box-shadow: 0 8px 25px rgba(76, 175, 80, 0.15);
    border: 2px solid rgba(76, 175, 80, 0.2);
    position: relative;
">
    <div style="
        position: absolute;
        top: 15px;
        right: 25px;
        font-size: 35px;
        opacity: 0.12;
    ">📊✨</div>
    
    <h1 style="
        font-size: 42px; 
        color: #1b5e20; 
        margin-bottom: 10px;
        font-weight: 800;
    ">
        📊 Screening Results
    </h1>
    
    <p style="
        font-size: 18px; 
        color: #388e3c; 
        max-width: 700px; 
        margin: 0 auto;
        line-height: 1.5;
    ">
        Your child's screening assessment results and recommendations
    </p>
</div>
""")

# ===== MAIN CONTENT =====

# Check if we have data
if not st.session_state.get('screening_data'):
    st.warning("📋 No screening data found. Please complete the screening first to see results.")
    
    # Go to Screening button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Go to Screening", type="primary", use_container_width=True):
            st.switch_page("pages/2_Screening.py")
    
    st.html("---")
    st.info("💡 **Tip:** Navigate to the 'Screening' page from the sidebar to begin the assessment.")
    st.stop()

# Get prediction using the improved predictor
data = st.session_state.screening_data

# Use the new predict_for_streamlit method
try:
    # Check if the method exists, otherwise use the regular predict
    if hasattr(predictor, 'predict_for_streamlit'):
        probability, risk_level, details = predictor.predict_for_streamlit(data)
        
    else:
        # Fallback to regular predict
        probability, risk_level, details = predictor.predict(data)
            # Fix probability display

    display_probability = probability
    
    # Apply the rules:
    # 1. If probability is 0.0%, show as 10.0%
    if display_probability < 1:
          display_probability = 0.10  # 10.0%
    
    # 2. If probability is higher than 85.0%, cap at 85.0%
    if display_probability > 0.85:
          display_probability = 0.85  # 85.0%

except Exception as e:
    st.error(f"Error during prediction: {e}")
    # Simple fallback calculation
    a_scores = [data.get(f'A{i}_score', 0) for i in range(1, 11)]
    probability = sum(a_scores) / 10
    
    # Use consistent risk level logic
    if probability <= 0.3:
        risk_level = "🟢 Low Risk"
    elif probability <= 0.6:
        risk_level = "🟡 Moderate Risk"
    elif probability <= 0.85:
        risk_level = "🟠 High Risk"
    else:
        risk_level = "🔴 Very High Risk"
        
    details = {'fallback': True, 'concern_count': sum(a_scores)}

# Display results
st.header("Screening Outcome")

# Visual gauge section
# st.html("""
# <div style="
#     background: linear-gradient(135deg, #f8fdf8 0%, #e8f5e9 100%);
#     border-radius: 18px;
#     padding: 30px;
#     border: 2px solid rgba(129, 199, 132, 0.3);
#     margin-bottom: 40px;
# ">
# """)

col1, col2 = st.columns([1, 2])
with col1:   
    # Probability score with custom styling
    st.html(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <div style="font-size: 14px; color: #2e7d32; font-weight: 600; margin-bottom: 5px;">SCREENING SCORE</div>
        <div style="font-size: 48px; color: #1b5e20; font-weight: 800; margin-bottom: 10px;">{display_probability:.1%}</div>
    </div>
    """)
    st.progress(float(display_probability))
    
with col2:
    # Get color based on risk level (now consistent with predictor)
    if "🟢 Low Risk" in risk_level or "🟢 Faible" in risk_level:
        color = "#4CAF50"
        risk_description = "Low probability - typical development likely"
    elif "🟡 Moderate Risk" in risk_level or "🟡 Moyen" in risk_level:
        color = "#FF9800"
        risk_description = "Moderate probability - monitoring recommended"
    elif "🟠 High Risk" in risk_level:
        color = "#FF6B00"
        risk_description = "High probability - evaluation recommended"
    elif "🔴 Very High Risk" in risk_level or "🔴 Élevé" in risk_level:
        color = "#F44336"
        risk_description = "Very high probability - urgent evaluation needed"
    else:
        # Fallback for any other risk level
        color = "#FF9800"
        risk_description = risk_level
    
    # Display recommendation from details or use default
    recommendation = details.get('recommendation', 
        "Based on the screening results, professional evaluation may be beneficial.")
    
    st.html(f"""
    <div style="
        background: linear-gradient(135deg, {color}20 0%, {color}10 100%);
        border-radius: 14px;
        padding: 25px;
        border: 2px solid {color}40;
    ">
        <div style="font-size: 24px; color: {color}; font-weight: 700; margin-bottom: 10px;">
            {risk_level}
        </div>
        <div style="font-size: 16px; color: #2e7d32; line-height: 1.5; margin-bottom: 15px;">
            {risk_description}
        </div>
        <div style="font-size: 14px; color: #555; line-height: 1.4; font-style: italic;">
            {recommendation}
        </div>
    </div>
    """)

st.html("</div>")

# Model Confidence Info
with st.expander("📊 Model Confidence Information"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Accuracy", f"{predictor.accuracy:.1%}" if predictor.accuracy else "N/A")
    
    with col2:
        concern_count = details.get('concern_count', 0)
        st.metric("Signs Detected", f"{concern_count}/10")
    
    with col3:
        if 'alternative_score' in details:
            alt_score = details['alternative_score']
            st.metric("Simple Score", f"{alt_score:.1%}")

# Detailed breakdown
st.header("Response Summary")

col1, col2 = st.columns(2)

with col1:
    st.html("""
    <div style="
        background: linear-gradient(135deg, #f8fdf8 0%, #e8f5e9 100%);
        border-radius: 18px;
        padding: 30px;
        border: 2px solid rgba(129, 199, 132, 0.3);
        height: 100%;
    ">
        <h3 style="color: #1b5e20; margin-top: 0;">Areas of Concern</h3>
    """)
    
    # Map questions to their text and determine concerning answers
    questions_map = {
        'A1_Score': {
            'question': 'Does your child look at you when you call his or her name?',
            'concerning_answer': 'No',  # "No" is concerning for A1
            'typical_answer': 'Yes'
        },
        'A2_Score': {
            'question': 'How easy is it for you to get eye contact with your child?',
            'concerning_answer': 'Difficult',  # "Difficult" is concerning
            'typical_answer': 'Easy'
        },
        'A3_Score': {
            'question': 'Does your child point to something to indicate that he or she wants something?',
            'concerning_answer': 'No',  # "No" is concerning
            'typical_answer': 'Yes'
        },
        'A4_Score': {
            'question': 'Does your child point to something to share interest with you?',
            'concerning_answer': 'No',  # "No" is concerning
            'typical_answer': 'Yes'
        },
        'A5_Score': {
            'question': 'Does your child pretend? (e.g., talk on the toy phone)',
            'concerning_answer': 'No',  # "No" is concerning
            'typical_answer': 'Yes'
        },
        'A6_Score': {
            'question': 'Does your child follow where you are looking at?',
            'concerning_answer': 'No',  # "No" is concerning
            'typical_answer': 'Yes'
        },
        'A7_Score': {
            'question': 'If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them?',
            'concerning_answer': 'No',  # "No" is concerning
            'typical_answer': 'Yes'
        },
        'A8_Score': {
            'question': 'How would you describe your child\'s first word?',
            'concerning_answer': 'Atypical',  # "Atypical" is concerning
            'typical_answer': 'Typical'
        },
        'A9_Score': {
            'question': 'Does your child use simple gestures? (example: waving for goodbye)',
            'concerning_answer': 'No',  # "No" is concerning
            'typical_answer': 'Yes'
        },
        'A10_Score': {
            'question': 'Does your child stare at nothing with no apparent purpose?',
            'concerning_answer': 'Yes',  # "Yes" is concerning for A10
            'typical_answer': 'No'
        }
    }
    
    concern_items = []
    
    for key, value in data.items():
      if key in questions_map and value == 1:  # value=1 means concerning
        question_info = questions_map[key]
        concern_items.append(f"""
        <div style="margin: 10px 0; padding: 15px; background: #f8fdf8; border-radius: 10px; border: 2px solid #c8e6c9;">
            <div style="font-weight: 600; color: #1b5e20; margin-bottom: 5px;">{question_info['question']}</div>
            <div style="color: #2e7d32; font-size: 14px;">Answer: <strong>{question_info['concerning_answer']}</strong> (concerning behavior)</div>
        </div>
        """)
    
    if concern_items:
        for item in concern_items:
            st.html(item)
    else:
        st.html('<div style="margin: 10px 0; padding: 15px; background: white; border-radius: 10px; border: 2px solid #e8f5e9; color: #2e7d32; text-align: center;">No concerning responses identified</div>')
    
    st.html("</div>")

with col2:
    st.html("""
    <div style="
        background: linear-gradient(135deg, #f8fdf8 0%, #e8f5e9 100%);
        border-radius: 18px;
        padding: 30px;
        border: 2px solid rgba(129, 199, 132, 0.3);
        height: 100%;
    ">
        <h3 style="color: #1b5e20; margin-top: 0;">Child Information</h3>
    """)
    
    # Additional risk factors
    risk_factors = []
    if data.get('family_asd') == 'Yes':
        risk_factors.append("Family history of autism")
    if data.get('jundice') == 'Yes':
        risk_factors.append("Born with jaundice")
    if data.get('age', 0) < 4:
        risk_factors.append("Young age (< 4 years)")
    
    info_items = [
        f"<strong>Age:</strong> {data.get('age', 'Not specified')} years",
        f"<strong>Gender:</strong> {data.get('gender', 'Not specified')}",
        f"<strong>Jaundice at birth:</strong> {data.get('jundice', 'Not specified')}",
        f"<strong>Family history:</strong> {data.get('family_asd', 'Not specified')}"
    ]
    
    if risk_factors:
        info_items.append(f"<strong>Risk factors:</strong> {', '.join(risk_factors)}")
    
    for item in info_items:
        st.html(f'<div style="margin: 10px 0; padding: 10px; background: white; border-radius: 10px; border: 2px solid #e8f5e9; color: #2e7d32;">{item}</div>')
    
    st.html("</div>")

# Action Plan based on risk levels (now consistent with predictor)
st.header("💡 Recommended Next Steps")

# Determine action plan based on risk level
if "🟢 Low Risk" in risk_level or "🟢 Faible" in risk_level:
    st.html("""
    <div style="
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-radius: 18px;
        padding: 30px;
        border: 2px solid #81c784;
        margin-bottom: 20px;
    ">
        <h4 style="color: #1b5e20; margin-top: 0;">🟢 Low Risk - Continue Monitoring</h4>
        <ol style="color: #1b5e20; font-size: 16px; line-height: 1.6;">
            <li style="margin-bottom: 15px;"><strong>Continue monitoring</strong> your child's development milestones</li>
            <li style="margin-bottom: 15px;">Schedule regular <strong>pediatric check-ups</strong> as recommended</li>
            <li style="margin-bottom: 15px;">Engage in <strong>interactive play</strong> to support social skills development</li>
            <li style="margin-bottom: 15px;">Maintain a <strong>developmental journal</strong> to track progress</li>
        </ol>
    </div>
    """)
    
elif "🟡 Moderate Risk" in risk_level or "🟡 Moyen" in risk_level:
    st.html("""
    <div style="
        background: linear-gradient(135deg, #fff3e0 0%, #ffecb3 100%);
        border-radius: 18px;
        padding: 30px;
        border: 2px solid #ffb74d;
        margin-bottom: 20px;
    ">
        <h4 style="color: #e65100; margin-top: 0;">🟡 Moderate Risk - Enhanced Monitoring</h4>
        <ol style="color: #e65100; font-size: 16px; line-height: 1.6;">
            <li style="margin-bottom: 15px;"><strong>Share these results</strong> with your pediatrician at your next visit</li>
            <li style="margin-bottom: 15px;">Request a <strong>developmental screening</strong> within the next 3-6 months</li>
            <li style="margin-bottom: 15px;">Consider <strong>early intervention services</strong> if available in your area</li>
            <li style="margin-bottom: 15px;"><strong>Monitor closely</strong> the specific areas mentioned above</li>
            <li style="margin-bottom: 15px;">Join a <strong>parent support group</strong> for guidance and resources</li>
        </ol>
    </div>
    """)
    
elif "🟠 High Risk" in risk_level:
    st.html("""
    <div style="
        background: linear-gradient(135deg, #ffe0e0 0%, #ffb3b3 100%);
        border-radius: 18px;
        padding: 30px;
        border: 2px solid #ff6b6b;
        margin-bottom: 20px;
    ">
        <h4 style="color: #c62828; margin-top: 0;">🟠 High Risk - Professional Evaluation Recommended</h4>
        <ol style="color: #c62828; font-size: 16px; line-height: 1.6;">
            <li style="margin-bottom: 15px;"><strong>Schedule an appointment</strong> with a developmental pediatrician within 1-2 months</li>
            <li style="margin-bottom: 15px;">Contact your local <strong>early intervention program</strong> immediately</li>
            <li style="margin-bottom: 15px;">Request a <strong>comprehensive evaluation</strong> including speech and occupational therapy assessments</li>
            <li style="margin-bottom: 15px;">Document <strong>specific behaviors and concerns</strong> to share with specialists</li>
            <li style="margin-bottom: 15px;">Explore <strong>educational resources</strong> and therapy options in your area</li>
        </ol>
    </div>
    """)
    
elif "🔴 Very High Risk" in risk_level or "🔴 Élevé" in risk_level:
    st.html("""
    <div style="
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-radius: 18px;
        padding: 30px;
        border: 2px solid #ef5350;
        margin-bottom: 20px;
    ">
        <h4 style="color: #b71c1c; margin-top: 0;">🔴 Very High Risk - Urgent Evaluation Needed</h4>
        <ol style="color: #b71c1c; font-size: 16px; line-height: 1.6;">
            <li style="margin-bottom: 15px;"><strong>Schedule an urgent appointment</strong> with a developmental pediatrician or child psychologist</li>
            <li style="margin-bottom: 15px;">Contact your local <strong>early intervention program</strong> as soon as possible</li>
            <li style="margin-bottom: 15px;">Request a <strong>comprehensive multidisciplinary evaluation</strong></li>
            <li style="margin-bottom: 15px;">Join a <strong>parent support group</strong> for immediate guidance and emotional support</li>
            <li style="margin-bottom: 15px;">Begin researching <strong>therapeutic interventions and educational options</strong></li>
            <li style="margin-bottom: 15px;">Consider <strong>genetic counseling</strong> if there's a strong family history</li>
        </ol>
    </div>
    """)

# Resources
st.header("📚 Helpful Resources")
resources = get_resources(probability)

for resource in resources:
    with st.expander(resource['title']):
        st.html(f"""
        <div style="
            background: white;
            border-radius: 12px;
            padding: 20px;
            border: 2px solid #e8f5e9;
        ">
            <p style="color: #2e7d32; line-height: 1.6; margin-bottom: 20px;">{resource['description']}</p>
        </div>
        """)
        if resource.get('link'):
            st.html(f"""
            <div style="text-align: right; margin-top: 10px;">
                <a href="{resource['link']}" target="_blank" style="
                    background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%);
                    color: white;
                    padding: 10px 20px;
                    border-radius: 10px;
                    text-decoration: none;
                    font-weight: 600;
                    display: inline-block;
                ">Learn more →</a>
            </div>
            """)

# Export and actions
st.html("<hr>")

# st.header("Actions")

# col1, col2, col3 = st.columns(3)

# with col1:
#     if st.button("🔄 Take Again", type="secondary", use_container_width=True):
#         st.session_state.screening_data = {}
#         st.switch_page("pages/2_📋_Screening.py")

# with col2:
#     if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
#         try:
#             pdf_path = generate_pdf_report(data, probability)
#             with open(pdf_path, "rb") as pdf_file:
#                 st.download_button(
#                     label="📥 Download Report",
#                     data=pdf_file,
#                     file_name="autism_screening_report.pdf",
#                     mime="application/pdf",
#                     use_container_width=True,
#                     type="primary"
#                 )
#         except Exception as e:
#             st.error(f"Error generating PDF: {e}")
#             st.info("Please try again or contact support.")

# with col3:
#     if st.button("🏠 Return Home", type="secondary", use_container_width=True):
#         st.switch_page("app.py")

# ===== FOOTER NOTE =====
st.html("""
<div style="
    text-align: center;
    padding: 25px 20px;
    margin-top: 50px;
    background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 100%);
    border-radius: 15px;
    border: 2px dashed #81c784;
">
    <p style="color: #2e7d32; margin: 0; font-size: 15px; font-weight: 600;">
        🔒 Your information is secure and private • 💡 These results are for informational purposes only
    </p>
    <p style="color: #666; margin: 10px 0 0 0; font-size: 14px;">
        Model accuracy: {accuracy} • Based on {samples} training samples
    </p>
</div>
""".format(
    accuracy=f"{predictor.accuracy:.1%}" if predictor.accuracy else "N/A",
    samples="161 labeled + 636 unlabeled"
))