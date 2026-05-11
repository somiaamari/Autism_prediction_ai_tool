import streamlit as st

# ====================
# HERO SECTION - FIXED
# ====================
st.html("""
<div style="
    text-align: center; 
    padding: 60px 20px; 
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    border-radius: 20px; 
    margin-bottom: 50px;
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
    ">🧩🧩🧩</div>
    
    <h1 style="
        font-size: 48px; 
        color: #1b5e20; 
        margin-bottom: 15px;
        font-weight: 700;
    ">
        🧩 Autism Screening Tool
    </h1>
    
    <p style="
        font-size: 22px; 
        color: #388e3c; 
        max-width: 800px; 
        margin: 0 auto 30px;
        line-height: 1.5;
    ">
        A compassionate, evidence-based approach to understanding your child's development
    </p>
    
    
    
    <div style="
        margin-top: 30px;
        padding: 15px;
        background: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
        display: inline-block;
    ">
        <p style="margin: 0; color: #555; font-size: 16px;">
            Use the buttons in the sidebar to get started!
        </p>
    </div>
</div>
""")
# Crée 4 colonnes
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image("C:/Users/Zbook/Desktop/semestre1ai/ml/autism/assets/social_interactions.png", use_container_width=True)


with col2:
    st.image("C:/Users/Zbook/Desktop/semestre1ai/ml/autism/assets/communication.png", use_container_width=True)


with col3:
    st.image("C:/Users/Zbook/Desktop/semestre1ai/ml/autism/assets/behavior.png", use_container_width=True)


with col4:
    st.image("C:/Users/Zbook/Desktop/semestre1ai/ml/autism/assets/early_signs.png", use_container_width=True)

# ====================
# QUICK START GUIDE
# ====================
st.markdown("""
<div style="
    background: linear-gradient(to right, #f1f8e9, #e8f5e9);
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 40px;
    border-left: 6px solid #4caf50;
">
    <h2 style="color: #1b5e20; margin-top: 0;">🚀 Quick Start Guide</h2>
    <p style="color: #555; font-size: 16px;">
        Getting started is simple. Just follow these three steps:
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div style="
        text-align: center;
        padding: 25px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.08);
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border-top: 5px solid #81c784;
    ">
        <div style="font-size: 42px; margin-bottom: 15px;">1️⃣</div>
        <h3 style="color: #1b5e20; margin: 0 0 10px 0;">Navigate</h3>
        <p style="color: #666; margin: 0;">
            Use the <strong>gradient buttons</strong> in the sidebar
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="
        text-align: center;
        padding: 25px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.08);
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border-top: 5px solid #66bb6a;
    ">
        <div style="font-size: 42px; margin-bottom: 15px;">2️⃣</div>
        <h3 style="color: #1b5e20; margin: 0 0 10px 0;">Screen</h3>
        <p style="color: #666; margin: 0;">
            Answer <strong>10-15 simple questions</strong> about behavior
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div style="
        text-align: center;
        padding: 25px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.08);
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border-top: 5px solid #4caf50;
    ">
        <div style="font-size: 42px; margin-bottom: 15px;">3️⃣</div>
        <h3 style="color: #1b5e20; margin: 0 0 10px 0;">Learn</h3>
        <p style="color: #666; margin: 0;">
            Get <strong>personalized guidance</strong> and next steps
        </p>
    </div>
    """, unsafe_allow_html=True)

# ====================
# FEATURE CARDS
# ====================
st.markdown("""
<div style="
    background: linear-gradient(to right, #f1f8e9, #e8f5e9);
    padding: 25px;
    border-radius: 15px;
    margin: 50px 0 30px 0;
    border-left: 6px solid #388e3c;
">
    <h2 style="color: #1b5e20; margin-top: 0;">✨ Why Choose Our Tool</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(46, 125, 50, 0.12);
        height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div style="font-size: 48px; margin-bottom: 20px;">🔍</div>
        <h3 style="color: #1b5e20; margin: 0 0 15px 0;">Evidence-Based</h3>
        <p style="color: #555; margin: 0; line-height: 1.5;">
        Based on clinically-informed methods used by professionals worldwide
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(46, 125, 50, 0.12);
        height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div style="font-size: 48px; margin-bottom: 20px;">🤗</div>
        <h3 style="color: #1b5e20; margin: 0 0 15px 0;">Supportive Design</h3>
        <p style="color: #555; margin: 0; line-height: 1.5;">
        Created with empathy to provide guidance, not judgment
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(46, 125, 50, 0.12);
        height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div style="font-size: 48px; margin-bottom: 20px;">🔒</div>
        <h3 style="color: #1b5e20; margin: 0 0 15px 0;">Completely Private</h3>
        <p style="color: #555; margin: 0; line-height: 1.5;">
        Your responses are temporary and never stored permanently
        </p>
    </div>
    """, unsafe_allow_html=True)

# ====================
# TIMELINE HOW-IT-WORKS
# ====================
st.markdown("""
<div style="
    background: linear-gradient(to right, #f1f8e9, #e8f5e9);
    padding: 25px;
    border-radius: 15px;
    margin: 50px 0 30px 0;
    border-left: 6px solid #2e7d32;
">
    <h2 style="color: #1b5e20; margin-top: 0;">📋 How It Works</h2>
</div>
""", unsafe_allow_html=True)

steps = [
    ("1", "📋 Start Screening", "Click 'Screening' in the sidebar to begin the questionnaire", "#4caf50"),
    ("2", "❓ Answer Questions", "10-15 simple questions about social and behavioral patterns", "#66bb6a"),
    ("3", "📊 View Results", "Get immediate, personalized results with visual indicators", "#81c784"),
    ("4", "💡 Get Guidance", "Receive actionable next steps and professional recommendations", "#a5d6a7")
]

for num, title, desc, color in steps:
    step_html = f"""
    <div style="
        display: flex;
        align-items: start;
        margin: 25px 0;
        padding: 20px;
        background: {'#f1f8e9' if int(num) % 2 == 0 else '#e8f5e9'};
        border-radius: 12px;
        border-left: 6px solid {color};
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    ">
        <div style="
            background: {color};
            color: white;
            width: 45px;
            height: 45px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 20px;
            font-weight: bold;
            font-size: 18px;
            flex-shrink: 0;
        ">{num}</div>
        <div>
            <h3 style="margin: 0 0 8px 0; color: #1b5e20;">{title}</h3>
            <p style="margin: 0; color: #555; line-height: 1.5;">{desc}</p>
        </div>
    </div>
    """
    st.markdown(step_html, unsafe_allow_html=True)

# ====================
# AGE GUIDE (Using Streamlit Native)
# ====================
# st.markdown("### 👶 Developmental Milestones Guide")

# ages = [
#     ("18-24 months", "First words, pointing, responding to name"),
#     ("2-3 years", "Simple sentences, pretend play, following instructions"),
#     ("3-4 years", "Conversations, sharing, understanding emotions"),
#     ("4-5 years", "Complex sentences, cooperative play, basic reasoning")
# ]

# for age_range, milestones in ages:
#     with st.container():
#         st.markdown(f"**{age_range}**")
#         st.markdown(f"{milestones}")
#         st.markdown("---")

# ====================
# FAQ (Using Streamlit Native)
# ====================
# with st.expander("❓ Frequently Asked Questions", expanded=False):
#     st.markdown("**How long does it take?**")
#     st.markdown("Approximately 10-15 minutes.")
#     st.markdown("---")
    
#     st.markdown("**Is this a diagnosis?**")
#     st.markdown("No, this is a screening tool only for guidance.")
#     st.markdown("---")
    
#     st.markdown("**Is my data saved?**")
#     st.markdown("No, all responses are temporary.")
#     st.markdown("---")
    
#     st.markdown("**What age is this for?**")
#     st.markdown("Designed for children 18 months to 5 years.")
#     st.markdown("---")

# ====================
# INSPIRATIONAL SECTION
# ====================

st.html("""
<div style="
    text-align: center;
    padding: 40px;
    background: linear-gradient(135deg, #f1f8e9, #e8f5e9);
    border-radius: 15px;
    margin: 50px 0;
    border: 2px dashed #81c784;
">
    <div style="
        position: relative;
        top: -25px;
        background: white;
        padding: 0 25px;
        color: #4caf50;
        font-size: 24px;
        font-weight: bold;
        display: inline-block;
    ">🌿</div>
    
    <h3 style="color: #1b5e20; margin-bottom: 15px; font-size: 24px;">
        Why We Chose Green
    </h3>
    <p style="
        max-width: 800px; 
        margin: 0 auto; 
        color: #555;
        font-size: 16px;
        line-height: 1.6;
    ">
        Green represents <strong style="color: #2e7d32;">growth</strong>, 
        <strong style="color: #388e3c;">calm</strong>, 
        <strong style="color: #4caf50;">hope</strong>, and 
        <strong style="color: #66bb6a;">nature</strong>.
    </p>
</div>
""")

# ====================
# DISCLAIMER (Using Streamlit Native)
# ====================
st.warning("""
**⚠️ Important Medical Disclaimer**

This tool provides screening information only and is not a medical diagnosis. 
Always consult with qualified healthcare professionals for proper assessment.
""")

# ====================
# FINAL CALL TO ACTION
# ====================

st.html("""
<div style="
    text-align: center; 
    padding: 40px; 
    background: linear-gradient(135deg, #4caf50, #388e3c);
    border-radius: 15px; 
    margin-top: 40px;
    box-shadow: 0 8px 25px rgba(76, 175, 80, 0.2);
">
    <h2 style="color: white; margin-bottom: 15px;">Ready to Begin?</h2>
    
    <p style="color: rgba(255, 255, 255, 0.9); font-size: 18px; margin-bottom: 25px;">
        Take the first step in understanding your child's development
    </p>
    
    <div style="
        display: inline-block;
        padding: 15px 30px;
        background: white;
        color: #2e7d32;
        border-radius: 50px;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    ">
        👉 Click "📋 Screening" in the sidebar
    </div>
    
    <p style="color: rgba(255, 255, 255, 0.8); margin-top: 20px; font-size: 14px;">
        You've got this. Every journey begins with a single step. 🌱
    </p>
</div>
""")

# ====================
# FOOTER NOTE
# ====================

st.html( """
<div style="
    text-align: center;
    padding: 20px;
    color: #81c784;
    font-size: 14px;
    margin-top: 30px;
">
    <hr style="border: none; border-top: 1px solid #e0e0e0; margin: 20px 0;">
    <p style="margin: 0;">
        🧩 Autism Screening Tool • Designed with Care • Supporting Families
    </p>
</div>
""")

