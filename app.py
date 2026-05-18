import streamlit as st
from pathlib import Path

# ====================
# PAGE CONFIGURATION
# ====================
st.set_page_config(
    page_title="Autism Screening Tool",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================
# NAVIGATION SETUP
# ====================
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

def navigate_to(page_name):
    """Universal navigation function"""
    st.session_state.current_page = page_name
    st.rerun()

# ====================
# GREEN THEME CSS STYLING
# ====================
st.markdown("""
<style>
    /* ===== MAIN GREEN THEME ===== */
    
    /* Main app background - soft green gradient */
    .stApp {
        background: linear-gradient(135deg, #f0fff4 0%, #e6fffa 100%) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Top header - light green gradient */
    header[data-testid="stHeader"] {
        background: linear-gradient(135deg, #d4f7d4 0%, #b0eac0 100%) !important;
        border-bottom: 1px solid #90c8a0 !important;
    }
    
    header[data-testid="stHeader"] * {
        color: #1a472a !important;
        font-weight: 500;
    }
    
    /* Sidebar background - calming green gradient */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e8f5e9 0%, #d0ebd0 100%) !important;
    border-right: 1px solid #b0d4b0 !important;
}

/* ===== FORCER TOUT LE TEXTE DE LA SIDEBAR À ÊTRE BLANC ===== */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Spécifiquement pour les boutons de Streamlit dans la sidebar */
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] .stButton > button span,
section[data-testid="stSidebar"] .stButton > button div,
section[data-testid="stSidebar"] .stButton > button p,
section[data-testid="stSidebar"] .stButton > button label {
    color: white !important;
    font-weight: 700 !important;
}

/* ===== NAVIGATION BUTTONS ===== */
.nav-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 20px 15px;
}

.stButton > button {
    width: 100% !important;
    height: 80px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    border: none !important;
    margin: 8px 0 !important;
    transition: all 0.3s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 4px 6px rgba(46, 125, 50, 0.1) !important;
}

/* Couleurs des boutons */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%) !important;
}

.stButton > button[kind="secondary"]:nth-of-type(1) {
    background: linear-gradient(135deg, #00695c 0%, #26a69a 100%) !important;
}

.stButton > button[kind="secondary"]:nth-of-type(2) {
    background: linear-gradient(135deg, #00796b 0%, #4db6ac 100%) !important;
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 12px rgba(46, 125, 50, 0.2) !important;
    filter: brightness(1.1) !important;
}

.stButton > button[kind="primary"] {
    box-shadow: 0 0 15px rgba(76, 175, 80, 0.3) !important;
}
    /* ===== SIDEBAR HEADER ===== */
    .sidebar-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #388e3c 0%, #66bb6a 100%);
        border-radius: 12px;
        margin: 0 10px 30px 10px;
        color: white !important;
        box-shadow: 0 4px 6px rgba(56, 142, 60, 0.2);
    }
    
    .sidebar-header * {
        color: white !important;
    }
    
    /* ===== MAIN CONTENT ===== */
    .main .block-container *,
    div[data-testid="stAppViewContainer"] *:not(.stButton button):not(.sidebar-header *) {
        color: #1a472a !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1b5e20 !important;
        font-weight: 600;
    }
    
    /* ===== HIDE UNWANTED ELEMENTS ===== */
    div[data-testid="stSidebarNav"] {
        display: none !important;
    }
    
    footer {
        display: none !important;
    }
    /* ===== TRANSPARENT HEADER ===== */
    
    /* Make header completely transparent */
    header[data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
    }
    
    /* Header inner elements */
    header[data-testid="stHeader"] > div {
        background: transparent !important;
    }
    
    /* Header text - make visible on transparent background */
    header[data-testid="stHeader"] * {
        color: #1a472a !important;  /* Dark green for contrast */
        text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8) !important;
    }
    
    /* Menu button - subtle on transparent */
    button[data-testid="baseButton-header"] {
        color: #1a472a !important;
        background: rgba(255, 255, 255, 0.8) !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: blur(5px);
    }
    
    button[data-testid="baseButton-header"]:hover {
        background: rgba(255, 255, 255, 0.95) !important;
        border-color: rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Remove any decorations */
    .st-emotion-cache-1dp5vir {
        background: transparent !important;
        height: 0 !important;
    }
    
    /* Ensure header shows through */
    .stApp > header {
        background: transparent !important;
    }
    
    /* App content starts right under header */
    .main .block-container {
        padding-top: 0 !important;
    }
            /* ===== INPUT FIELDS - Light background for forms ===== */

/* Pour tous les inputs dans la page principale */
   div[data-testid="stForm"] input,
   div[data-testid="stForm"] textarea,
   div[data-testid="stForm"] select,
   .stTextInput > div > div > input,
   .stNumberInput > div > div > input,
   .stSelectbox > div > div > select,
   .stTextArea > div > div > textarea,
   .stDateInput > div > div > input,
   .stTimeInput > div > div > input {
     background-color: #ffffff !important;
     color: #333333 !important;
     border: 1px solid #b0d4b0 !important;
     border-radius: 8px !important;
    }

/* Labels des champs */
    .stTextInput label,
    .stNumberInput label,
    .stSelectbox label,
    .stTextArea label,
    .stDateInput label,
    .stTimeInput label {
     color: #2e7d32 !important; /* Vert pour correspondre au thème */
     font-weight: 600 !important;
   }

/* Focus state */
   div[data-testid="stForm"] input:focus,
   div[data-testid="stForm"] textarea:focus,
   div[data-testid="stForm"] select:focus,
   .stTextInput > div > div > input:focus,
   .stNumberInput > div > div > input:focus,
   .stSelectbox > div > div > select:focus,
   .stTextArea > div > div > textarea:focus {
     border-color: #4caf50 !important;
     box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2) !important;
     outline: none !important;
   }
</style>
""", unsafe_allow_html=True)

# ====================
# SIDEBAR NAVIGATION (ONLY NAVIGATION)
# ====================
with st.sidebar:
    # Logo at the very top
    # st.image("assets/logo.png", width=150)
        # ===== CENTERED LOGO =====
    # Create 3 columns and put image in middle column
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
      st.image("assets/logo.png", width=150)
    
    # Navigation buttons container
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="sidebar-header">
        <h2 style="margin: 0; font-size: 24px;">🧩 Autism Screening</h2>
        <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 14px;">Supportive Growth Tool</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons - ONLY PLACE WITH BUTTONS
    st.markdown('<div class="nav-container" style="color=white;">', unsafe_allow_html=True)
    
    pages = [
        ("Home", "🏠 Home"),
        ("Learning", "📚 Understand Autism"),
        ("Screening", "📋 Screening"),
        ("Results", "📊 Results"),
        ("About", "📈 About")
    ]

    
    for page_key, page_display in pages:
        is_active = st.session_state.current_page == page_key
        button_type = "primary" if is_active else "secondary"
        
        if st.button(
            page_display,
            key=f"sidebar_btn_{page_key}",
            type=button_type,
            use_container_width=True
        ):
            navigate_to(page_key)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="
        padding: 15px; 
        background: rgba(200, 230, 201, 0.5); 
        border-radius: 10px; 
        border-left: 4px solid #388e3c;
    ">
        <p style="font-size: 13px; color: #1a472a; margin: 0;">
        <strong>🌿 Important Note</strong><br>
        This is a screening tool for supportive guidance only.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ====================
# MAIN CONTENT (NO BUTTONS)
# ====================
selected_page = st.session_state.current_page

# Load page content
page_files = {
    "Home": "1_Home.py",
    'Learning': "5_Learning.py",
    "Screening": "2_Screening.py",
    "Results": "3_Results.py",
    "About": "4_About.py"
}

filename = page_files.get(selected_page)
if filename:
    page_path = Path("pages") / filename
    
    if page_path.exists():
        try:
            # Read and execute page content
            with open(page_path, "r", encoding="utf-8") as f:
                page_content = f.read()
            
            # Execute with only st available
            namespace = {"st": st}
            exec(page_content, namespace)
            
        except Exception as e:
            st.error(f"Error loading page: {str(e)}")
            # show_fallback_content(selected_page)
    else:
        st.error(f"Page file not found: {page_path}")
        # show_fallback_content(selected_page)
else:
    st.error(f"Unknown page: {selected_page}")
    navigate_to("Home")

# ====================
# HELPER FUNCTION
# ====================
def show_fallback_content(page_name):
    """Show fallback content without buttons"""
    if page_name == "Home":
        st.title("🏠 Welcome to Autism Screening Tool")
        st.markdown("""
        ## A Supportive First Step
        
        This tool helps parents understand your child's development 
        through a simple, evidence-based questionnaire.
        
        ### 🌱 How to Use:
        **Use the buttons in the sidebar to navigate:**
        1. Click **Screening** in the sidebar to begin
        2. Answer questions about your child's behavior
        3. View results and personalized recommendations
        
        ### ⚠️ Important Note
        **This is a screening tool, not a medical diagnosis.** 
        Always consult with a qualified healthcare professional.
        """)
                
    elif page_name == "Screening":
        st.title("📋 Screening Questionnaire")
        st.markdown("""
        ## Understanding Your Child's Development
        
        Please use the **Screening** button in the sidebar 
        to access the questionnaire.
        
        The questionnaire includes questions about:
        - Social interactions and communication
        - Behavioral patterns and routines
        - Age-appropriate developmental milestones
        """)
        
    elif page_name == "Results":
        st.title("📊 Results & Interpretation")
        st.markdown("""
        ## Personalized Screening Results
        
        Complete the screening questionnaire first to see 
        personalized results and recommendations.
        
        Results will include:
        - Probability assessment
        - Developmental area breakdown
        - Recommended next steps
        - Support resources
        """)

# ====================
# FOOTER
# ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #4caf50; font-size: 14px;">
    <p>🧩 Autism Screening Tool • Supportive Guidance • Green for Growth 🌱</p>
</div>
""", unsafe_allow_html=True)