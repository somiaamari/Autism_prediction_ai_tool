import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Understanding Autism | Autism Screening",
    page_icon="🧠",
    layout="wide"
)

# CSS pour la page
st.html("""
<style>
/* Style général */
.main-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

/* Sections */
.section-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fdf8 100%);
    border-radius: 20px;
    padding: 35px;
    border: 3px solid #e8f5e9;
    box-shadow: 0 10px 30px rgba(46, 125, 50, 0.08);
    margin-bottom: 40px;
    transition: transform 0.3s ease;
}

.section-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(46, 125, 50, 0.12);
}

/* Titres */
.section-title {
    color: #1b5e20;
    border-left: 5px solid #4caf50;
    padding-left: 15px;
    margin: 40px 0 25px 0;
    font-weight: 800;
}

.subsection-title {
    color: #2e7d32;
    margin: 30px 0 15px 0;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Cartes d'information */
.info-card {
    background: white;
    border-radius: 15px;
    padding: 25px;
    border: 2px solid #e8f5e9;
    margin: 20px 0;
    box-shadow: 0 5px 15px rgba(129, 199, 132, 0.1);
}

.question-card {
    background: #f8fdf8;
    border-radius: 15px;
    padding: 25px;
    border-left: 5px solid #4caf50;
    margin: 20px 0;
    border-right: 2px solid #e8f5e9;
    border-top: 2px solid #e8f5e9;
    border-bottom: 2px solid #e8f5e9;
}

/* Badges */
.badge {
    display: inline-block;
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    color: #1b5e20;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 14px;
    margin: 5px;
}

/* Images et vidéos */
.media-container {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    margin: 20px 0;
}

.caption {
    text-align: center;
    color: #666;
    font-size: 14px;
    margin-top: 10px;
    font-style: italic;
}

/* Timeline */
.timeline-item {
    border-left: 3px solid #4caf50;
    padding-left: 20px;
    margin: 25px 0;
    position: relative;
}

.timeline-item::before {
    content: "•";
    position: absolute;
    left: -8px;
    top: 0;
    color: #4caf50;
    font-size: 24px;
}

/* Boutons */
.action-btn {
    display: inline-block;
    background: linear-gradient(135deg, #2e7d32, #4caf50);
    color: white;
    padding: 12px 25px;
    border-radius: 10px;
    text-decoration: none;
    font-weight: 600;
    margin: 10px 5px;
    transition: all 0.3s ease;
    border: none;
    cursor: pointer;
}

.action-btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 20px rgba(46, 125, 50, 0.3);
}

.secondary-btn {
    display: inline-block;
    background: #e8f5e9;
    color: #1b5e20;
    padding: 12px 25px;
    border-radius: 10px;
    text-decoration: none;
    font-weight: 600;
    margin: 10px 5px;
    transition: all 0.3s ease;
    border: 2px solid #c8e6c9;
}

.secondary-btn:hover {
    background: #d0ebd0;
    transform: translateY(-3px);
}

/* Responsive */
@media (max-width: 768px) {
    .section-card {
        padding: 20px;
    }
    .info-card {
        padding: 15px;
    }
}
</style>
""")

# Titre principal
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
        Understanding Autism Spectrum Disorder
    </h1>
    
    <p style="
        font-size: 22px; 
        color: #388e3c; 
        max-width: 800px; 
        margin: 0 auto 30px;
        line-height: 1.5;
    ">
        A Guide for Parents: Recognizing Symptoms & Understanding Screening
    </p>
    

</div>
""")

# ===== SECTION 1: WHAT IS AUTISM? =====
st.html('<h2 class="section-title">Understanding Autism Spectrum Disorder</h2>')

col1, col2 = st.columns([2, 1])

with col1:
    st.html("""
    <div class="section-card">
        <h3>What is Autism?</h3>
        <p>Autism Spectrum Disorder (ASD) is a <strong>neurodevelopmental condition</strong> that affects 
        how a person communicates, interacts with others, and experiences the world around them.</p>
        
        <div class="info-card">
            <h4>🧩 Key Characteristics:</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0;">
                <span class="badge">Social Communication</span>
                <span class="badge">Social Interaction</span>
                <span class="badge">Repetitive Behaviors</span>
                <span class="badge">Sensory Sensitivities</span>
                <span class="badge">Specialized Interests</span>
            </div>
            
            <p><strong>Important:</strong> Autism is a <em>spectrum</em>, meaning it affects each person differently. 
            Some individuals may need significant support, while others may need less.</p>
        </div>
        
        <h4>📊 Prevalence:</h4>
        <ul>
        <li>1 in 36 children are diagnosed with autism (CDC, 2023)</li>
        <li>Boys are 4 times more likely to be diagnosed than girls</li>
        <li>Early diagnosis leads to better outcomes</li>
        <li>Symptoms usually appear by age 2-3</li>
        </ul>
    </div>
    """)

with col2:
    st.html("""
    <div class="media-container">
        <img src="https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=500&h=350&fit=crop" 
             style="width: 100%; height: 350px; object-fit: cover;" 
             alt="Child playing with colorful blocks">
        <p class="caption">Every child with autism has unique strengths and challenges</p>
    </div>
    
    <div style="margin-top: 20px; text-align: center;">
        <a href="https://www.youtube.com/watch?v=TJuwhCIQQTs" target="_blank" class="action-btn">
            🎥 What is Autism? (WHO)
        </a>
    </div>
    """)

# ===== SECTION 2: EARLY SIGNS & SYMPTOMS =====
st.html('<h2 class="section-title">Early Signs & Symptoms to Watch For</h2>')

st.html("""
<div class="section-card">
    <h3>👶 Developmental Milestones & Red Flags</h3>
    
    <div class="timeline-item">
        <h4>By 12 Months</h4>
        <p><strong>Expected:</strong> Responds to name, points at objects, says single words</p>
        <p><strong>Potential concern:</strong> No babbling, no gestures, doesn't respond to name</p>
    </div>
    
    <div class="timeline-item">
        <h4>By 18 Months</h4>
        <p><strong>Expected:</strong> Uses several single words, follows simple instructions</p>
        <p><strong>Potential concern:</strong> No single words, loss of previously acquired words</p>
    </div>
    
    <div class="timeline-item">
        <h4>By 24 Months</h4>
        <p><strong>Expected:</strong> Uses 2-word phrases, imitates actions, shows pretend play</p>
        <p><strong>Potential concern:</strong> No 2-word phrases, limited eye contact, unusual play patterns</p>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 30px;">
        <div class="info-card">
            <h4>🔍 Social Communication Signs</h4>
            <ul>
            <li>Limited eye contact</li>
            <li>Doesn't respond to name by 12 months</li>
            <li>Difficulty understanding others' feelings</li>
            <li>Delayed speech and language skills</li>
            <li>Doesn't point or wave goodbye</li>
            </ul>
        </div>
        
        <div class="info-card">
            <h4>🔄 Behavioral Signs</h4>
            <ul>
            <li>Repetitive movements (flapping, rocking)</li>
            <li>Lines up toys or objects</li>
            <li>Gets upset by minor changes</li>
            <li>Has obsessive interests</li>
            <li>Unusual reactions to sensory input</li>
            </ul>
        </div>
    </div>
    
    <div style="text-align: center; margin-top: 30px;">
        <a href="https://www.youtube.com/watch?v=WRRF4NZB3WQ" target="_blank" class="action-btn">
            🎥 Early Signs of Autism
        </a>
    </div>
</div>
""")

# ===== SECTION 3: UNDERSTANDING THE SCREENING QUESTIONS =====
st.html('<h2 class="section-title">Understanding Our Screening Questions</h2>')

st.html("""
<div class="section-card">
    <h3>🔬 Why These Specific Questions?</h3>
    <p>Our screening tool is based on validated autism screening instruments and clinical expertise. 
    Each question targets specific developmental areas associated with autism spectrum disorder.</p>
    
    <div style="background: #f1f8e9; padding: 20px; border-radius: 10px; margin: 20px 0;">
    <p style="margin: 0; color: #1b5e20;"><strong>💡 Note:</strong> These questions are designed to identify 
    <em>potential signs</em> of autism, not to provide a diagnosis. A formal diagnosis requires 
    comprehensive evaluation by qualified professionals.</p>
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


# Questions par catégorie avec VRAIS liens YouTube
categories = [
    {
        "title": "👥 Social Interaction Questions (A1-A4)",
        "questions": [
            {
                "number": "A1",
                "question": "Does your child respond when you call their name?",
                "explanation": "This assesses <strong>joint attention</strong> - the ability to share focus with another person. Children with autism often have difficulty responding to their name, which is one of the earliest signs.",
                "why_important": "Difficulty responding to name is a core social communication deficit in autism. It indicates challenges with social orientation and attention.",
                "video": "https://www.youtube.com/watch?v=WRRF4NZB3WQ",
                "video_title": "Early Autism Signs"
            },
            {
                "number": "A2",
                "question": "Does your child make eye contact when interacting?",
                "explanation": "Eye contact is fundamental to social interaction. Many children with autism avoid or have unusual eye contact patterns.",
                "why_important": "Appropriate eye contact is crucial for social learning and communication. Atypical eye contact is a hallmark of autism.",
                "video": "https://www.youtube.com/watch?v=K_DJOuj51jQ",
                "video_title": "Autism & Eye Contact"
            },
            {
                "number": "A3",
                "question": "Does your child share enjoyment with you?",
                "explanation": "This evaluates <strong>social referencing</strong> - looking to others to share emotional experiences. Children with autism may not seek to share enjoyment.",
                "why_important": "Sharing enjoyment builds social bonds. Lack of social sharing is a key diagnostic criterion for autism.",
                "video": "https://www.youtube.com/watch?v=jZ_L-y99ccs",
                "video_title": "Joint Attention"
            },
            {
                "number": "A4",
                "question": "Does your child respond to your smiles?",
                "explanation": "Smiling in response to others' smiles demonstrates social reciprocity - the back-and-forth flow of social interaction.",
                "why_important": "Social reciprocity deficits are central to autism diagnosis. This question measures basic social engagement.",
                "video": "https://www.youtube.com/watch?v=rvXYzD1NuG8",
                "video_title": "Social Development"
            }
        ]
    },
    {
        "title": "💬 Communication Questions (A5-A7)",
        "questions": [
            {
                "number": "A5",
                "question": "Does your child use gestures (pointing, waving)?",
                "explanation": "Gestures are precursors to verbal language. Pointing to show or request (declarative and imperative pointing) is particularly important.",
                "why_important": "Lack of pointing by 18 months is a strong predictor of autism. Gestures are crucial for early communication development.",
                "video": "https://www.youtube.com/watch?v=y2j-qMWxgEQ",
                "video_title": "Communication Gestures"
            },
            {
                "number": "A6",
                "question": "Does your child use words meaningfully?",
                "explanation": "This assesses functional language use beyond echolalia (repeating words without meaning).",
                "why_important": "Pragmatic language difficulties (using language appropriately in social contexts) are common in autism.",
                "video": "https://www.youtube.com/watch?v=ifOeX3K1Jxk",
                "video_title": "Language Development"
            },
            {
                "number": "A7",
                "question": "Does your child engage in pretend play?",
                "explanation": "Pretend play involves imagination and symbolic thinking (e.g., pretending a banana is a phone).",
                "why_important": "Limited pretend play is characteristic of autism and reflects challenges with symbolic thought and imagination.",
                "video": "https://www.youtube.com/watch?v=iFqTpup0wTE",
                "video_title": "Pretend Play & Autism"
            }
        ]
    },
    {
        "title": "🔄 Behavioral Pattern Questions (A8-A10)",
        "questions": [
            {
                "number": "A8",
                "question": "Does your child have unusual repetitive movements?",
                "explanation": "Stereotyped movements like hand-flapping, rocking, or spinning are common in autism.",
                "why_important": "Repetitive behaviors are one of the two core diagnostic domains for autism (DSM-5).",
                "video": "https://www.youtube.com/watch?v=2LhI23QPoi8",
                "video_title": "Repetitive Behaviors"
            },
            {
                "number": "A9",
                "question": "Does your child get unusually upset by small changes?",
                "explanation": "This assesses <strong>insistence on sameness</strong> - a need for routine and predictability.",
                "why_important": "Resistance to change is a common behavioral characteristic of autism that can impact daily functioning.",
                "video": "https://www.youtube.com/watch?v=z8Dst6fSNTE",
                "video_title": "Routine & Autism"
            },
            {
                "number": "A10",
                "question": "Does your child have intense, focused interests?",
                "explanation": "Highly restricted, fixated interests that are abnormal in intensity or focus.",
                "why_important": "Circumscribed interests are included in DSM-5 diagnostic criteria for autism.",
                "video": "https://www.youtube.com/watch?v=qwYBdIL9b90",
                "video_title": "Special Interests"
            }
        ]
    }
]

# Afficher chaque catégorie
for category in categories:
    st.html(f'<h3 style="color: #2e7d32; margin-top: 40px;">{category["title"]}</h3>')
    
    for q in category["questions"]:
        st.html(f"""
        <div class="question-card">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;">
                <h4 style="margin: 0; color: #1b5e20;">{q['number']}: {q['question']}</h4>
                <span class="badge" style="background: #e8f5e9;">Screening Question</span>
            </div>
            
            <div style="margin: 15px 0; padding: 15px; background: white; border-radius: 10px;">
                <h5 style="color: #2e7d32; margin-top: 0;">🧠 What This Measures:</h5>
                <p>{q['explanation']}</p>
                
                <h5 style="color: #2e7d32;">🎯 Why It's Important for Screening:</h5>
                <p>{q['why_important']}</p>
            </div>
            
            <div style="text-align: center; margin-top: 15px;">
                <a href="{q['video']}" target="_blank" class="action-btn" style="padding: 8px 15px; font-size: 14px;">
                    🎬 {q['video_title']}
                </a>
            </div>
        </div>
        """)

# ===== SECTION 4: EXPERT INSIGHTS =====
# st.html('<h2 class="section-title">Expert Insights & Professional Perspectives</h2>')

# col1, col2 = st.columns(2)

# with col1:
#     st.html("""
#     <div class="section-card">
#         <h3>👨‍⚕️ Dr. Sarah Johnson, Developmental Pediatrician</h3>
#         <div class="media-container">
#             <img src="https://images.unsplash.com/photo-1551601651-2a8555f1a136?w=400&h=250&fit=crop" 
#                  style="width: 100%; height: 200px; object-fit: cover;" 
#                  alt="Doctor consulting with parents">
#         </div>
        
#         <div style="margin-top: 20px;">
#             <p><strong>"Early screening is crucial because:</strong></p>
#             <ul>
#             <li>Brain plasticity is highest in early childhood</li>
#             <li>Early intervention can improve outcomes by 30-40%</li>
#             <li>It reduces parental stress by providing answers sooner</li>
#             <li>It connects families with resources and support"</li>
#             </ul>
#         </div>
        
    #     <div style="text-align: center; margin-top: 20px;">
    #         <a href="https://youtu.be/6fy7gUIjN_c" target="_blank" class="action-btn">
    #             🎥 What is Autism? (WHO)
    #         </a>
    #     </div>
    # </div>
    # """)

# with col2:
#     st.html("""
#     <div class="section-card">
#         <h3>👩‍🏫 Dr. Michael Chen, Child Psychologist</h3>
#         <div class="media-container">
#             <img src="https://images.unsplash.com/photo-1576091160399-112ba8d25d1f?w=400&h=250&fit=crop" 
#                  style="width: 100%; height: 200px; object-fit: cover;" 
#                  alt="Therapist working with child">
#         </div>
        
#         <div style="margin-top: 20px;">
#             <p><strong>"Understanding screening results:</strong></p>
#             <ul>
#             <li>Screening tools are like 'check engine' lights - they indicate when to seek professional evaluation</li>
#             <li>No single behavior confirms autism - it's about patterns</li>
#             <li>Cultural factors can influence how symptoms present</li>
#             <li>Girls may show different symptoms than boys"</li>
#             </ul>
#         </div>
        
    #     <div style="text-align: center; margin-top: 20px;">
    #         <a href="https://youtu.be/w_8x9QkHnkU" target="_blank" class="action-btn">
    #             🎥 Early Signs of Autism
    #         </a>
    #     </div>
    # </div>
    # """)

# Section vidéos éducatives supplémentaires
st.html('<h3 style="color: #1b5e20; margin-top: 40px;">🎬 Additional Educational Resources</h3>')

st.html("""
<div class="section-card">
    <h4>Recommended Videos for Parents</h4>
    <div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0;">
        <a href="https://www.youtube.com/watch?v=hwaaphuStxY" target="_blank" class="badge" style="text-decoration: none;">
            🎥 What is Autism? (WHO)
        </a>
        <a href="https://www.youtube.com/watch?v=YtvP5A5OHpU" target="_blank" class="badge" style="text-decoration: none;">
            🎥 Early Signs of Autism
        </a>
        <a href="https://www.youtube.com/watch?v=So05QaAjkKw" target="_blank" class="badge" style="text-decoration: none;">
            🎥 Autism Diagnosis Process
        </a>
        <a href="https://www.youtube.com/watch?v=9iGfOxtiV-M" target="_blank" class="badge" style="text-decoration: none;">
            🎥 Autism & Communication
        </a>
        <a href="https://www.youtube.com/watch?v=uunOBIBpi6E" target="_blank" class="badge" style="text-decoration: none;">
            🎥 Autism & Social Skills
        </a>
        <a href="https://www.youtube.com/watch?v=Cqu8r640cA4" target="_blank" class="badge" style="text-decoration: none;">
            🎥 Understanding Behaviors
        </a>
    </div>
    
    <div style="margin-top: 20px;">
        <h5>📚 Other Valuable Resources:</h5>
        <ul>
        <li><strong>Autism Speaks:</strong> <a href="https://www.autismspeaks.org" target="_blank">www.autismspeaks.org</a></li>
        <li><strong>CDC Autism Resources:</strong> <a href="https://www.cdc.gov/ncbddd/autism" target="_blank">www.cdc.gov/ncbddd/autism</a></li>
        <li><strong>Autism Society:</strong> <a href="https://www.autism-society.org" target="_blank">www.autism-society.org</a></li>
        <li><strong>National Autism Association:</strong> <a href="https://nationalautismassociation.org" target="_blank">nationalautismassociation.org</a></li>
        </ul>
    </div>
</div>
""")

# ===== SECTION 5: NEXT STEPS AFTER SCREENING =====
st.html('<h2 class="section-title">Next Steps & Resources</h2>')

st.html("""
<div class="section-card">
    <h3>📋 If Screening Suggests Further Evaluation</h3>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0;">
        <div class="info-card">
            <h4>1. Consult Your Pediatrician</h4>
            <p>Share your screening results with your child's doctor. They can provide referrals to specialists.</p>
        </div>
        
        <div class="info-card">
            <h4>2. Seek Comprehensive Evaluation</h4>
            <p>This typically involves a multidisciplinary team including developmental pediatricians, psychologists, and speech therapists.</p>
        </div>
        
        <div class="info-card">
            <h4>3. Early Intervention Services</h4>
            <p>Even before formal diagnosis, children can benefit from early intervention services if developmental delays are suspected.</p>
        </div>
    </div>
    
    <div style="background: linear-gradient(135deg, #e8f5e9, #c8e6c9); padding: 25px; border-radius: 15px; margin-top: 25px;">
        <h4 style="color: #1b5e20; margin-top: 0;">💡 Remember:</h4>
        <p style="color: #1a472a; margin-bottom: 0;">
        <strong>Autism is not a tragedy - it's a different way of experiencing the world.</strong> 
        With understanding, support, and appropriate interventions, individuals with autism can thrive 
        and reach their full potential. Early identification is the first step toward providing 
        the right support at the right time.
        </p>
    </div>
</div>
""")

# ===== CALL TO ACTION =====
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
    
    <p style="color: #666; margin-top: 30px; font-size: 14px;">
    <em>This educational content is for informational purposes only and should not replace 
    professional medical advice. Always consult with qualified healthcare providers for diagnosis 
    and treatment decisions.</em>
    </p>
</div>
""")

st.html('</div>')