import json
import os
from datetime import datetime
import pandas as pd

def save_screening_data(data):
    """Save screening data to a JSON file (optional)"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Add timestamp
        data['timestamp'] = datetime.now().isoformat()
        
        # Save to file (anonymized)
        filename = f"data/screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data: {e}")

def get_resources(probability):
    """Get appropriate resources based on probability"""
    resources_file = "data/resources.json"
    
    # Default resources if file doesn't exist
    default_resources = {
        "low": [
            {
                "title": "CDC Developmental Milestones",
                "description": "Track your child's development with age-appropriate milestones.",
                "link": "https://www.cdc.gov/ncbddd/actearly/milestones/index.html"
            }
        ],
        "medium": [
            {
                "title": "Talk to Your Pediatrician",
                "description": "How to prepare for and discuss developmental concerns with your doctor.",
                "link": ""
            },
            {
                "title": "Early Intervention Programs",
                "description": "Learn about free or low-cost early intervention services.",
                "link": ""
            }
        ],
        "high": [
            {
                "title": "Autism Speaks: First 100 Days Kit",
                "description": "A guide for families after an autism diagnosis.",
                "link": "https://www.autismspeaks.org/tool-kit/100-day-kit"
            },
            {
                "title": "Finding a Specialist",
                "description": "How to find and choose a developmental pediatrician.",
                "link": ""
            }
        ]
    }
    
    # Determine which resources to show
    if probability < 0.3:
        return default_resources["low"]
    elif probability < 0.7:
        return default_resources["medium"]
    else:
        return default_resources["high"]

def generate_pdf_report(data, probability):
    """Generate a simple PDF report (simplified version)"""
    # In a real implementation, use ReportLab or similar
    # For now, create a simple text file
    import tempfile
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
    
    report_content = f"""
    Autism Screening Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    
    SCREENING RESULTS:
    Probability Score: {probability:.1%}
    
    CHILD INFORMATION:
    Age: {data.get('age', 'N/A')} years
    Gender: {data.get('gender', 'N/A')}
    
    AREAS ASSESSED:
    - Response to name: {'Concern' if data.get('A1') else 'Typical'}
    - Eye contact: {'Concern' if data.get('A2') else 'Typical'}
    - Social sharing: {'Concern' if data.get('A3') else 'Typical'}
    - Response to smiles: {'Concern' if data.get('A4') else 'Typical'}
    
    RECOMMENDATIONS:
    Please share this report with your healthcare provider.
    This is a screening tool, not a diagnostic assessment.
    """
    
    with open(temp_file.name, 'w') as f:
        f.write(report_content)
    
    return temp_file.name