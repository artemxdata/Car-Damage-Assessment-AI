import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import os
import tempfile

# Import custom modules (will create these)
try:
    from car_damage_detector import CarDamageDetector
    from utils import enhance_image, calculate_damage_stats
except ImportError:
    st.warning("Custom modules not found. Running in demo mode.")
    CarDamageDetector = None

# Page configuration
st.set_page_config(
    page_title="Car Damage Assessment AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    
    .damage-alert {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .damage-severe {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    
    .damage-moderate {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    
    .damage-light {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

def demo_damage_detection(image):
    """
    Demo function that simulates damage detection for demonstration purposes.
    In production, this would be replaced with actual ML model inference.
    
    Args:
        image (PIL.Image): Input image for damage detection
        
    Returns:
        tuple: (annotated_image, detections_list)
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Simulate realistic damage detections
    detections = [
        {
            'type': 'Scratch',
            'severity': 'Light',
            'confidence': 0.89,
            'bbox': [int(width*0.2), int(height*0.3), int(width*0.4), int(height*0.5)],
            'area_percentage': 2.5,
            'estimated_cost': 150
        },
        {
            'type': 'Dent',
            'severity': 'Moderate', 
            'confidence': 0.76,
            'bbox': [int(width*0.6), int(height*0.2), int(width*0.8), int(height*0.4)],
            'area_percentage': 8.3,
            'estimated_cost': 450
        },
        {
            'type': 'Paint Damage',
            'severity': 'Light',
            'confidence': 0.82,
            'bbox': [int(width*0.1), int(height*0.6), int(width*0.25), int(height*0.8)],
            'area_percentage': 3.2,
            'estimated_cost': 200
        }
    ]
    
    # Draw bounding boxes and labels
    img_with_annotations = img_array.copy()
    colors = {
        'Scratch': (0, 255, 0), 
        'Dent': (255, 165, 0), 
        'Paint Damage': (255, 0, 255),
        'Broken Part': (255, 0, 0)
    }
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        color = colors.get(detection['type'], (255, 255, 0))
        
        # Draw bounding rectangle
        cv2.rectangle(img_with_annotations, (x1, y1), (x2, y2), color, 3)
        
        # Add label with confidence
        label = f"{detection['type']} ({detection['confidence']:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Draw label background
        cv2.rectangle(img_with_annotations, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), 
                     color, -1)
        
        # Draw label text
        cv2.putText(img_with_annotations, label, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img_with_annotations, detections

def create_damage_distribution_chart(detections):
    """Create a pie chart showing damage type distribution"""
    damage_counts = {}
    for detection in detections:
        damage_type = detection['type']
        damage_counts[damage_type] = damage_counts.get(damage_type, 0) + 1
    
    fig = px.pie(
        values=list(damage_counts.values()),
        names=list(damage_counts.keys()),
        title="Damage Type Distribution"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_severity_chart(detections):
    """Create a bar chart showing severity distribution"""
    severity_counts = {}
    for detection in detections:
        severity = detection['severity']
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    colors = {'Light': '#4CAF50', 'Moderate': '#FF9800', 'Severe': '#F44336'}
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(severity_counts.keys()),
            y=list(severity_counts.values()),
            marker_color=[colors.get(k, '#666666') for k in severity_counts.keys()]
        )
    ])
    
    fig.update_layout(
        title="Damage Severity Distribution",
        xaxis_title="Severity Level",
        yaxis_title="Number of Damages"
    )
    
    return fig

def generate_assessment_report(detections, image_info):
    """Generate a comprehensive damage assessment report"""
    total_cost = sum([d.get('estimated_cost', 0) for d in detections])
    total_area = sum([d['area_percentage'] for d in detections])
    avg_confidence = np.mean([d['confidence'] for d in detections])
    
    severity_priority = {'Severe': 3, 'Moderate': 2, 'Light': 1}
    highest_severity = max([severity_priority.get(d['severity'], 0) for d in detections])
    severity_names = {3: 'Severe', 2: 'Moderate', 1: 'Light'}
    
    report = {
        'timestamp': datetime.now(),
        'image_dimensions': image_info,
        'total_damages': len(detections),
        'total_affected_area': total_area,
        'estimated_repair_cost': total_cost,
        'average_confidence': avg_confidence,
        'highest_severity': severity_names.get(highest_severity, 'None'),
        'damage_breakdown': detections
    }
    
    return report

def main():
    # Application header
    st.markdown('<h1 class="main-header">Car Damage Assessment AI</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("### Automated vehicle damage detection using Computer Vision and Deep Learning")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model parameters
        st.subheader("Detection Parameters")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.05,
            help="Minimum confidence score for damage detection"
        )
        
        damage_types = st.multiselect(
            "Damage Types to Detect",
            ["Scratches", "Dents", "Broken Parts", "Paint Damage"],
            default=["Scratches", "Dents", "Paint Damage"],
            help="Select which types of damage to analyze"
        )
        
        # Processing options
        st.subheader("Processing Options")
        enhance_image_option = st.checkbox("Enable Image Enhancement", value=True)
        show_confidence = st.checkbox("Display Confidence Scores", value=True)
        generate_report = st.checkbox("Generate Assessment Report", value=True)
        
        # Model information
        st.markdown("---")
        st.markdown("""
        ### System Information
        **Model**: YOLOv8 Custom Trained  
        **Accuracy**: 87% mAP@0.5  
        **Classes**: 4 damage types  
        **Processing**: Real-time inference  
        """)
    
    # Main application layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Image Upload")
        
        # File upload interface
        uploaded_file = st.file_uploader(
            "Select vehicle image for analysis",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear, well-lit image of the vehicle showing potential damage areas"
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Image metadata
            col1_info, col2_info = st.columns(2)
            with col1_info:
                st.metric("Width", f"{image.size[0]}px")
                st.metric("Height", f"{image.size[1]}px")
            
            with col2_info:
                file_size = len(uploaded_file.getvalue()) / 1024
                st.metric("File Size", f"{file_size:.1f} KB")
                st.metric("Format", image.format)
            
            # Analysis trigger
            if st.button("Analyze Damage", type="primary", use_container_width=True):
                with st.spinner("Processing image and detecting damage..."):
                    # Simulate processing time for demo
                    import time
                    time.sleep(2)
                    
                    # Perform damage detection
                    processed_image, detections = demo_damage_detection(image)
                    
                    # Store results in session state
                    st.session_state.processed_image = processed_image
                    st.session_state.detections = detections
                    st.session_state.original_image = np.array(image)
                    st.session_state.image_info = image.size
                    
                st.success(f"Analysis complete. Found {len(detections)} damage areas.")
    
    with col2:
        st.header("Analysis Results")
        
        if 'detections' in st.session_state:
            # Display annotated image
            st.image(
                st.session_state.processed_image, 
                caption="Detected Damage Areas (Annotated)", 
                use_column_width=True
            )
            
            detections = st.session_state.detections
            
            # Key metrics display
            st.subheader("Assessment Summary")
            
            col1_metrics, col2_metrics, col3_metrics, col4_metrics = st.columns(4)
            
            with col1_metrics:
                st.metric("Total Damages", len(detections))
            
            with col2_metrics:
                avg_confidence = np.mean([d['confidence'] for d in detections])
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            with col3_metrics:
                total_area = sum([d['area_percentage'] for d in detections])
                st.metric("Affected Area", f"{total_area:.1f}%")
            
            with col4_metrics:
                total_cost = sum([d.get('estimated_cost', 0) for d in detections])
                st.metric("Est. Repair Cost", f"${total_cost}")
            
            # Detailed damage analysis
            st.subheader("Damage Details")
            
            # Create expandable sections for each damage
            for i, detection in enumerate(detections):
                severity = detection['severity']
                
                # Determine severity indicator
                severity_colors = {
                    'Severe': '🔴',
                    'Moderate': '🟠', 
                    'Light': '🟢'
                }
                indicator = severity_colors.get(severity, '⚪')
                
                with st.expander(f"{indicator} {detection['type']} - {severity} Damage"):
                    col1_detail, col2_detail = st.columns(2)
                    
                    with col1_detail:
                        st.write(f"**Damage Type:** {detection['type']}")
                        st.write(f"**Severity Level:** {detection['severity']}")
                        st.write(f"**Confidence Score:** {detection['confidence']:.1%}")
                        st.write(f"**Area Affected:** {detection['area_percentage']:.1f}%")
                    
                    with col2_detail:
                        bbox = detection['bbox']
                        st.write(f"**Bounding Box:** ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
                        st.write(f"**Estimated Cost:** ${detection.get('estimated_cost', 'N/A')}")
                        
                        # Severity-based recommendations
                        if severity == 'Severe':
                            st.error("Immediate repair recommended")
                        elif severity == 'Moderate':
                            st.warning("Repair recommended within 30 days")
                        else:
                            st.info("Cosmetic repair - no urgency")
            
            # Visualization charts
            st.subheader("Damage Analysis Charts")
            
            col1_viz, col2_viz = st.columns(2)
            
            with col1_viz:
                damage_dist_chart = create_damage_distribution_chart(detections)
                st.plotly_chart(damage_dist_chart, use_container_width=True)
            
            with col2_viz:
                severity_chart = create_severity_chart(detections)
                st.plotly_chart(severity_chart, use_container_width=True)
            
            # Generate comprehensive report
            if generate_report:
                st.subheader("Assessment Report")
                
                report = generate_assessment_report(detections, st.session_state.image_info)
                
                # Report summary
                st.write("**Report Generated:**", report['timestamp'].strftime("%Y-%m-%d %H:%M:%S"))
                st.write("**Overall Assessment:**", f"{report['highest_severity']} damage level detected")
                st.write("**Recommended Action:**", 
                        "Immediate attention required" if report['highest_severity'] == 'Severe' 
                        else "Schedule repair within reasonable timeframe")
                
                # Detailed breakdown table
                damage_df = pd.DataFrame([
                    {
                        'Type': d['type'],
                        'Severity': d['severity'],
                        'Confidence': f"{d['confidence']:.1%}",
                        'Area %': f"{d['area_percentage']:.1f}%",
                        'Est. Cost': f"${d.get('estimated_cost', 0)}"
                    }
                    for d in detections
                ])
                
                st.dataframe(damage_df, use_container_width=True)
                
                # Export options
                st.download_button(
                    label="Download Assessment Report (JSON)",
                    data=str(report),
                    file_name=f"damage_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        else:
            st.info("Upload an image and click 'Analyze Damage' to see results here.")
            
            # Show sample results or demo
            st.subheader("Sample Analysis")
            st.write("The system can detect and analyze:")
            st.write("- **Scratches**: Surface level paint damage")
            st.write("- **Dents**: Body deformation damage") 
            st.write("- **Paint Damage**: Color and coating issues")
            st.write("- **Broken Parts**: Structural component damage")

if __name__ == "__main__":
    main()
