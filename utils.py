import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from datetime import datetime
import json
import os

def enhance_image(image: Union[np.ndarray, Image.Image], 
                 brightness: float = 1.1,
                 contrast: float = 1.2,
                 sharpness: float = 1.1) -> np.ndarray:
    """
    Enhance image quality for better damage detection.
    
    Args:
        image: Input image (numpy array or PIL Image)
        brightness: Brightness enhancement factor (1.0 = no change)
        contrast: Contrast enhancement factor (1.0 = no change)
        sharpness: Sharpness enhancement factor (1.0 = no change)
        
    Returns:
        np.ndarray: Enhanced image as numpy array
    """
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image.copy()
        
        # Apply enhancements
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(contrast)
        
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(sharpness)
        
        # Convert back to numpy array
        return np.array(pil_image)
        
    except Exception as e:
        print(f"Error enhancing image: {str(e)}")
        # Return original image if enhancement fails
        return np.array(image) if isinstance(image, Image.Image) else image

def preprocess_for_detection(image: np.ndarray, 
                           target_size: Tuple[int, int] = (640, 640),
                           normalize: bool = True) -> np.ndarray:
    """
    Preprocess image for optimal model inference.
    
    Args:
        image: Input image as numpy array
        target_size: Target image dimensions (width, height)
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        np.ndarray: Preprocessed image
    """
    try:
        # Resize image while maintaining aspect ratio
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # Normalize if requested
        if normalize:
            padded = padded.astype(np.float32) / 255.0
        
        return padded
        
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return image

def calculate_damage_stats(detections: List[Dict]) -> Dict:
    """
    Calculate comprehensive statistics from damage detections.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Dict: Calculated statistics
    """
    if not detections:
        return {
            "total_damages": 0,
            "damage_types": {},
            "severity_distribution": {},
            "average_confidence": 0.0,
            "total_area_affected": 0.0,
            "total_estimated_cost": 0,
            "risk_assessment": "No damage detected"
        }
    
    # Initialize counters
    damage_types = {}
    severity_counts = {"light": 0, "moderate": 0, "severe": 0}
    total_area = 0.0
    total_cost = 0
    confidence_scores = []
    
    # Process each detection
    for detection in detections:
        # Count damage types
        damage_type = detection.get("type", "unknown")
        damage_types[damage_type] = damage_types.get(damage_type, 0) + 1
        
        # Count severity levels
        severity = detection.get("severity", "light")
        if severity in severity_counts:
            severity_counts[severity] += 1
        
        # Accumulate areas and costs
        total_area += detection.get("area_percentage", 0.0)
        total_cost += detection.get("estimated_cost", 0)
        confidence_scores.append(detection.get("confidence", 0.0))
    
    # Calculate averages
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    
    # Determine risk assessment
    if severity_counts["severe"] > 0:
        risk_level = "High Risk"
    elif severity_counts["moderate"] > 2:
        risk_level = "Moderate Risk"
    elif severity_counts["moderate"] > 0:
        risk_level = "Low-Moderate Risk"
    else:
        risk_level = "Low Risk"
    
    return {
        "total_damages": len(detections),
        "damage_types": damage_types,
        "severity_distribution": severity_counts,
        "average_confidence": round(avg_confidence, 3),
        "total_area_affected": round(total_area, 2),
        "total_estimated_cost": total_cost,
        "risk_assessment": risk_level,
        "most_common_damage": max(damage_types.items(), key=lambda x: x[1])[0] if damage_types else "None"
    }

def create_damage_report(detections: List[Dict], 
                        image_info: Dict,
                        include_recommendations: bool = True) -> Dict:
    """
    Generate a comprehensive damage assessment report.
    
    Args:
        detections: List of damage detections
        image_info: Image metadata dictionary
        include_recommendations: Whether to include repair recommendations
        
    Returns:
        Dict: Comprehensive damage report
    """
    stats = calculate_damage_stats(detections)
    
    report = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "report_version": "1.0",
            "assessment_type": "Automated Visual Inspection"
        },
        "image_information": image_info,
        "damage_summary": stats,
        "detailed_findings": detections,
        "cost_breakdown": _calculate_cost_breakdown(detections)
    }
    
    if include_recommendations:
        report["recommendations"] = _generate_recommendations(detections, stats)
    
    return report

def _calculate_cost_breakdown(detections: List[Dict]) -> Dict:
    """Calculate detailed cost breakdown by damage type and severity."""
    breakdown = {
        "by_damage_type": {},
        "by_severity": {"light": 0, "moderate": 0, "severe": 0},
        "total_cost": 0
    }
    
    for detection in detections:
        damage_type = detection.get("type", "unknown")
        severity = detection.get("severity", "light")
        cost = detection.get("estimated_cost", 0)
        
        # Breakdown by damage type
        if damage_type not in breakdown["by_damage_type"]:
            breakdown["by_damage_type"][damage_type] = 0
        breakdown["by_damage_type"][damage_type] += cost
        
        # Breakdown by severity
        if severity in breakdown["by_severity"]:
            breakdown["by_severity"][severity] += cost
        
        breakdown["total_cost"] += cost
    
    return breakdown

def _generate_recommendations(detections: List[Dict], stats: Dict) -> Dict:
    """Generate repair recommendations based on damage analysis."""
    recommendations = {
        "priority_level": "low",
        "immediate_actions": [],
        "repair_sequence": [],
        "preventive_measures": [],
        "estimated_timeline": "1-2 weeks"
    }
    
    # Determine priority based on severity
    if stats["severity_distribution"]["severe"] > 0:
        recommendations["priority_level"] = "high"
        recommendations["estimated_timeline"] = "Immediate attention required"
        recommendations["immediate_actions"].append(
            "Schedule immediate inspection with certified repair facility"
        )
    elif stats["severity_distribution"]["moderate"] > 1:
        recommendations["priority_level"] = "medium"
        recommendations["estimated_timeline"] = "1-2 weeks"
    
    # Generate repair sequence
    severe_damages = [d for d in detections if d.get("severity") == "severe"]
    moderate_damages = [d for d in detections if d.get("severity") == "moderate"]
    light_damages = [d for d in detections if d.get("severity") == "light"]
    
    sequence = []
    if severe_damages:
        sequence.extend([f"Repair {d['type']} damage ({d['location']})" for d in severe_damages])
    if moderate_damages:
        sequence.extend([f"Address {d['type']} damage ({d['location']})" for d in moderate_damages])
    if light_damages:
        sequence.extend([f"Cosmetic repair of {d['type']} ({d['location']})" for d in light_damages])
    
    recommendations["repair_sequence"] = sequence
    
    # Add preventive measures
    if stats["most_common_damage"] in ["scratch", "paint_damage"]:
        recommendations["preventive_measures"].append("Consider paint protection film")
        recommendations["preventive_measures"].append("Regular waxing and detailing")
    
    if "dent" in stats["damage_types"]:
        recommendations["preventive_measures"].append("Avoid tight parking spaces")
        recommendations["preventive_measures"].append("Use parking sensors or cameras")
    
    return recommendations

def validate_image(image_path: str) -> Tuple[bool, str]:
    """
    Validate if an image file is suitable for damage detection.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return False, "Image file not found"
        
        # Check file size (max 50MB)
        file_size = os.path.getsize(image_path)
        if file_size > 50 * 1024 * 1024:
            return False, "Image file too large (max 50MB)"
        
        # Try to open image
        with Image.open(image_path) as img:
            # Check image dimensions
            width, height = img.size
            if width < 100 or height < 100:
                return False, "Image resolution too low (minimum 100x100)"
            
            if width > 4000 or height > 4000:
                return False, "Image resolution too high (maximum 4000x4000)"
            
            # Check image format
            if img.format not in ['JPEG', 'PNG', 'JPG']:
                return False, "Unsupported image format (use JPEG or PNG)"
        
        return True, "Image is valid"
        
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

def convert_detections_to_dataframe(detections: List[Dict]) -> pd.DataFrame:
    """
    Convert detection results to pandas DataFrame for analysis.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        pd.DataFrame: Detection results as DataFrame
    """
    if not detections:
        return pd.DataFrame()
    
    # Flatten detection data
    flattened_data = []
    for i, detection in enumerate(detections):
        row = {
            "detection_id": i + 1,
            "damage_type": detection.get("type", "unknown"),
            "severity": detection.get("severity", "light"),
            "confidence": detection.get("confidence", 0.0),
            "area_percentage": detection.get("area_percentage", 0.0),
            "estimated_cost": detection.get("estimated_cost", 0),
            "location": detection.get("location", "unknown"),
            "bbox_x1": detection.get("bbox", [0, 0, 0, 0])[0],
            "bbox_y1": detection.get("bbox", [0, 0, 0, 0])[1],
            "bbox_x2": detection.get("bbox", [0, 0, 0, 0])[2],
            "bbox_y2": detection.get("bbox", [0, 0, 0, 0])[3]
        }
        flattened_data.append(row)
    
    return pd.DataFrame(flattened_data)

def export_results_to_json(results: Dict, output_path: str) -> bool:
    """
    Export detection results to JSON file.
    
    Args:
        results: Detection results dictionary
        output_path: Output file path
        
    Returns:
        bool: Success status
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error exporting results: {str(e)}")
        return False

def resize_image_for_display(image: np.ndarray, 
                           max_width: int = 800, 
                           max_height: int = 600) -> np.ndarray:
    """
    Resize image for display while maintaining aspect ratio.
    
    Args:
        image: Input image array
        max_width: Maximum display width
        max_height: Maximum display height
        
    Returns:
        np.ndarray: Resized image
    """
    h, w = image.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        float: IoU value between 0 and 1
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # Calculate IoU
    if union == 0:
        return 0.0
    
    return intersection / union

def filter_overlapping_detections(detections: List[Dict], 
                                iou_threshold: float = 0.5) -> List[Dict]:
    """
    Filter out overlapping detections using Non-Maximum Suppression.
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for overlap filtering
        
    Returns:
        List[Dict]: Filtered detections
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence score (descending)
    sorted_detections = sorted(detections, 
                             key=lambda x: x.get("confidence", 0), 
                             reverse=True)
    
    filtered = []
    
    for detection in sorted_detections:
        # Check if this detection overlaps significantly with any kept detection
        keep = True
        for kept_detection in filtered:
            iou = calculate_iou(detection["bbox"], kept_detection["bbox"])
            if iou > iou_threshold:
                keep = False
                break
        
        if keep:
            filtered.append(detection)
    
    return filtered

def create_confidence_heatmap(detections: List[Dict], 
                            image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a confidence heatmap showing detection reliability across the image.
    
    Args:
        detections: List of detection dictionaries
        image_shape: Image dimensions (height, width)
        
    Returns:
        np.ndarray: Confidence heatmap
    """
    height, width = image_shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    for detection in detections:
        bbox = detection["bbox"]
        confidence = detection.get("confidence", 0.0)
        
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        
        # Add confidence to the bounding box area
        heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], confidence)
    
    return heatmap

def get_image_metadata(image: Union[str, np.ndarray, Image.Image]) -> Dict:
    """
    Extract metadata from an image.
    
    Args:
        image: Input image (path, numpy array, or PIL Image)
        
    Returns:
        Dict: Image metadata
    """
    metadata = {
        "width": 0,
        "height": 0,
        "channels": 0,
        "format": "unknown",
        "size_bytes": 0,
        "aspect_ratio": 0.0
    }
    
    try:
        if isinstance(image, str):
            # Load from file path
            with Image.open(image) as img:
                metadata["width"], metadata["height"] = img.size
                metadata["format"] = img.format
                metadata["size_bytes"] = os.path.getsize(image)
        elif isinstance(image, Image.Image):
            metadata["width"], metadata["height"] = image.size
            metadata["format"] = image.format
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                metadata["height"], metadata["width"], metadata["channels"] = image.shape
            else:
                metadata["height"], metadata["width"] = image.shape
                metadata["channels"] = 1
        
        # Calculate aspect ratio
        if metadata["height"] > 0:
            metadata["aspect_ratio"] = metadata["width"] / metadata["height"]
        
    except Exception as e:
        print(f"Error extracting image metadata: {str(e)}")
    
    return metadata

def main():
    """
    Example usage of utility functions.
    """
    print("Car Damage Assessment Utilities")
    print("Available functions:")
    print("- enhance_image: Improve image quality")
    print("- calculate_damage_stats: Generate statistics")
    print("- create_damage_report: Generate comprehensive reports")
    print("- validate_image: Check image suitability")
    print("- filter_overlapping_detections: Remove duplicate detections")

if __name__ == "__main__":
    main()
