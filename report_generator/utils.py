"""
Utility functions for Medical Report Generator
============================================

This module contains utility functions for data validation, formatting,
and JSON processing used by the medical report generator.

Author: Brain Tumor AI Team
Date: 2024
"""

import json
import re
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

def validate_tumor_data(tumor_name: str, confidence_score: str) -> bool:
    """
    Validate tumor diagnosis data
    
    Args:
        tumor_name (str): Name of the tumor
        confidence_score (str): Confidence score as percentage
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    try:
        # Validate tumor name
        if not tumor_name or not isinstance(tumor_name, str):
            logging.error("Tumor name must be a non-empty string")
            return False
        
        if len(tumor_name.strip()) == 0:
            logging.error("Tumor name cannot be empty")
            return False
        
        # Validate confidence score
        if not confidence_score or not isinstance(confidence_score, str):
            logging.error("Confidence score must be a non-empty string")
            return False
        
        # Check if confidence score contains percentage
        if '%' not in confidence_score:
            logging.error("Confidence score must include percentage symbol (%)")
            return False
        
        # Extract numeric value from confidence score
        numeric_value = re.search(r'(\d+(?:\.\d+)?)', confidence_score)
        if not numeric_value:
            logging.error("Invalid confidence score format")
            return False
        
        confidence_value = float(numeric_value.group(1))
        if not (0 <= confidence_value <= 100):
            logging.error("Confidence score must be between 0 and 100")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating tumor data: {str(e)}")
        return False

def format_confidence_score(confidence_score: str) -> str:
    """
    Format confidence score to ensure proper percentage format
    
    Args:
        confidence_score (str): Raw confidence score
        
    Returns:
        str: Formatted confidence score
    """
    try:
        # Extract numeric value
        numeric_value = re.search(r'(\d+(?:\.\d+)?)', confidence_score)
        if numeric_value:
            value = float(numeric_value.group(1))
            return f"{value:.1f}%"
        else:
            logging.warning("Could not extract numeric value from confidence score")
            return confidence_score
            
    except Exception as e:
        logging.error(f"Error formatting confidence score: {str(e)}")
        return confidence_score

def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON data from AI response text
    
    Args:
        response_text (str): Raw response text from AI
        
    Returns:
        Dict[str, Any]: Parsed JSON data or None if failed
    """
    try:
        if not response_text:
            logging.error("Empty response text provided")
            return None
        
        # Find JSON-like content in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            logging.warning("No JSON brackets found in response")
            return None
        
        if start_idx >= end_idx:
            logging.warning("Invalid JSON structure in response")
            return None
        
        # Extract JSON string
        json_str = response_text[start_idx:end_idx + 1]
        
        # Try to parse JSON
        parsed_json = json.loads(json_str)
        
        logging.info("Successfully extracted JSON from response")
        return parsed_json
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error extracting JSON: {str(e)}")
        return None

def sanitize_tumor_name(tumor_name: str) -> str:
    """
    Sanitize tumor name for consistent processing
    
    Args:
        tumor_name (str): Raw tumor name
        
    Returns:
        str: Sanitized tumor name
    """
    try:
        # Remove extra whitespace and normalize
        sanitized = tumor_name.strip()
        
        # Capitalize first letter of each word
        sanitized = ' '.join(word.capitalize() for word in sanitized.split())
        
        # Remove any special characters that might cause issues
        sanitized = re.sub(r'[^\w\s\-\.]', '', sanitized)
        
        return sanitized
        
    except Exception as e:
        logging.error(f"Error sanitizing tumor name: {str(e)}")
        return tumor_name

def validate_medical_report(report_data: Dict[str, Any]) -> bool:
    """
    Validate medical report data structure
    
    Args:
        report_data (Dict[str, Any]): Medical report data
        
    Returns:
        bool: True if report is valid, False otherwise
    """
    try:
        required_fields = [
            'patient_id', 'diagnosis_date', 'tumor_diagnosis',
            'ai_analysis', 'recommendations', 'follow_up_instructions'
        ]
        
        for field in required_fields:
            if field not in report_data:
                logging.error(f"Missing required field: {field}")
                return False
        
        # Validate tumor diagnosis
        tumor_diagnosis = report_data.get('tumor_diagnosis', {})
        if not isinstance(tumor_diagnosis, dict):
            logging.error("tumor_diagnosis must be a dictionary")
            return False
        
        required_tumor_fields = ['tumor_name', 'confidence_score']
        for field in required_tumor_fields:
            if field not in tumor_diagnosis:
                logging.error(f"Missing required tumor field: {field}")
                return False
        
        # Validate lists
        if not isinstance(report_data.get('recommendations', []), list):
            logging.error("recommendations must be a list")
            return False
        
        if not isinstance(report_data.get('follow_up_instructions', []), list):
            logging.error("follow_up_instructions must be a list")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating medical report: {str(e)}")
        return False

def generate_report_summary(report_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary of the medical report
    
    Args:
        report_data (Dict[str, Any]): Complete medical report
        
    Returns:
        Dict[str, Any]: Report summary
    """
    try:
        tumor_diagnosis = report_data.get('tumor_diagnosis', {})
        
        summary = {
            'report_id': report_data.get('report_id', 'N/A'),
            'patient_id': report_data.get('patient_id', 'N/A'),
            'diagnosis_date': report_data.get('diagnosis_date', 'N/A'),
            'tumor_name': tumor_diagnosis.get('tumor_name', 'N/A'),
            'confidence_score': tumor_diagnosis.get('confidence_score', 'N/A'),
            'severity_level': tumor_diagnosis.get('severity_level', 'N/A'),
            'urgency_level': tumor_diagnosis.get('urgency_level', 'N/A'),
            'consult_doctor': tumor_diagnosis.get('consult_doctor', 'N/A'),
            'recommendations_count': len(report_data.get('recommendations', [])),
            'follow_up_count': len(report_data.get('follow_up_instructions', []))
        }
        
        return summary
        
    except Exception as e:
        logging.error(f"Error generating report summary: {str(e)}")
        return {}

def format_report_for_display(report_data: Dict[str, Any]) -> str:
    """
    Format medical report for human-readable display
    
    Args:
        report_data (Dict[str, Any]): Medical report data
        
    Returns:
        str: Formatted report string
    """
    try:
        tumor_diagnosis = report_data.get('tumor_diagnosis', {})
        
        formatted_report = f"""
MEDICAL DIAGNOSIS REPORT
========================

Report ID: {report_data.get('report_id', 'N/A')}
Patient ID: {report_data.get('patient_id', 'N/A')}
Diagnosis Date: {report_data.get('diagnosis_date', 'N/A')}

TUMOR DIAGNOSIS
---------------
Tumor Name: {tumor_diagnosis.get('tumor_name', 'N/A')}
Confidence Score: {tumor_diagnosis.get('confidence_score', 'N/A')}
Tumor Type: {tumor_diagnosis.get('tumor_type', 'N/A')}
Severity Level: {tumor_diagnosis.get('severity_level', 'N/A')}
Urgency Level: {tumor_diagnosis.get('urgency_level', 'N/A')}

CONDITION
---------
{tumor_diagnosis.get('condition', 'N/A')}

TREATMENT OPTIONS
-----------------
Surgery: {tumor_diagnosis.get('treatment', {}).get('surgery', 'N/A')}
Radiation: {tumor_diagnosis.get('treatment', {}).get('radiation', 'N/A')}
Chemotherapy: {tumor_diagnosis.get('treatment', {}).get('chemotherapy', 'N/A')}

PRECAUTIONS
-----------
"""
        
        precautions = tumor_diagnosis.get('precautions', [])
        for i, precaution in enumerate(precautions, 1):
            formatted_report += f"{i}. {precaution}\n"
        
        formatted_report += f"""
CONSULT DOCTOR
--------------
{tumor_diagnosis.get('consult_doctor', 'N/A')}

EXPLANATION
-----------
{tumor_diagnosis.get('explanation', 'N/A')}

RECOMMENDATIONS
---------------
"""
        
        recommendations = report_data.get('recommendations', [])
        for i, recommendation in enumerate(recommendations, 1):
            formatted_report += f"{i}. {recommendation}\n"
        
        formatted_report += f"""
FOLLOW-UP INSTRUCTIONS
----------------------
"""
        
        follow_up = report_data.get('follow_up_instructions', [])
        for i, instruction in enumerate(follow_up, 1):
            formatted_report += f"{i}. {instruction}\n"
        
        return formatted_report
        
    except Exception as e:
        logging.error(f"Error formatting report for display: {str(e)}")
        return "Error formatting report"

def calculate_risk_score(tumor_diagnosis: Dict[str, Any]) -> float:
    """
    Calculate risk score based on tumor diagnosis
    
    Args:
        tumor_diagnosis (Dict[str, Any]): Tumor diagnosis data
        
    Returns:
        float: Risk score between 0 and 1
    """
    try:
        risk_score = 0.0
        
        # Base risk from confidence score
        confidence_str = tumor_diagnosis.get('confidence_score', '0%')
        confidence_match = re.search(r'(\d+(?:\.\d+)?)', confidence_str)
        if confidence_match:
            confidence = float(confidence_match.group(1)) / 100.0
            risk_score += confidence * 0.3  # 30% weight
        
        # Severity level risk
        severity = tumor_diagnosis.get('severity_level', '').lower()
        if severity == 'high':
            risk_score += 0.4
        elif severity == 'medium':
            risk_score += 0.2
        # Low severity adds 0
        
        # Urgency level risk
        urgency = tumor_diagnosis.get('urgency_level', '').lower()
        if urgency == 'immediate':
            risk_score += 0.3
        elif urgency == 'urgent':
            risk_score += 0.15
        # Regular urgency adds 0
        
        # Cap at 1.0
        return min(risk_score, 1.0)
        
    except Exception as e:
        logging.error(f"Error calculating risk score: {str(e)}")
        return 0.5  # Default moderate risk

def get_tumor_category(tumor_name: str) -> str:
    """
    Categorize tumor based on name
    
    Args:
        tumor_name (str): Name of the tumor
        
    Returns:
        str: Tumor category
    """
    try:
        tumor_name_lower = tumor_name.lower()
        
        # Define tumor categories
        categories = {
            'glioblastoma': 'High-Grade Glioma',
            'astrocytoma': 'Glioma',
            'meningioma': 'Meningeal Tumor',
            'pituitary': 'Pituitary Tumor',
            'schwannoma': 'Nerve Sheath Tumor',
            'ependymoma': 'Ependymal Tumor',
            'medulloblastoma': 'Embryonal Tumor',
            'oligodendroglioma': 'Glioma',
            'craniopharyngioma': 'Sellar Region Tumor',
            'hemangioblastoma': 'Vascular Tumor'
        }
        
        for keyword, category in categories.items():
            if keyword in tumor_name_lower:
                return category
        
        return 'Other Brain Tumor'
        
    except Exception as e:
        logging.error(f"Error categorizing tumor: {str(e)}")
        return 'Unknown'

def validate_api_response(response: Dict[str, Any]) -> bool:
    """
    Validate API response structure
    
    Args:
        response (Dict[str, Any]): API response
        
    Returns:
        bool: True if response is valid, False otherwise
    """
    try:
        if not isinstance(response, dict):
            return False
        
        # Check for required fields in typical AI response
        if '1' in response:
            tumor_data = response['1']
            required_fields = ['tumor_type', 'confidence_score', 'condition']
            
            for field in required_fields:
                if field not in tumor_data:
                    return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating API response: {str(e)}")
        return False 