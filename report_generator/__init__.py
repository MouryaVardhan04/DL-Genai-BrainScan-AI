"""
Medical Report Generator Package
==============================

A comprehensive package for generating medical reports based on brain tumor
diagnosis using AI-powered analysis.

Author: Brain Tumor AI Team
Date: 2024
"""

__version__ = "1.0.0"
__author__ = "Brain Tumor AI Team"
__email__ = "team@braintumor.ai"

# Import main classes and functions for easy access
from .report_generator import (
    MedicalReportGenerator,
    TumorDiagnosis,
    MedicalReport,
    generate_quick_report,
    create_full_report
)

from .utils import (
    validate_tumor_data,
    format_confidence_score,
    extract_json_from_response,
    sanitize_tumor_name,
    validate_medical_report,
    generate_report_summary,
    format_report_for_display,
    calculate_risk_score,
    get_tumor_category,
    validate_api_response
)

from .logger import (
    setup_logger,
    setup_medical_logger,
    MedicalReportLogger,
    log_medical_event,
    log_diagnosis_report,
    log_api_call,
    log_performance_metrics,
    get_medical_logger
)

# Package-level imports
__all__ = [
    # Main classes
    'MedicalReportGenerator',
    'TumorDiagnosis', 
    'MedicalReport',
    
    # Convenience functions
    'generate_quick_report',
    'create_full_report',
    
    # Utility functions
    'validate_tumor_data',
    'format_confidence_score',
    'extract_json_from_response',
    'sanitize_tumor_name',
    'validate_medical_report',
    'generate_report_summary',
    'format_report_for_display',
    'calculate_risk_score',
    'get_tumor_category',
    'validate_api_response',
    
    # Logging functions
    'setup_logger',
    'setup_medical_logger',
    'MedicalReportLogger',
    'log_medical_event',
    'log_diagnosis_report',
    'log_api_call',
    'log_performance_metrics',
    'get_medical_logger'
]

def get_version():
    """Get the package version"""
    return __version__

def get_author():
    """Get the package author"""
    return __author__

def get_package_info():
    """Get complete package information"""
    return {
        'name': 'medical_report_generator',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'description': 'Medical report generator for brain tumor diagnosis'
    } 