"""
Logging configuration for Medical Report Generator
===============================================

This module provides comprehensive logging functionality for the medical
report generator, including different log levels, file and console output,
and structured logging for medical applications.

Author: Brain Tumor AI Team
Date: 2024
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

# Default logging configuration
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_FILE = 'medical_reports.log'
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT = 5

class MedicalReportLogger:
    """
    Custom logger class for medical report generation
    """
    
    def __init__(self, name: str, log_level: int = DEFAULT_LOG_LEVEL):
        """
        Initialize the medical report logger
        
        Args:
            name (str): Logger name
            log_level (int): Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / DEFAULT_LOG_FILE,
            maxBytes=DEFAULT_MAX_BYTES,
            backupCount=DEFAULT_BACKUP_COUNT
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)

def setup_logger(name: str, 
                log_level: int = DEFAULT_LOG_LEVEL,
                log_file: Optional[str] = None,
                console_output: bool = True,
                file_output: bool = True) -> logging.Logger:
    """
    Setup and configure logger for medical report generation
    
    Args:
        name (str): Logger name
        log_level (int): Logging level
        log_file (str, optional): Custom log file path
        console_output (bool): Enable console output
        file_output (bool): Enable file output
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers to prevent duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Use custom log file or default
        log_file_path = log_file or log_dir / DEFAULT_LOG_FILE
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=DEFAULT_MAX_BYTES,
            backupCount=DEFAULT_BACKUP_COUNT
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger

def setup_medical_logger(name: str = "medical_report_generator") -> logging.Logger:
    """
    Setup specialized logger for medical applications
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Medical logger with specialized configuration
    """
    logger = setup_logger(
        name=name,
        log_level=logging.INFO,
        console_output=True,
        file_output=True
    )
    
    # Log initial setup
    logger.info("Medical report logger initialized")
    logger.info(f"Log level: {logging.getLevelName(logger.level)}")
    
    return logger

def log_medical_event(logger: logging.Logger, 
                     event_type: str, 
                     patient_id: Optional[str] = None,
                     tumor_name: Optional[str] = None,
                     confidence_score: Optional[str] = None,
                     additional_data: Optional[Dict[str, Any]] = None):
    """
    Log medical events with structured data
    
    Args:
        logger (logging.Logger): Logger instance
        event_type (str): Type of medical event
        patient_id (str, optional): Patient identifier
        tumor_name (str, optional): Tumor name
        confidence_score (str, optional): Confidence score
        additional_data (Dict[str, Any], optional): Additional event data
    """
    event_data = {
        'event_type': event_type,
        'timestamp': datetime.now().isoformat(),
        'patient_id': patient_id,
        'tumor_name': tumor_name,
        'confidence_score': confidence_score
    }
    
    if additional_data:
        event_data.update(additional_data)
    
    logger.info(f"Medical Event: {event_type} - {event_data}")

def log_diagnosis_report(logger: logging.Logger,
                        tumor_name: str,
                        confidence_score: str,
                        report_data: Dict[str, Any],
                        success: bool = True):
    """
    Log diagnosis report generation
    
    Args:
        logger (logging.Logger): Logger instance
        tumor_name (str): Name of the tumor
        confidence_score (str): Confidence score
        report_data (Dict[str, Any]): Generated report data
        success (bool): Whether report generation was successful
    """
    event_type = "diagnosis_report_success" if success else "diagnosis_report_failure"
    
    additional_data = {
        'report_id': report_data.get('report_id', 'N/A'),
        'severity_level': report_data.get('tumor_diagnosis', {}).get('severity_level', 'N/A'),
        'urgency_level': report_data.get('tumor_diagnosis', {}).get('urgency_level', 'N/A')
    }
    
    log_medical_event(
        logger=logger,
        event_type=event_type,
        tumor_name=tumor_name,
        confidence_score=confidence_score,
        additional_data=additional_data
    )

def log_api_call(logger: logging.Logger,
                api_name: str,
                request_data: Dict[str, Any],
                response_success: bool,
                response_time: Optional[float] = None,
                error_message: Optional[str] = None):
    """
    Log API calls for monitoring and debugging
    
    Args:
        logger (logging.Logger): Logger instance
        api_name (str): Name of the API being called
        request_data (Dict[str, Any]): Request data
        response_success (bool): Whether API call was successful
        response_time (float, optional): Response time in seconds
        error_message (str, optional): Error message if failed
    """
    event_type = "api_call_success" if response_success else "api_call_failure"
    
    additional_data = {
        'api_name': api_name,
        'request_data': request_data,
        'response_time': response_time,
        'error_message': error_message
    }
    
    log_medical_event(
        logger=logger,
        event_type=event_type,
        additional_data=additional_data
    )

def log_performance_metrics(logger: logging.Logger,
                          operation: str,
                          duration: float,
                          success: bool = True,
                          additional_metrics: Optional[Dict[str, Any]] = None):
    """
    Log performance metrics for monitoring
    
    Args:
        logger (logging.Logger): Logger instance
        operation (str): Operation being measured
        duration (float): Duration in seconds
        success (bool): Whether operation was successful
        additional_metrics (Dict[str, Any], optional): Additional metrics
    """
    event_type = "performance_metrics"
    
    metrics_data = {
        'operation': operation,
        'duration_seconds': duration,
        'success': success
    }
    
    if additional_metrics:
        metrics_data.update(additional_metrics)
    
    log_medical_event(
        logger=logger,
        event_type=event_type,
        additional_data=metrics_data
    )

def get_logger_config() -> Dict[str, Any]:
    """
    Get default logger configuration
    
    Returns:
        Dict[str, Any]: Logger configuration
    """
    return {
        'log_level': DEFAULT_LOG_LEVEL,
        'log_format': DEFAULT_LOG_FORMAT,
        'log_file': DEFAULT_LOG_FILE,
        'max_bytes': DEFAULT_MAX_BYTES,
        'backup_count': DEFAULT_BACKUP_COUNT,
        'console_output': True,
        'file_output': True
    }

def create_log_summary(log_file_path: str) -> Dict[str, Any]:
    """
    Create a summary of log entries
    
    Args:
        log_file_path (str): Path to log file
        
    Returns:
        Dict[str, Any]: Log summary statistics
    """
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
        
        summary = {
            'total_entries': len(lines),
            'error_count': 0,
            'warning_count': 0,
            'info_count': 0,
            'debug_count': 0,
            'medical_events': 0,
            'api_calls': 0,
            'performance_metrics': 0
        }
        
        for line in lines:
            if 'ERROR' in line:
                summary['error_count'] += 1
            elif 'WARNING' in line:
                summary['warning_count'] += 1
            elif 'INFO' in line:
                summary['info_count'] += 1
            elif 'DEBUG' in line:
                summary['debug_count'] += 1
            
            # Count specific event types
            if 'Medical Event:' in line:
                summary['medical_events'] += 1
            elif 'api_call' in line:
                summary['api_calls'] += 1
            elif 'performance_metrics' in line:
                summary['performance_metrics'] += 1
        
        return summary
        
    except Exception as e:
        return {'error': f"Could not read log file: {str(e)}"}

# Convenience function for quick logger setup
def get_medical_logger(name: str = "medical_report_generator") -> logging.Logger:
    """
    Get a configured medical logger
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Configured medical logger
    """
    return setup_medical_logger(name) 