"""
Medical Report Generator for Brain Tumor Diagnosis
================================================

This module provides comprehensive functionality for generating medical reports
based on brain tumor diagnosis results using AI-powered analysis.

Author: Brain Tumor AI Team
Date: 2024
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

from .utils import validate_tumor_data, format_confidence_score, extract_json_from_response
from .logger import setup_logger

# Load environment variables
load_dotenv()

@dataclass
class TumorDiagnosis:
    """Data class for tumor diagnosis information"""
    tumor_name: str
    confidence_score: str
    tumor_type: Optional[str] = None
    condition: Optional[str] = None
    treatment: Optional[Dict[str, str]] = None
    precautions: Optional[List[str]] = None
    consult_doctor: Optional[str] = None
    explanation: Optional[str] = None
    severity_level: Optional[str] = None
    urgency_level: Optional[str] = None

@dataclass
class MedicalReport:
    """Data class for complete medical report"""
    patient_id: str
    diagnosis_date: str
    tumor_diagnosis: TumorDiagnosis
    ai_analysis: Dict[str, Any]
    recommendations: List[str]
    follow_up_instructions: List[str]
    report_id: Optional[str] = None

class MedicalReportGenerator:
    """
    Main class for generating comprehensive medical reports for brain tumor diagnosis
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the Medical Report Generator
        
        Args:
            api_key (str, optional): OpenAI API key. If None, loads from environment
            model_name (str): OpenAI model to use for analysis
        """
        self.logger = setup_logger(__name__)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model_name = model_name
        self.llm = ChatOpenAI(
            openai_api_key=self.api_key,
            model_name=model_name,
            temperature=0.3
        )
        
        self.response_template = {
            "1": {
                "tumor_type": "Tumor name here",
                "confidence_score": "Confidence as a percentage",
                "condition": "Brief condition description",
                "treatment": {
                    "surgery": "Surgical approach if any",
                    "radiation": "Radiation approach if any",
                    "chemotherapy": "Chemotherapy protocol if any"
                },
                "precautions": [
                    "Precaution 1",
                    "Precaution 2",
                    "Precaution 3"
                ],
                "consult_doctor": "Doctor specialty to consult",
                "explanation": "Short explanation of tumor type, urgency, treatment rationale",
                "severity_level": "Low/Medium/High",
                "urgency_level": "Immediate/Urgent/Regular"
            }
        }
        
        self.prompt_template = PromptTemplate(
            input_variables=["tumor_name", "confidence_score", "response_json"],
            template=self._get_prompt_template()
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            output_key="report",
            verbose=False
        )
        
        self.logger.info(f"MedicalReportGenerator initialized with model: {model_name}")
    
    def _get_prompt_template(self) -> str:
        """Get the prompt template for medical analysis"""
        return """
Tumor: {tumor_name}
Confidence Score: {confidence_score}

You are a medical diagnosis assistant specializing in brain tumor analysis. Based on the tumor name and confidence score provided above, generate:

1. A comprehensive **description** of the tumor (AI-generated clinical summary).
2. A structured JSON response that includes:
   - The type of tumor with medical accuracy
   - The confidence score (as a percentage string, e.g., "85%")
   - A detailed description of the tumor's condition and characteristics
   - Recommended treatment options (including surgery, radiation, and chemotherapy if applicable)
   - Specific precautionary steps the patient should follow
   - The type of doctor to consult (specialist recommendation)
   - A comprehensive explanation of the tumor's severity, typical progression, and urgency
   - Severity level assessment (Low/Medium/High)
   - Urgency level for medical attention (Immediate/Urgent/Regular)

Format your response exactly like the RESPONSE_JSON below.

### RESPONSE_JSON
{response_json}
"""
    
    def generate_diagnosis_report(self, tumor_name: str, confidence_score: str) -> Optional[Dict[str, Any]]:
        """
        Generate a comprehensive diagnosis report for a brain tumor
        
        Args:
            tumor_name (str): Name of the detected tumor
            confidence_score (str): Confidence score as percentage
            
        Returns:
            Dict[str, Any]: Structured medical report or None if failed
        """
        try:
            self.logger.info(f"Generating diagnosis report for {tumor_name} (Confidence: {confidence_score})")
            
            # Validate input data
            if not validate_tumor_data(tumor_name, confidence_score):
                self.logger.error("Invalid tumor data provided")
                return None
            
            # Format confidence score
            formatted_confidence = format_confidence_score(confidence_score)
            
            # Generate AI analysis
            response = self.chain.run({
                "tumor_name": tumor_name,
                "confidence_score": formatted_confidence,
                "response_json": json.dumps(self.response_template, indent=2)
            })
            
            self.logger.info("AI analysis completed successfully")
            
            # Extract JSON from response
            json_data = extract_json_from_response(response)
            
            if json_data:
                self.logger.info("Successfully extracted structured data from AI response")
                return json_data
            else:
                self.logger.warning("Could not extract structured JSON, returning raw response")
                return {"raw_response": response}
                
        except Exception as e:
            self.logger.error(f"Error generating diagnosis report: {str(e)}")
            return None
    
    def create_comprehensive_report(self, 
                                  patient_id: str,
                                  tumor_name: str, 
                                  confidence_score: str,
                                  additional_notes: Optional[str] = None) -> Optional[MedicalReport]:
        """
        Create a comprehensive medical report with all necessary information
        
        Args:
            patient_id (str): Unique patient identifier
            tumor_name (str): Name of the detected tumor
            confidence_score (str): Confidence score as percentage
            additional_notes (str, optional): Additional medical notes
            
        Returns:
            MedicalReport: Complete medical report object
        """
        try:
            self.logger.info(f"Creating comprehensive report for patient {patient_id}")
            
            # Generate AI diagnosis
            diagnosis_data = self.generate_diagnosis_report(tumor_name, confidence_score)
            
            if not diagnosis_data:
                self.logger.error("Failed to generate diagnosis data")
                return None
            
            # Extract diagnosis information
            diagnosis_info = diagnosis_data.get("1", {})
            
            # Create tumor diagnosis object
            tumor_diagnosis = TumorDiagnosis(
                tumor_name=tumor_name,
                confidence_score=confidence_score,
                tumor_type=diagnosis_info.get("tumor_type"),
                condition=diagnosis_info.get("condition"),
                treatment=diagnosis_info.get("treatment"),
                precautions=diagnosis_info.get("precautions"),
                consult_doctor=diagnosis_info.get("consult_doctor"),
                explanation=diagnosis_info.get("explanation"),
                severity_level=diagnosis_info.get("severity_level"),
                urgency_level=diagnosis_info.get("urgency_level")
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(tumor_diagnosis)
            
            # Generate follow-up instructions
            follow_up = self._generate_follow_up_instructions(tumor_diagnosis)
            
            # Create comprehensive report
            report = MedicalReport(
                patient_id=patient_id,
                diagnosis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                tumor_diagnosis=tumor_diagnosis,
                ai_analysis=diagnosis_data,
                recommendations=recommendations,
                follow_up_instructions=follow_up,
                report_id=f"RPT_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            self.logger.info(f"Comprehensive report created successfully: {report.report_id}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive report: {str(e)}")
            return None
    
    def _generate_recommendations(self, tumor_diagnosis: TumorDiagnosis) -> List[str]:
        """Generate specific recommendations based on tumor diagnosis"""
        recommendations = []
        
        # Base recommendations
        recommendations.append("Schedule immediate consultation with the recommended specialist")
        recommendations.append("Bring all previous medical records and imaging studies")
        recommendations.append("Prepare a list of current symptoms and their progression")
        
        # Severity-based recommendations
        if tumor_diagnosis.severity_level == "High":
            recommendations.append("Seek emergency medical attention if symptoms worsen")
            recommendations.append("Consider second opinion from a specialized center")
        elif tumor_diagnosis.severity_level == "Medium":
            recommendations.append("Monitor symptoms closely and report any changes")
            recommendations.append("Schedule follow-up within recommended timeframe")
        
        # Treatment-specific recommendations
        if tumor_diagnosis.treatment:
            if tumor_diagnosis.treatment.get("surgery"):
                recommendations.append("Discuss surgical options with neurosurgeon")
            if tumor_diagnosis.treatment.get("radiation"):
                recommendations.append("Consult with radiation oncologist")
            if tumor_diagnosis.treatment.get("chemotherapy"):
                recommendations.append("Consult with medical oncologist")
        
        return recommendations
    
    def _generate_follow_up_instructions(self, tumor_diagnosis: TumorDiagnosis) -> List[str]:
        """Generate follow-up instructions based on tumor diagnosis"""
        instructions = []
        
        # Urgency-based instructions
        if tumor_diagnosis.urgency_level == "Immediate":
            instructions.append("Seek medical attention within 24 hours")
            instructions.append("Do not delay treatment initiation")
        elif tumor_diagnosis.urgency_level == "Urgent":
            instructions.append("Schedule appointment within 1 week")
            instructions.append("Monitor for symptom progression")
        else:
            instructions.append("Schedule routine follow-up within 2-4 weeks")
            instructions.append("Continue regular monitoring")
        
        # General follow-up instructions
        instructions.append("Keep detailed symptom diary")
        instructions.append("Attend all scheduled appointments")
        instructions.append("Follow prescribed treatment plan strictly")
        
        return instructions
    
    def save_report(self, report: MedicalReport, filename: Optional[str] = None) -> bool:
        """
        Save medical report to JSON file
        
        Args:
            report (MedicalReport): Medical report to save
            filename (str, optional): Custom filename. If None, uses default naming
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            if not filename:
                filename = f"medical_report_{report.report_id}.json"
            
            # Convert report to dictionary
            report_dict = asdict(report)
            
            with open(filename, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            self.logger.info(f"Report saved successfully: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving report: {str(e)}")
            return False
    
    def load_report(self, filename: str) -> Optional[MedicalReport]:
        """
        Load medical report from JSON file
        
        Args:
            filename (str): Path to the JSON file
            
        Returns:
            MedicalReport: Loaded medical report or None if failed
        """
        try:
            with open(filename, 'r') as f:
                report_dict = json.load(f)
            
            # Reconstruct objects from dictionary
            tumor_diagnosis = TumorDiagnosis(**report_dict["tumor_diagnosis"])
            report = MedicalReport(
                patient_id=report_dict["patient_id"],
                diagnosis_date=report_dict["diagnosis_date"],
                tumor_diagnosis=tumor_diagnosis,
                ai_analysis=report_dict["ai_analysis"],
                recommendations=report_dict["recommendations"],
                follow_up_instructions=report_dict["follow_up_instructions"],
                report_id=report_dict["report_id"]
            )
            
            self.logger.info(f"Report loaded successfully: {filename}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error loading report: {str(e)}")
            return None

# Convenience functions for easy usage
def generate_quick_report(tumor_name: str, confidence_score: str) -> Optional[Dict[str, Any]]:
    """
    Quick function to generate a basic diagnosis report
    
    Args:
        tumor_name (str): Name of the tumor
        confidence_score (str): Confidence score
        
    Returns:
        Dict[str, Any]: Basic diagnosis report
    """
    try:
        generator = MedicalReportGenerator()
        return generator.generate_diagnosis_report(tumor_name, confidence_score)
    except Exception as e:
        logging.error(f"Error in quick report generation: {str(e)}")
        return None

def create_full_report(patient_id: str, tumor_name: str, confidence_score: str) -> Optional[MedicalReport]:
    """
    Quick function to create a full medical report
    
    Args:
        patient_id (str): Patient identifier
        tumor_name (str): Name of the tumor
        confidence_score (str): Confidence score
        
    Returns:
        MedicalReport: Complete medical report
    """
    try:
        generator = MedicalReportGenerator()
        return generator.create_comprehensive_report(patient_id, tumor_name, confidence_score)
    except Exception as e:
        logging.error(f"Error in full report creation: {str(e)}")
        return None 