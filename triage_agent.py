"""
Medical Triage Agent for Symptom Analysis
This module implements a decision tree-based triage agent to classify
symptom severity and recommend appropriate care pathways.
"""

from enum import Enum
from typing import List, Dict, Any
import re

class TriageSeverity(Enum):
    """Enumeration of triage severity levels"""
    EMERGENCY = "emergency"  # Immediate medical attention needed
    URGENT = "urgent"        # Care needed within 24 hours
    SEMI_URGENT = "semi-urgent"  # Care needed within 72 hours
    ROUTINE = "routine"      # Care needed but not urgent
    SELF_CARE = "self-care"  # Can be managed at home

class CarePathway(Enum):
    """Enumeration of care pathways"""
    EMERGENCY_ROOM = "emergency_room"
    URGENT_CARE = "urgent_care"
    PRIMARY_CARE = "primary_care"
    TELEHEALTH = "telehealth"
    SELF_MANAGEMENT = "self_management"

class TriageAgent:
    """
    A decision tree-based agent for medical triage
    """
    
    def __init__(self):
        """Initialize the triage agent with decision rules"""
        # Emergency symptoms that require immediate medical attention
        self.emergency_indicators = [
            r"difficulty breathing", r"shortness of breath", r"chest pain", 
            r"severe (pain|bleeding)", r"unconscious", r"unresponsive",
            r"stroke", r"heart attack", r"seizure", r"unable to (speak|move)",
            r"sudden (numbness|weakness)", r"severe head(ache)? with (fever|stiff neck)",
            r"suicidal", r"overdose", r"poisoning",
            # Pediatric emergency indicators
            r"(baby|infant|child).*(not breathing|turning blue|choking)",
            r"(baby|infant|child).*(unresponsive|unconscious|won't wake up)",
            r"(baby|infant|child).*(seizure|convulsion)",
            r"(baby|infant|child).*(severe dehydration|not urinating)",
            r"(baby|infant|child).*(rash.*(doesn't blanch|doesn't fade))",
            r"(baby|infant|child).*(lethargic|extremely weak)",
            r"(baby|infant|child).*(high fever.*(under|less than) (3|three) months)"
        ]
        
        # Urgent symptoms that require prompt but not immediate care
        self.urgent_indicators = [
            r"high fever", r"persistent vomiting", r"dehydration",
            r"infection", r"moderate (pain|bleeding)", r"broken bone",
            r"deep cut", r"burn", r"allergic reaction", r"pregnancy complication",
            r"severe (rash|swelling)", r"eye injury", r"mental health crisis",
            # Pediatric urgent indicators
            r"(baby|infant|child).*(fever.*(102|103|104|105))",
            r"(baby|infant|child).*(not feeding|refusing to eat|drink)",
            r"(baby|infant|child).*(unusual (crying|screaming))",
            r"(baby|infant|child).*(bulging fontanelle)",
            r"(baby|infant|child).*(persistent vomiting|diarrhea)"
        ]
        
        # Semi-urgent symptoms
        self.semi_urgent_indicators = [
            r"ear(ache|infection)", r"sinus (pain|infection)", r"minor infection",
            r"mild to moderate pain", r"(sprain|strain)", r"minor injury",
            r"persistent symptoms", r"worsening chronic condition",
            r"fever.*(101|102)", r"swollen.*(ankle|joint|knee|wrist)",
            r"diarrhea.*(3|three|several) days", r"persistent diarrhea",
            r"fall", r"fell", r"falling", r"twisted", r"twist",
            r"asthma.*(significantly worse|severe|attack)",
            r"blood sugar.*(very high|dangerously high|extremely elevated)",
            r"(diabetes|diabetic).*(uncontrolled|out of control)",
            r"back pain.*(severe|debilitating|can't move)",
            # Pediatric semi-urgent indicators
            r"(baby|infant|child).*(ear infection|ear pain)",
            r"(baby|infant|child).*(fever.*(100|101))",
            r"(baby|infant|child).*(mild rash)",
            r"(baby|infant|child).*(cough|cold).*(several days)"
        ]
        
        # Routine care symptoms
        self.routine_indicators = [
            r"chronic condition", r"follow-up", r"medication refill",
            r"mild symptoms", r"general check-up", r"non-urgent concern",
            r"mild (rash|pain)", r"routine screening",
            r"asthma.*(worse|worsening)", r"persistent.*(pain|ache)",
            r"blood sugar.*(higher|elevated|abnormal)",
            r"(diabetes|diabetic).*(control|management)",
            r"back pain.*(chronic|persistent|ongoing)",
            r"(month|months|week|weeks|day|days)",
            # Pediatric routine indicators
            r"(baby|infant|child).*(mild fever)",
            r"(baby|infant|child).*(minor cough|runny nose)",
            r"(baby|infant|child).*(diaper rash)",
            r"(baby|infant|child).*(feeding question|growth concern)"
        ]
        
        # Self-care symptoms
        self.self_care_indicators = [
            r"common cold", r"minor headache", r"mild fever", r"sore throat",
            r"minor cut", r"scrape", r"bruise", r"mild allergies", 
            r"mild digestive issues", r"general fatigue", r"tired", r"fatigue",
            r"runny nose", r"sneezing", r"cough", r"mild cough", r"occasional cough",
            r"slight headache", r"mild pain", r"minor pain", r"slight pain",
            r"itchy", r"itching", r"dry skin", r"rash", r"minor rash",
            r"upset stomach", r"indigestion", r"gas", r"bloating",
            # Pediatric self-care indicators
            r"(baby|infant|child).*(slight fever|low-grade fever)",
            r"(baby|infant|child).*(minor cough|sniffle)",
            r"(baby|infant|child).*(small scrape|minor bruise)",
            r"(baby|infant|child).*(teething)",
            r"(baby|infant|child).*(mild diaper rash)",
            r"(baby|infant|child).*(occasional fussiness)"
        ]
    
    def _check_pattern_match(self, text: str, patterns: List[str]) -> bool:
        """
        Check if any pattern matches the text
        
        Args:
            text: Text to check
            patterns: List of regex patterns
            
        Returns:
            True if any pattern matches, False otherwise
        """
        text = text.lower()
        print(f"Checking text: {text}")
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                print(f"Matched pattern: {pattern}")
                return True
        return False
    
    def classify_severity(self, symptoms: str) -> TriageSeverity:
        """
        Classify the severity of symptoms
        
        Args:
            symptoms: Description of symptoms
            
        Returns:
            TriageSeverity level
        """
        print(f"Classifying severity for symptoms: {symptoms}")
        
        # Split multiple symptoms if they appear to be in a list format
        symptom_list = symptoms.split("\n")
        if len(symptom_list) == 1:
            # Try splitting by bullet points or dashes
            if " - " in symptoms:
                symptom_list = symptoms.split(" - ")
            elif "• " in symptoms:
                symptom_list = symptoms.split("• ")
            elif ", " in symptoms:
                symptom_list = symptoms.split(", ")
        
        # Remove empty items and strip whitespace
        symptom_list = [s.strip() for s in symptom_list if s.strip()]
        print(f"Parsed symptom list: {symptom_list}")
        
        # Check each symptom individually and take the most severe classification
        severity_levels = []
        
        for symptom in symptom_list:
            print(f"Analyzing symptom: {symptom}")
            
            # Check for chronic conditions with specific keywords
            is_chronic = any(term in symptom.lower() for term in ["chronic", "persistent", "ongoing", "months", "weeks"])
            is_worsening = any(term in symptom.lower() for term in ["worse", "worsening", "increased", "higher", "elevated"])
            
            # Add some common symptom patterns that might be missed
            if any(term in symptom.lower() for term in ["severe", "extreme", "intense", "unbearable", "worst"]):
                print(f"Detected severe symptom indicators in: {symptom}")
                if any(term in symptom.lower() for term in ["pain", "headache", "chest", "breathing"]):
                    print(f"Classified as EMERGENCY due to severe pain/breathing issues in: {symptom}")
                    severity_levels.append(TriageSeverity.EMERGENCY)
                    continue
            
            # Check for specific chronic conditions
            if "asthma" in symptom.lower():
                if any(term in symptom.lower() for term in ["attack", "can't breathe", "severe", "emergency"]):
                    print(f"Classified as EMERGENCY (severe asthma): {symptom}")
                    severity_levels.append(TriageSeverity.EMERGENCY)
                elif is_worsening:
                    print(f"Classified as SEMI_URGENT (worsening asthma): {symptom}")
                    severity_levels.append(TriageSeverity.SEMI_URGENT)
                else:
                    print(f"Classified as ROUTINE (stable asthma): {symptom}")
                    severity_levels.append(TriageSeverity.ROUTINE)
                continue
                
            if "blood sugar" in symptom.lower() or "diabetes" in symptom.lower():
                if any(term in symptom.lower() for term in ["very high", "extremely", "dangerously"]):
                    print(f"Classified as URGENT (dangerous blood sugar): {symptom}")
                    severity_levels.append(TriageSeverity.URGENT)
                elif is_worsening:
                    print(f"Classified as SEMI_URGENT (elevated blood sugar): {symptom}")
                    severity_levels.append(TriageSeverity.SEMI_URGENT)
                else:
                    print(f"Classified as ROUTINE (blood sugar concern): {symptom}")
                    severity_levels.append(TriageSeverity.ROUTINE)
                continue
                
            if "back pain" in symptom.lower():
                if any(term in symptom.lower() for term in ["severe", "can't move", "debilitating"]):
                    print(f"Classified as SEMI_URGENT (severe back pain): {symptom}")
                    severity_levels.append(TriageSeverity.SEMI_URGENT)
                elif is_chronic:
                    print(f"Classified as ROUTINE (chronic back pain): {symptom}")
                    severity_levels.append(TriageSeverity.ROUTINE)
                else:
                    print(f"Classified as SELF_CARE (mild back pain): {symptom}")
                    severity_levels.append(TriageSeverity.SELF_CARE)
                continue
            
            # Standard pattern matching
            if self._check_pattern_match(symptom, self.emergency_indicators):
                print(f"Classified as EMERGENCY: {symptom}")
                severity_levels.append(TriageSeverity.EMERGENCY)
                continue
            
            if self._check_pattern_match(symptom, self.urgent_indicators):
                print(f"Classified as URGENT: {symptom}")
                severity_levels.append(TriageSeverity.URGENT)
                continue
            
            if self._check_pattern_match(symptom, self.semi_urgent_indicators):
                print(f"Classified as SEMI_URGENT: {symptom}")
                severity_levels.append(TriageSeverity.SEMI_URGENT)
                continue
            
            if self._check_pattern_match(symptom, self.routine_indicators):
                print(f"Classified as ROUTINE: {symptom}")
                severity_levels.append(TriageSeverity.ROUTINE)
                continue
            
            # If it's a chronic condition that's worsening, at least classify as ROUTINE
            if is_chronic and is_worsening:
                print(f"Classified as ROUTINE (worsening chronic condition): {symptom}")
                severity_levels.append(TriageSeverity.ROUTINE)
                continue
                
            print(f"No specific patterns matched for: {symptom}, defaulting to SELF_CARE")
            severity_levels.append(TriageSeverity.SELF_CARE)
        
        # If we have multiple severity levels, take the most severe one
        if severity_levels:
            # Sort by severity (EMERGENCY is most severe)
            severity_order = {
                TriageSeverity.EMERGENCY: 0,
                TriageSeverity.URGENT: 1,
                TriageSeverity.SEMI_URGENT: 2,
                TriageSeverity.ROUTINE: 3,
                TriageSeverity.SELF_CARE: 4
            }
            severity_levels.sort(key=lambda x: severity_order[x])
            most_severe = severity_levels[0]
            print(f"Multiple symptoms found. Most severe classification: {most_severe.value}")
            return most_severe
        
        # If no symptoms were processed, default to SELF_CARE
        print("No symptoms were successfully processed, defaulting to SELF_CARE")
        return TriageSeverity.SELF_CARE
    
    def recommend_care_pathway(self, severity: TriageSeverity) -> CarePathway:
        """
        Recommend a care pathway based on severity
        
        Args:
            severity: TriageSeverity level
            
        Returns:
            Recommended CarePathway
        """
        pathway_map = {
            TriageSeverity.EMERGENCY: CarePathway.EMERGENCY_ROOM,
            TriageSeverity.URGENT: CarePathway.URGENT_CARE,
            TriageSeverity.SEMI_URGENT: CarePathway.PRIMARY_CARE,
            TriageSeverity.ROUTINE: CarePathway.TELEHEALTH,
            TriageSeverity.SELF_CARE: CarePathway.SELF_MANAGEMENT
        }
        
        return pathway_map[severity]
    
    def _is_pediatric_case(self, symptoms: str) -> bool:
        """
        Determine if the symptoms are related to a pediatric case
        
        Args:
            symptoms: Description of symptoms
            
        Returns:
            True if pediatric case, False otherwise
        """
        pediatric_indicators = [
            r"(baby|infant|child|kid|toddler|newborn)",
            r"(\d+)[\s-]*(month|year|week)[\s-]*(old)",
            r"my son", r"my daughter",
            r"pediatric", r"children"
        ]
        
        return self._check_pattern_match(symptoms, pediatric_indicators)
    
    def get_care_instructions(self, pathway: CarePathway, symptoms: str = "") -> str:
        """
        Get care instructions based on the recommended pathway and symptoms
        
        Args:
            pathway: Recommended CarePathway
            symptoms: Description of symptoms (optional)
            
        Returns:
            Care instructions
        """
        # Check if symptoms involve a child
        is_pediatric = self._is_pediatric_case(symptoms)
        
        if is_pediatric:
            instructions = {
                CarePathway.EMERGENCY_ROOM: 
                    "SEEK IMMEDIATE MEDICAL ATTENTION FOR YOUR CHILD. Go to the nearest pediatric emergency room or call emergency services (911). " +
                    "For infants and young children, emergency symptoms require immediate professional evaluation.",
                
                CarePathway.URGENT_CARE:
                    "Take your child to a pediatric urgent care center within 24 hours. If symptoms worsen, go to the emergency room immediately. " +
                    "Children can deteriorate quickly, so close monitoring is essential.",
                
                CarePathway.PRIMARY_CARE:
                    "Schedule an appointment with your child's pediatrician within the next few days. " +
                    "In the meantime, monitor your child's symptoms closely and ensure they stay hydrated.",
                
                CarePathway.TELEHEALTH:
                    "Consider scheduling a telehealth appointment with your child's pediatrician. " +
                    "Have a thermometer and other relevant home medical equipment ready for the consultation.",
                
                CarePathway.SELF_MANAGEMENT:
                    "Your child's symptoms can likely be managed at home with appropriate care. " +
                    "Ensure they get plenty of rest, stay hydrated, and monitor their temperature regularly. " +
                    "If symptoms persist beyond 48 hours, worsen suddenly, or if your child appears unusually lethargic, consult a healthcare provider."
            }
        else:
            instructions = {
                CarePathway.EMERGENCY_ROOM: 
                    "SEEK IMMEDIATE MEDICAL ATTENTION. Go to the nearest emergency room or call emergency services (911).",
                
                CarePathway.URGENT_CARE:
                    "Visit an urgent care center within 24 hours. If symptoms worsen, go to the emergency room.",
                
                CarePathway.PRIMARY_CARE:
                    "Schedule an appointment with your primary care physician within the next few days.",
                
                CarePathway.TELEHEALTH:
                    "Consider scheduling a telehealth appointment with a healthcare provider.",
                
                CarePathway.SELF_MANAGEMENT:
                    "Your symptoms can likely be managed at home with rest and over-the-counter remedies. " +
                    "If symptoms persist or worsen, consult a healthcare provider."
            }
        
        
        return instructions[pathway]
    
    def triage_symptoms(self, symptoms: str) -> Dict[str, Any]:
        """
        Perform triage on symptoms
        
        Args:
            symptoms: Description of symptoms
            
        Returns:
            Dictionary with triage results
        """
        severity = self.classify_severity(symptoms)
        pathway = self.recommend_care_pathway(severity)
        instructions = self.get_care_instructions(pathway, symptoms)
        
        return {
            "severity": severity.value,
            "care_pathway": pathway.value,
            "instructions": instructions
        }

# Example usage
if __name__ == "__main__":
    agent = TriageAgent()
    result = agent.triage_symptoms("I have a severe headache with fever and stiff neck")
    print(f"Severity: {result['severity']}")
    print(f"Care Pathway: {result['care_pathway']}")
    print(f"Instructions: {result['instructions']}")