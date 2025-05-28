from fastapi import FastAPI, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import requests
import os
import json


# Import our custom modules
from pubmed_rag import PubMedRAG
from triage_agent import TriageAgent

app = FastAPI(
    title="MedPrompt: LLM-Powered Medical Triage Assistant",
    description="An LLM agent system for patient triage and care routing",
    version="1.0.0"
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize our components
pubmed_rag = PubMedRAG()
triage_agent = TriageAgent()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "medllama2"  # Using MedLLaMA 2 for symptom analysis

@app.get("/")
def serve_homepage():
    """ Serve the index.html file when accessing the root URL """
    return FileResponse(os.path.join("static", "index.html"))

@app.post("/analyze_symptoms")
def analyze_symptoms(symptoms: str = Form(...)):
    """
    Analyze symptoms using RAG-enhanced LLM and provide triage recommendations
    """
    try:
        # Step 1: Perform triage analysis
        triage_result = triage_agent.triage_symptoms(symptoms)
        
        # Step 2: Get relevant medical context from PubMed
        try:
            medical_context = pubmed_rag.get_context_from_symptoms(symptoms)
            print("Successfully retrieved medical context from PubMed")
        except Exception as e:
            print(f"Error retrieving medical context: {str(e)}")
            # Use a generic medical context instead
            medical_context = "Unable to retrieve specific medical literature at this time. " + \
                             "The analysis will proceed based on general medical knowledge."
        
        # Step 3: Enhance the prompt with RAG context and triage information
        headers = {"Content-Type": "application/json"}
        
        prompt = f"""You are a medical AI assistant providing brief, focused symptom analysis.
        
        User Symptoms: {symptoms}
        
        Triage Assessment: 
        - Severity: {triage_result['severity']}
        - Care: {triage_result['care_pathway']}
        
        Medical Context:
        {medical_context}
        
        Provide a VERY BRIEF response (maximum 250 words) with:
        1. Possible explanations for these specific symptoms (2-3 sentences)
        2. Brief self-care advice (2-3 bullet points)
        3. When to seek medical help (1-2 sentences)
        
        IMPORTANT:
        - Be extremely concise and specific
        - Focus only on the symptoms described
        - Avoid generic advice or repetition
        - For serious symptoms, emphasize seeking care
        
        Medical AI:"""
        
        # Step 4: Send the enhanced prompt to the LLM with a timeout
        try:
            # First check if Ollama is running
            try:
                health_check = requests.get("http://localhost:11434/api/tags", timeout=5)
                if health_check.status_code != 200:
                    raise ConnectionError("Ollama health check failed")
                
                # Check if medllama2 model is available
                models_data = health_check.json()
                model_available = False
                for model in models_data.get("models", []):
                    if model.get("name") == "medllama2:latest" or model.get("name") == MODEL_NAME:
                        model_available = True
                        break
                
                if not model_available:
                    raise ConnectionError(f"Model {MODEL_NAME} not available in Ollama")
                    
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                print("Ollama service is not running or not responding")
                raise ConnectionError("Ollama service is not running")
            except ValueError as ve:
                print(f"Error parsing Ollama response: {str(ve)}")
                raise ConnectionError("Error checking Ollama model availability")
                
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME, 
                    "prompt": prompt, 
                    "stream": False,
                    "num_predict": 256,  # Reduce token generation to prevent timeouts
                    "temperature": 0.7,   # Add some randomness but keep responses focused
                    "num_ctx": 2048,     # Reduce context size to help with memory issues
                    "num_gpu": 1         # Use GPU if available
                },
                headers=headers,
                timeout=15  # Reduce timeout to 15 seconds
            )
            
            print("Ollama Response:", response.text)
            
            # Ensure valid JSON response
            response_data = response.text.strip()
            try:
                json_response = json.loads(response_data)
                
                # Check for specific error messages
                if "error" in json_response:
                    error_msg = json_response["error"]
                    print(f"Ollama error: {error_msg}")
                    
                    if "model requires more system memory" in error_msg:
                        ai_response = "The medical AI model requires more memory than is currently available on this system. " + \
                                     "Please try again later when more system resources are available, or consider using a smaller model."
                    else:
                        ai_response = f"The medical AI encountered an error: {error_msg}. Please try again later."
                else:
                    ai_response = json_response.get("response", "I'm sorry, but I couldn't generate a response.")
            except json.JSONDecodeError:
                print(f"Invalid JSON response from Ollama: {response_data}")
                ai_response = "I apologize, but I encountered an issue processing your symptoms. Please try again with a more concise description."
        except requests.exceptions.Timeout:
            print("Ollama request timed out")
            # Check if the symptoms are classified as emergency
            if triage_result["severity"] == "emergency":
                ai_response = "MEDICAL EMERGENCY DETECTED: Your symptoms suggest a potentially life-threatening condition that requires IMMEDIATE medical attention. " + \
                             "Please call emergency services (911) or go to the nearest emergency room immediately. " + \
                             "Do not wait for the AI analysis to complete."
            else:
                ai_response = "I apologize for the delay. The analysis is taking longer than expected. Please try again with a more concise description of your main symptoms."
        except requests.exceptions.ConnectionError as ce:
            print(f"Connection error when connecting to Ollama: {str(ce)}")
            ai_response = "I'm unable to connect to the medical AI service at the moment. Please ensure Ollama is running with the medllama2 model loaded."
        except Exception as e:
            print(f"Error during Ollama request: {str(e)}")
            ai_response = f"An unexpected error occurred: {str(e)}. Please try again with a simpler description."
        
        # Step 5: Return the combined response with triage information
        return {
            "response": ai_response,
            "triage": {
                "severity": triage_result["severity"],
                "care_pathway": triage_result["care_pathway"],
                "instructions": triage_result["instructions"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing symptoms: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "components": {"rag": "active", "triage": "active"}}

# Run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)





