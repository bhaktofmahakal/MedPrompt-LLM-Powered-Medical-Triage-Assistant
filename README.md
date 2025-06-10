# MedPrompt: AI-Powered Medical Triage Assistant

![MedPrompt Banner](https://img.shields.io/badge/MedPrompt-AI%20Medical%20Triage-4361ee?style=for-the-badge&logo=medical&logoColor=white)

MedPrompt is a sophisticated medical triage assistant powered by Large Language Models (LLMs) that analyzes patient-reported symptoms and provides evidence-based care recommendations. By combining advanced NLP capabilities with medical knowledge retrieval, MedPrompt helps users understand the potential severity of their symptoms and directs them to appropriate care pathways.

## 🌟 Key Features

- **Intelligent Symptom Analysis**: Leverages MedLLaMA 2, a specialized medical LLM, to analyze and interpret complex symptom descriptions
- **Evidence-Based Responses**: Integrates with PubMed through a Retrieval-Augmented Generation (RAG) pipeline to provide medically accurate information
- **Multi-Level Triage System**: Classifies symptoms into five severity levels and recommends appropriate care pathways
- **Responsive Web Interface**: Clean, intuitive UI with real-time feedback and accessibility features
- **Robust Error Handling**: Gracefully manages connection issues, timeouts, and resource limitations
- **Pediatric Symptom Recognition**: Specialized patterns for recognizing and appropriately triaging children's symptoms
- **Chronic Condition Management**: Intelligent assessment of worsening chronic conditions like asthma, diabetes, and chronic pain


---


## 🔬 Technical Architecture

### 1. RAG (Retrieval-Augmented Generation) Pipeline
- **Real-time PubMed Integration**: Searches and retrieves relevant medical literature based on symptom descriptions
- **Vector Embedding**: Uses sentence-transformers to create semantic embeddings of medical texts
- **FAISS Vector Store**: Efficiently indexes and retrieves relevant medical information
- **Context Enhancement**: Augments LLM prompts with evidence-based medical information
- **Fallback Mechanisms**: Provides graceful degradation when external services are unavailable

### 2. Advanced Triage System
- **Pattern-Based Classification**: Uses sophisticated regex patterns to identify symptom severity
- **Multi-Symptom Analysis**: Intelligently parses and evaluates multiple symptoms simultaneously
- **Severity Hierarchy**: Implements a priority-based classification system that highlights the most urgent concerns
- **Specialized Pattern Recognition**: Contains over 100 medical patterns for accurate symptom classification
- **Care Pathway Mapping**: Maps severity levels to appropriate healthcare settings (Emergency Room, Urgent Care, etc.)
- **Age-Appropriate Recommendations**: Provides different guidance for pediatric vs. adult cases

### 3. LLM Integration
- **Local Inference**: Uses Ollama to run MedLLaMA 2 locally, ensuring privacy and reducing latency
- **Optimized Prompting**: Carefully crafted prompts that combine symptom information, triage results, and medical context
- **Resource Management**: Intelligent handling of memory constraints and timeout scenarios
- **Response Formatting**: Structured outputs with explanations, self-care advice, and warning signs

### 4. FastAPI Backend
- **Asynchronous Processing**: Handles multiple requests efficiently
- **Robust Error Handling**: Gracefully manages exceptions and provides helpful error messages
- **Health Monitoring**: Includes health check endpoints for system monitoring
- **Static File Serving**: Efficiently serves the frontend application


---


## 📊 Triage Classification System

MedPrompt classifies symptoms into five severity levels:

| Severity Level | Description | Care Pathway | Example Symptoms |
|----------------|-------------|--------------|------------------|
| **Emergency** | Life-threatening conditions requiring immediate medical attention | Emergency Room | Severe chest pain, difficulty breathing, stroke symptoms |
| **Urgent** | Serious conditions requiring prompt care within 24 hours | Urgent Care | High fever, persistent vomiting, moderate injuries |
| **Semi-Urgent** | Conditions requiring care within 72 hours | Primary Care | Ear infections, mild sprains, worsening chronic conditions |
| **Routine** | Non-urgent conditions requiring medical attention | Telehealth | Medication refills, mild symptoms, follow-up care |
| **Self-Care** | Minor conditions that can be managed at home | Self-Management | Common cold, minor headaches, mild allergies |

---
## 🌟 Screenshots

<div align="center">
  <img src="med ai/Screenshot 2025-06-10 124847.png" alt="Chat Interface" width="80%" style="border:1px solid #ccc; border-radius:10px;" />
  <p><em>Responsive, real-time AI Coder system for users.</em></p>
</div>
<div align="center">
  <img src="med ai/Screenshot 2025-06-10 124908.png" alt="Chat Interface" width="80%" style="border:1px solid #ccc; border-radius:10px;" />
  <p><em>Responsive, real-time AI Code Companion system for users.</em></p>
</div>


---

## 🚀 Installation & Setup

<details>
<summary>click to view</summary>

### Prerequisites
- Python 3.8+
- Ollama (for local LLM inference)
- 8GB+ RAM recommended for optimal performance

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/medical_ai_symptom_checker.git
cd medical_ai_symptom_checker
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set up Ollama with MedLLaMA 2
```bash
# Install Ollama from https://ollama.ai/
ollama pull medllama2
```

### Step 4: Start the application
```bash
# For development with auto-reload
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# For production
python app.py
```

### Step 5: Access the web interface
Open your browser and navigate to:
```
http://localhost:8000
```

## 🧪 Testing & Validation

### Example Test Cases

Test the application with these symptom scenarios to evaluate its performance:

#### 1. Mild Symptoms
- "I've had a mild cough for 2 days"
- "I have a slight headache and feel tired"
- "My skin is itchy but there's no rash"

#### 2. Moderate Symptoms
- "I've had a fever of 101°F for the past day with body aches"
- "My ankle is swollen after falling yesterday"
- "I've had diarrhea for 3 days but am still drinking fluids"

#### 3. Severe Symptoms
- "I have a severe headache with fever and stiff neck"
- "I'm experiencing chest pain that radiates to my left arm"
- "I can't catch my breath and my lips look bluish"

#### 4. Chronic Conditions
- "My asthma symptoms have been worse than usual for a week"
- "I've had persistent lower back pain for 3 months"
- "My blood sugar readings have been higher than normal for several days"

#### 5. Pediatric Symptoms
- "My 2-year-old has had a fever of 102°F for 24 hours"
- "My child has a rash that doesn't blanch when pressed"
- "My baby seems lethargic and isn't feeding normally"

</details>

---

## 📁 Project Structure

<details>
<summary>click to view</summary>

```
medical_ai_symptom_checker/
├── app.py                 # FastAPI application with main routes and LLM integration
├── pubmed_rag.py          # RAG pipeline for PubMed data retrieval and processing
├── triage_agent.py        # Decision tree-based triage classification system
├── requirements.txt       # Project dependencies
├── static/                # Frontend files
│   ├── index.html         # Main HTML page with embedded CSS and JavaScript
│   ├── style.css          # CSS styles (placeholder)
│   └── script.js          # JavaScript functionality (placeholder)
├── pubmed_cache/          # Cache directory for PubMed data

```
</details>

---

## 🔧 Advanced Configuration

<details>
<summary>click to view</summary>

### Memory Optimization
For systems with limited RAM, you can adjust these parameters in app.py:
```python
# Reduce token generation to prevent timeouts
"num_predict": 256,
# Reduce context size to help with memory issues
"num_ctx": 2048,
# Use GPU if available
"num_gpu": 1
```

### Timeout Settings
Adjust the timeout value based on your system's performance:
```python
# Reduce timeout for faster feedback on slow systems
timeout=15  # seconds
```

### PubMed API Configuration
Set your email for the PubMed API in pubmed_rag.py:
```python
# Set your email for Entrez (required by NCBI)
Entrez.email = "your-email@example.com"
```

</details>

---

## ⚠️ Medical Disclaimer

**IMPORTANT**: MedPrompt is designed for informational purposes only and does not provide medical diagnosis, advice, or treatment. The triage recommendations are based on pattern matching and AI analysis, which may not capture all medical nuances or individual circumstances.

Always consult with a qualified healthcare provider for medical concerns. In case of emergency, call emergency services (911 in the US) or go to your nearest emergency room immediately.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- Utsav Mishra - Initial development and design

## 🙏 Acknowledgments

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Python-blue?style=for-the-badge&logo=python&logoColor=white" alt="Made with Python">
  <img src="https://img.shields.io/badge/Powered%20by-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="Powered by FastAPI">
  <img src="https://img.shields.io/badge/AI-MedLLaMA%202-red?style=for-the-badge&logo=openai&logoColor=white" alt="AI - MedLLaMA 2">
</p>
