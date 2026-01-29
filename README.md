# 🤖 Agentic AI Chatbot – Team Project

This repository contains a team-built **Agentic AI Chatbot** designed to assist DevOps/SRE workflows using multiple intelligent tools such as Retrieval-Augmented Generation (RAG), OCR, and structured query generation.

This README highlights **my individual contributions** to the project.

---

## 👨‍💻 My Contributions

I was responsible for designing and implementing the following core components of the chatbot:

### 1️⃣ Data Scraping for RAG Implementation
- Scraped high-quality **question–answer datasets from Stack Overflow** relevant to DevOps, SRE, and debugging use cases.
- Cleaned, structured, and stored the scraped data for **RAG-based knowledge retrieval**.
- Ensured the dataset was optimized for embedding and vector storage.

**Purpose:**  
To enable the chatbot to generate accurate, context-aware answers using real-world technical discussions.

---

### 2️⃣ RAG (Retrieval-Augmented Generation) Tool
- Designed and implemented the **RAG pipeline** within the agentic chatbot architecture.
- Integrated vector-based retrieval to fetch relevant context before LLM response generation.
- Enabled the chatbot to answer technical queries grounded in scraped Stack Overflow data.

**Key Features:**
- Context-aware responses  
- Reduced hallucinations  
- Scalable knowledge integration  

---

### 3️⃣ OCR Tool Integration
- Built an **OCR tool** capable of extracting text from:
  - Digital PDFs  
  - Images (PNG, JPG, scanned documents)
- Integrated the OCR pipeline into the chatbot so users can:
  - Upload files
  - Ask questions directly about the extracted content

**Use Case:**  
Helps users analyze logs, error screenshots, reports, and documents using natural language queries.

---

### 4️⃣ Structured Query Generation Tool
- Implemented a **structured query tool** that converts natural language user queries into:
  - Datadog searchable queries
  - Jira-compatible structured formats (where applicable)
- Helps bridge the gap between **human language** and **monitoring / ticketing systems**.

**Example:**  
> User: “Show payment API errors in the last 1 hour”  
> Tool Output → Datadog/Jira compatible query format

---

## 🛠️ Technologies Used
- Python  
- LangChain / Agentic AI concepts  
- OCR libraries (for PDF & image text extraction)  
- Vector databases (for RAG)  
- Web scraping tools  

---

## 📌 Note
This is a **team project**, and the repository includes work done by multiple contributors.  
The sections above strictly represent **my personal contributions** to the project.

---

## 📬 Contact
If you have any questions about my work or would like to collaborate, feel free to reach out via GitHub.

