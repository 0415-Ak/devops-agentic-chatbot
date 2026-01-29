from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import JSONLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory, ConversationSummaryMemory
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import SystemMessage, HumanMessage
import time
import os
import requests
import re
from datetime import datetime

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2', model_kwargs={'token': HF_TOKEN})

# Enhanced LLM initialization with error handling
try:
    llm = ChatGroq(model='gemma2-9b-it', temperature=0.2, api_key=GROQ_API_KEY)
    summary_llm = ChatGroq(model='gemma2-9b-it', temperature=0.2, api_key=GROQ_API_KEY)
    print("✅ LLM models initialized successfully")
except Exception as e:
    print(f"❌ Error loading LLM: {e}")
    raise

def extract_technical_keywords(query):
    """Enhanced keyword extraction for better Stack Overflow searches"""
    try:
        enhanced_prompt = ChatPromptTemplate.from_template("""
Extract the most specific technical keywords for Stack Overflow search from this DevOps query.

EXTRACTION RULES:
1. Prioritize: error codes, service names, technology components
2. Include: programming languages, frameworks, cloud services, specific tools
3. Focus on: concrete technical terms over generic descriptions
4. Format: 3-5 most relevant keywords, space-separated

EXAMPLES:
- "Docker container won't start with nginx" → "docker nginx container startup error"
- "AWS Lambda timeout error in production" → "aws lambda timeout production error"
- "Kubernetes pod stuck in pending state" → "kubernetes pod pending state troubleshooting"
- "Jenkins pipeline failing on deploy stage" → "jenkins pipeline deploy stage failure"

Query: {query}

Extract keywords (return only the keywords):""")
        
        response = summary_llm.invoke(enhanced_prompt.invoke({'query': query}))
        keywords = response.content.strip()
        
        # Fallback keyword extraction if LLM fails
        if not keywords or len(keywords.split()) < 2:
            # Basic regex-based extraction as fallback
            tech_terms = re.findall(r'\b(?:error|failed|timeout|exception|docker|kubernetes|aws|azure|jenkins|nginx|apache|mysql|postgres|redis|kafka|elasticsearch)\b', query.lower())
            keywords = ' '.join(tech_terms[:4]) if tech_terms else query[:50]
            
        return keywords
        
    except Exception as e:
        print(f"⚠️ Keyword extraction failed: {e}")
        # Emergency fallback - use first 4 meaningful words
        words = re.findall(r'\b\w{3,}\b', query.lower())
        return ' '.join(words[:4]) if words else query

def search_stackexchange_enhanced(query, site="stackoverflow", pagesize=2):
    """Enhanced Stack Overflow search with better filtering"""
    try:
        url = "https://api.stackexchange.com/2.3/search/advanced"
        params = {
            "order": "desc",
            "sort": "relevance",
            "q": query,
            "site": site,
            "filter": "withbody",
            "pagesize": pagesize,
            "min_views": 100,  # Filter for quality
            "accepted": True   # Prefer accepted answers
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        # If no accepted answers, try without acceptance filter
        if not data.get("items"):
            params["accepted"] = False
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
        return data
        
    except requests.Timeout:
        print("⚠️ Stack Overflow API timeout")
        return {"items": [], "error": "API timeout"}
    except Exception as e:
        print(f"⚠️ Stack Overflow search error: {e}")
        return {"items": [], "error": str(e)}

def strip_html_enhanced(html):
    """Enhanced HTML stripping with better formatting"""
    try:
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text and clean up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:1500]  # Limit length for token efficiency
    except Exception:
        return html[:1500]

def extract_context_enhanced(data):
    """Enhanced context extraction with quality scoring"""
    try:
        context_blocks = []
        for item in data.get("items", [])[:2]:  # Limit to top 2 results
            title = item.get("title", "")
            body = item.get("body", "")
            score = item.get("score", 0)
            view_count = item.get("view_count", 0)
            
            # Quality indicator
            quality = "⭐ High Quality" if score > 5 and view_count > 1000 else "📝 Standard"
            
            cleaned_body = strip_html_enhanced(body)
            context_block = f"{quality} Solution:\nTitle: {title}\nDetails: {cleaned_body}\n"
            context_blocks.append(context_block)
            
        return "\n".join(context_blocks) if context_blocks else "No relevant Stack Overflow solutions found."
        
    except Exception as e:
        print(f"⚠️ Context extraction error: {e}")
        return f"Error extracting context: {str(e)}"

# Enhanced prompts with better structure and guidance
enhanced_context_summary_prompt = ChatPromptTemplate.from_template("""
Analyze the Stack Overflow solutions and create a technical summary for DevOps troubleshooting.

ANALYSIS REQUIREMENTS:
- Focus on actionable troubleshooting steps and technical insights
- Identify root causes and solution patterns
- Extract specific commands, configurations, or code fixes
- Note any version-specific or environment-specific considerations
- Maintain technical accuracy and relevance

FORMAT: Single comprehensive paragraph (250-300 words)
AVOID: Repeating the search query, generic advice, or non-technical content

Stack Overflow Context:
{context}

Search Query: {st_query}

Technical Summary:
""")

enhanced_incident_resolution_prompt = ChatPromptTemplate.from_template("""
You are a Senior DevOps Engineer providing incident resolution guidance.

RESPONSE REQUIREMENTS:
- Begin with a brief 2–3 sentence **overview** summarizing the incident or error
- Identify 2–3 **common technical causes** for the issue
- Provide immediate actionable steps
- Format in clear Markdown with proper structure
- Add confidence level and reasoning
- Include monitoring/prevention recommendations
- If any code, configuration, command, or logs are included, place them inside a **dedicated fenced code block** (```) on a separate line.  
- Do not keep code inline within sentences.  
- Always label code blocks if the language is clear (e.g., ```bash, ```yaml, ```json).

RESPONSE STRUCTURE:
## 📝 Overview
[Brief summary of the issue in 2–3 sentences, describing what the problem is and possible high-level cause]

## 🔍 Common Causes
- [Possible cause 1]
- [Possible cause 2]
- [Possible cause 3]                                                                     

## 🚨 Immediate Actions
[What to do right now]

## 🛠️ Resolution Steps
[Step-by-step fix instructions]

## 📊 Prevention & Monitoring
[How to prevent recurrence]

## ⚠️ Confidence Level
[High/Medium/Low with reasoning]
                                                                       
- End your response with a **short, dynamic follow-up line (1–2 sentences)**.  
   - Acknowledge the specific issue type (e.g., API, networking, Kubernetes).  
   - Ask for additional details if needed.  
   - Invite the user to ask related questions.  
   - Keep it natural and not always the same wording.  
   - Example variations:  
   *“Can you share the exact API error or recent config changes? I can guide you better once I have that. Also, let me know if you’d like to check anything else related to this issue.”*  
   *“Could you provide the affected pod logs or deployment details? That will help narrow it down. Feel free to ask about any related Kubernetes concerns too.”*  
   *“If you can share the VLAN settings from both switches, I’ll give you a more precise fix. And let me know if you’d like to go deeper into related networking issues.”*  

   - Always maintain a **professional, technical tone**.                                                               

Stack Overflow Summary:
{summary}

User Query:
{query}

DevOps Resolution:
""")

def generate_enhanced_answer(summary, query):
    """Generate enhanced answer with error handling and confidence scoring"""
    try:
        response = llm.invoke(enhanced_incident_resolution_prompt.invoke({
            'summary': summary, 
            'query': query
        }))
        
        # Add timestamp and source attribution
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        attribution = f"\n\n---\n*Analysis generated: {timestamp} | Sources: Stack Overflow Community*"
        
        return response.content + attribution
        
    except Exception as e:
        print(f"⚠️ Answer generation failed: {e}")
        return f"""
## ⚠️ Analysis Error
I encountered an issue generating the full analysis. However, based on my DevOps knowledge:

**For your query: "{query}"**

Please check:
1. Service logs for specific error messages
2. Resource utilization (CPU, memory, disk)
3. Network connectivity and dependencies
4. Recent configuration changes

Would you like to provide more specific details about the error or logs?

*Error details: {str(e)}*
"""

def enhanced_incident_tool_func(query):
    """Enhanced incident resolution with better error handling and fallbacks"""
    try:
        print(f"🔍 Processing incident query: {query[:100]}...")
        
        # Step 1: Extract technical keywords
        keywords = extract_technical_keywords(query)
        print(f"🔑 Search keywords: {keywords}")
        
        # Step 2: Search Stack Overflow
        data = search_stackexchange_enhanced(keywords, pagesize=2)
        
        # Step 3: Handle API errors gracefully
        if data.get("error") or not data.get("items"):
            print("⚠️ Limited Stack Overflow results, using general DevOps knowledge")
            fallback_summary = f"No specific Stack Overflow solutions found for '{keywords}'. Providing general DevOps guidance."
            return generate_enhanced_answer(fallback_summary, query)
        
        # Step 4: Extract and summarize context
        context = extract_context_enhanced(data)
        print(f"📄 Context extracted: {len(context)} characters")
        
        # Step 5: Generate technical summary
        summary = summary_llm.invoke(enhanced_context_summary_prompt.invoke({
            'context': context, 
            'st_query': keywords
        }))
        
        # Step 6: Generate final answer
        return generate_enhanced_answer(summary.content, query)
        
    except Exception as e:
        print(f"❌ Critical error in incident tool: {e}")
        return f"""
## 🚨 System Error in Incident Analysis

I encountered a technical issue while analyzing your query. Let me provide basic guidance:

**Your Query:** {query}

**General Troubleshooting Steps:**
1. **Check Service Status:** Verify if the service is running
2. **Review Logs:** Look for recent error messages or warnings  
3. **Resource Check:** Monitor CPU, memory, and disk usage
4. **Network Validation:** Test connectivity to dependencies
5. **Recent Changes:** Review any recent deployments or config changes

**Next Steps:**
- Share specific error messages or logs for detailed analysis
- Provide service/infrastructure details for targeted help

*Technical error: {str(e)}*
"""

# Enhanced tool with better description and routing logic
enhanced_incident_tool = Tool(
    name="stackoverflow_devops_resolver",
    func=enhanced_incident_tool_func,
    description="""
CRITICAL: Use this tool for DevOps, SecOps, and SRE incident resolution queries.

TRIGGER CONDITIONS:
- Error messages, stack traces, or exception details
- Deployment, CI/CD, or release issues
- Infrastructure problems (containers, orchestration, cloud services)
- Monitoring alerts or performance issues  
- Service outages or availability problems
- Configuration or networking issues

EXAMPLES THAT TRIGGER THIS TOOL:
- "Docker container keeps crashing with exit code 1"
- "Jenkins pipeline failing at deployment stage"
- "AWS Lambda timeout errors in production"
- "Kubernetes pods stuck in pending state"
- "Nginx returning 502 bad gateway"
- "Database connection pool exhausted"

DO NOT USE FOR:
- Trailblazer Ops product questions (use FAQResponder)
- Log analysis requiring RCA reports (use RCA tool)
- General questions without technical issues
""",
    return_direct=True
)

# Enhanced FAQ tool remains similar but with better routing
try:
    faq_loader = JSONLoader(
        file_path="trailblazer_ops_faq.json",
        jq_schema=".[] | {question: .question, answer: .answer}",
        text_content=False
    )
    faq_docs = faq_loader.load()
    faq_vectordb = FAISS.from_documents(faq_docs, embedding)
    retriever = faq_vectordb.as_retriever(search_kwargs={"k": 3})
    faq_qa = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        chain_type='stuff',
        return_source_documents=True
    )
    print("✅ FAQ system initialized successfully")
    
except Exception as e:
    print(f"❌ Error loading FAQ system: {e}")
    faq_qa = None

def enhanced_faq_func(query):
    """Enhanced FAQ function with relevance checking"""
    try:
        if not faq_qa:
            return "FAQ system unavailable. Please contact support for Trailblazer Ops questions."
            
        # Check if query is actually about Trailblazer Ops
        tbo_keywords = ['trailblazer', 'tbo', 'extension', 'subscription', 'onboarding', 'features', 'pricing', 'integration']
        is_tbo_query = any(keyword in query.lower() for keyword in tbo_keywords)
        
        if not is_tbo_query:
            return "This query doesn't appear to be about Trailblazer Ops. Please use more specific product-related terms or ask about technical incidents instead."
        
        result = faq_qa({"query": query})
        answer = result.get('result', 'No relevant FAQ found.')
        
        # Add source attribution if available
        sources = result.get('source_documents', [])
        if sources:
            source_info = f"\n\n*Source: Trailblazer Ops Knowledge Base*"
            answer += source_info
            
        return answer
        
    except Exception as e:
        print(f"⚠️ FAQ tool error: {e}")
        return f"Error accessing FAQ system: {str(e)}. Please contact support for Trailblazer Ops questions."

enhanced_faq_tool = Tool(
    name="trailblazer_ops_faq",
    func=enhanced_faq_func,
    description="""
EXCLUSIVE USE: Trailblazer Ops product-specific questions only.

TRIGGER CONDITIONS (query must contain these terms):
- "Trailblazer Ops", "TBO", or "Trailblazer"
- Product features, capabilities, or limitations
- Subscription, pricing, or billing questions
- Extension installation or configuration
- Onboarding, setup, or account management
- Integration with other tools or platforms
- Usage instructions or how-to questions

EXAMPLES THAT TRIGGER THIS TOOL:
- "How do I install Trailblazer Ops extension?"
- "What integrations does TBO support?"
- "Trailblazer Ops pricing plans"
- "How to configure Trailblazer Ops with AWS?"

STRICT RULE: If query doesn't mention Trailblazer Ops explicitly, DO NOT use this tool.
""",
    return_direct=True
)

# Enhanced log parsing and RCA generation
enhanced_log_parsing_prompt = ChatPromptTemplate.from_template("""
Extract structured incident details from this log entry with high precision.

EXTRACTION REQUIREMENTS:
- Parse timestamps in any format to ISO format
- Identify specific error codes and types
- Extract service/component names
- Capture host/server information
- Find correlation IDs or trace IDs
- Note severity levels

LOG PARSING FORMAT:
- **Incident ID:** [Extract from log or generate if missing]
- **Timestamp:** [Convert to YYYY-MM-DD HH:MM:SS format]
- **Error Type:** [HTTP error, Application error, Infrastructure error, etc.]
- **Error Code:** [Specific error code if present]
- **Error Message:** [Exact error message]
- **Service/Component:** [Service name or component]
- **Host/Server:** [Hostname or server identifier]
- **Severity:** [Critical/High/Medium/Low]
- **Trace ID:** [If present]

Raw Log:
{log}

Structured Incident Details:
""")

enhanced_rca_prompt = ChatPromptTemplate.from_template("""
You are a Principal Site Reliability Engineer with 15+ years of experience. Generate a comprehensive RCA report.

ANALYSIS FRAMEWORK:
1. **Timeline Reconstruction:** Build sequence of events from log timestamps
2. **Error Propagation:** Trace how the error spread through systems
3. **Impact Assessment:** Evaluate business and technical impact
4. **Pattern Recognition:** Compare with known incident patterns

QUALITY REQUIREMENTS:
- Use specific technical terminology
- Reference exact log entries and timestamps
- Provide actionable remediation steps
- Include monitoring and alerting improvements
- Add confidence scores for conclusions

Incident Details:
{incident_details}

Stack Overflow Context:
{summary}

---

# 🚨 Root Cause Analysis Report

## 📋 Executive Summary
*[2-3 sentence summary of the incident and resolution]*

## 📊 Incident Overview
{incident_details}

## 🔍 Root Cause Analysis

### What Happened
*[Detailed timeline of events with timestamps]*

### Why It Happened  
*[Technical root cause with supporting evidence]*

### Contributing Factors
*[Environmental, process, or human factors]*

## 🛠️ Immediate Resolution
*[Steps taken to resolve the incident]*

## 🔒 Prevention Measures

### Short-term (1-2 weeks)
- [ ] [Specific action items]

### Long-term (1-3 months)  
- [ ] [Strategic improvements]

## 📈 Monitoring & Alerting Improvements
*[Specific monitoring recommendations]*

## 📚 Lessons Learned
*[Key takeaways and process improvements]*

## ⚠️ Confidence Assessment
**Confidence Level:** [High 90%+ | Medium 70-89% | Low <70%]
**Reasoning:** [Why this confidence level]

---
*Report generated: {timestamp} | Analyst: Fixora AI*
""")

def enhanced_rca_tool_func(log):
    """Enhanced RCA generation with better error handling and structured analysis"""
    try:
        print(f"📋 Processing RCA for log entry: {len(log)} characters")
        
        # Step 1: Parse incident details with enhanced extraction
        incident_details = summary_llm.invoke(enhanced_log_parsing_prompt.invoke({'log': log}))
        print("✅ Incident details extracted")
        time.sleep(0.5)  # Rate limiting
        
        # Step 2: Generate search keywords from log
        keywords = extract_technical_keywords(log)
        print(f"🔑 RCA search keywords: {keywords}")
        
        # Step 3: Search for similar incidents
        data = search_stackexchange_enhanced(keywords)
        context = extract_context_enhanced(data)
        
        # Step 4: Generate contextual summary
        summary = summary_llm.invoke(enhanced_context_summary_prompt.invoke({
            'context': context, 
            'st_query': keywords
        }))
        print("✅ Context summary generated")
        time.sleep(0.5)  # Rate limiting
        
        # Step 5: Generate comprehensive RCA
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        rca = llm.invoke(enhanced_rca_prompt.invoke({
            'incident_details': incident_details.content,
            'summary': summary.content,
            'timestamp': timestamp
        }))
        
        print("✅ RCA report generated successfully")
        return rca.content
        
    except Exception as e:
        print(f"❌ RCA generation failed: {e}")
        return f"""
# 🚨 RCA Generation Error

## Issue
Failed to generate complete RCA report due to: {str(e)}

## Manual Analysis Required
Based on the provided log, please:

1. **Identify Key Elements:**
   - Timestamp of the incident
   - Error messages or codes
   - Affected services or components
   - Server/host information

2. **Investigate:**
   - Check service health dashboards
   - Review related application logs
   - Validate infrastructure metrics
   - Examine recent deployment history

3. **Document:**
   - Timeline of events
   - Root cause hypothesis
   - Resolution steps taken
   - Prevention measures

**Log Summary:** {log[:500]}...

*Please contact your DevOps team for manual RCA generation.*
"""

enhanced_rca_tool = Tool(
    name='comprehensive_rca_generator',
    func=enhanced_rca_tool_func,
    description="""
SPECIFIC USE: Generate comprehensive RCA reports from raw log data and incident traces.

TRIGGER CONDITIONS (query must contain):
- Multi-line log entries with timestamps
- Error stack traces or exception dumps
- Structured log data from monitoring systems
- Incident reports with technical details
- Raw application or infrastructure logs

REQUIRED LOG CHARACTERISTICS:
- Contains timestamps (any format)
- Has error messages or exception details
- Includes service/component identifiers
- Shows system state or error conditions

EXAMPLES THAT TRIGGER THIS TOOL:
- Pasted log files from applications
- Stack traces from error monitoring
- Infrastructure monitoring alerts with details
- Raw syslog or application log entries
- Multi-line error reports

DO NOT USE FOR:
- Single-line error messages (use stackoverflow_devops_resolver)
- General questions about errors
- Trailblazer Ops product questions
""",
    return_direct=True
)

# Enhanced system message with clear routing logic and conversation state management
enhanced_system_message = SystemMessage(content="""
You are Fixora, the AI DevOps Expert for Trailblazer Ops platform.

CORE CAPABILITIES:
🔧 DevOps incident resolution and troubleshooting
📊 Root Cause Analysis (RCA) report generation  
❓ Trailblazer Ops platform support and guidance
🧠 General technical consultation and reasoning

ROUTING DECISION TREE:

1. **FOR TRAILBLAZER OPS QUESTIONS** → Use `trailblazer_ops_faq`
   - Query mentions: "Trailblazer Ops", "TBO", "extension", "features", "pricing"
   - Questions about: installation, configuration, integrations, billing

2. **FOR RAW LOG ANALYSIS** → Use `comprehensive_rca_generator`  
   - Input contains: multi-line logs, timestamps, stack traces
   - Structured log data from monitoring systems
   - Incident dumps requiring formal RCA reports

3. **FOR DEVOPS INCIDENTS** → Use `stackoverflow_devops_resolver`
   - Error messages without full logs
   - Deployment, CI/CD, infrastructure issues
   - Service outages or performance problems
   - Configuration or networking troubles

4. **FOR GENERAL QUESTIONS** → Answer directly using your knowledge
   - Conceptual DevOps questions
   - Best practices and recommendations
   - Technology comparisons and explanations
   - Always maintain a **professional, technical tone**.

CONVERSATION MANAGEMENT:
- Track incident context across multiple turns
- Reference previous troubleshooting attempts
- Build cumulative understanding of user's infrastructure
- Maintain professional, technical tone
- Focus on actionable solutions

RESPONSE QUALITY STANDARDS:
- Provide confidence levels for recommendations
- Include immediate actions and long-term prevention
- Use structured formatting with clear sections
- Add relevant monitoring and alerting suggestions
- Reference best practices and industry standards

RESTRICTIONS:
- Never mention LLM models, OpenAI, Google, Meta, or tool names
- Avoid personal topics or identity-related discussions  
- Focus on the most recent question in multi-part queries
- Maintain technical accuracy and professional demeanor

Remember: You are a senior DevOps expert helping teams resolve critical infrastructure and application issues efficiently.
""")

# Enhanced memory with structured conversation tracking
enhanced_memory = ConversationSummaryBufferMemory(
    llm=summary_llm,
    memory_key="chat_history", 
    return_messages=True,
    max_token_limit=4000,
    ai_prefix="Fixora",
    human_prefix="User"
)

# Add the enhanced system message to memory
enhanced_memory.chat_memory.add_message(enhanced_system_message)

# Initialize enhanced agent with improved error handling
try:
    enhanced_agent = initialize_agent(
        tools=[enhanced_faq_tool, enhanced_incident_tool, enhanced_rca_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=enhanced_memory,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=3,  # Prevent infinite loops
        early_stopping_method="generate"  # Better error recovery
    )
    
    print("🚀 Enhanced Fixora AI agent initialized successfully!")
    print("🔧 Available capabilities:")
    print("  - DevOps incident resolution")
    print("  - RCA report generation") 
    print("  - Trailblazer Ops platform support")
    print("  - Technical consultation")
    
except Exception as e:
    print(f"❌ Error initializing enhanced agent: {e}")
    enhanced_agent = None
    raise

# Export the enhanced agent for use
agent = enhanced_agent


if __name__ == "__main__":
    if agent:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                break
            if not user_input:
                continue

            try:
                response = agent.invoke(user_input)
                print(response["output"], end="\n\n")  # ✅ Only print the final answer
            except Exception as e:
                print(f"Error: {str(e)}\n")
    else:
        print("Agent initialization failed.")

