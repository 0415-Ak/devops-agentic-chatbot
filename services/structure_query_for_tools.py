import os
import json
import re
import logging
from typing import Union, Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# ADD THIS LINE TO DISABLE HTTP REQUEST LOGGING
logging.getLogger("httpx").setLevel(logging.WARNING)

#load api key
api_key = os.getenv("GROQ_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryType(Enum):
    SEARCH = "search"
    CLARIFICATION = "clarification"

class ToolType(Enum):
    JIRA = "jira"
    DATADOG = "datadog"

@dataclass
class ToolQuery:
    """Individual tool query structure"""
    service_type: str  # Changed from 'tool'
    query_params: Dict  # This will contain tool-specific structure
    search_context: Optional[str] = None

@dataclass
class QueryResult:
    """Final structured result with multiple tool queries"""
    overall_confidence: float
    is_searchable: bool
    has_time_mention: bool
    mentioned_tools: List[str]
    tool_queries: List[ToolQuery]
    raw_query: str
    clarification_needed: Optional[str] = None

class ToolConfig:
    """Enhanced tool configuration with technical indicators"""
    CONFIG = {
        ToolType.JIRA.value: {
            "keywords": ["jira", "ticket", "issue", "jql", "bug", "story", "task", "epic"],
            "technical_indicators": ["project", "assignee", "status", "priority", "component", "version", "sprint"],
            "search_fields": ["project_key", "issue_type", "status", "assignee", "priority", "labels"]
        },
        ToolType.DATADOG.value: {
            "keywords": ["datadog", "dd", "logs", "metrics", "traces", "apm", "monitoring"],
            "technical_indicators": ["service", "host", "environment", "error", "status", "level", "response_time"],
            "search_fields": ["service_name", "host", "env", "status", "level", "source", "http.status_code"]
        }
    }

class IntelligentQueryProcessor:
    """
    Advanced query processor that generates tool-specific queries with confidence scoring
    """

    def __init__(self, groq_api_key: str, model_name: str = 'llama-3.3-70b-versatile'):
        try:
            self.llm = ChatGroq(
                model_name=model_name,
                temperature=0.2,
                api_key=groq_api_key,
                max_tokens=2048,
                timeout=30
            )
            self.prompts = self._create_intelligent_prompts()
            self.tool_config = ToolConfig()
            self.confidence_threshold = 0.49
        except Exception as e:
            logger.error(f"❌ Failed to initialize QueryProcessor: {e}")
            raise RuntimeError(f"Failed to initialize LLM: {e}")

    def _create_intelligent_prompts(self) -> Dict[str, ChatPromptTemplate]:
        """Creates intelligent prompts for each tool with native query generation"""
        
        # Master template for tool-specific query generation with native queries
        tool_template = """
**ROLE:** You are an Expert {tool_name} Query Generator. Your job is to create precise search parameters AND native {native_query_type} queries for {tool_name} based on user queries.

**CRITICAL RULES:**
1. Extract ONLY relevant entities for {tool_name}
2. Do NOT include time_range in JSON unless user explicitly mentioned time
3. Generate a valid {native_query_type} query in the native_query field
4. Return VALID JSON only - no explanations or markdown
5. Use exact field names as specified in the mapping

**{tool_name} FIELD MAPPING:**
{field_mapping}

**{native_query_type} SYNTAX GUIDELINES:**
{native_syntax_guide}

**USER QUERY:** "{query}"
**TIME MENTIONED:** {has_time}
{time_context}

```python
**REQUIRED JSON OUTPUT FORMAT:**
```json
{{{{
  "tool": "{tool_lowercase}",
  "query_type": "search",
  "filters": {{{{
    // Extract relevant filters based on user query
  }}}},
  "native_query": "// Generate {native_query_type} query here"{time_range_template}
}}}}

**YOUR JSON RESPONSE:**
"""

        # Template for confidence assessment
        confidence_template = """
**ROLE:** Expert Technical Query Analyzer for DevOps/SRE Tools

**MISSION:** Analyze the technical specificity and searchability of user queries for monitoring, incident management, and logging tools (Datadog, JIRA).

**CONFIDENCE SCORING CRITERIA:**

**HIGH CONFIDENCE (0.7-1.0):**
- Specific service/application names (e.g., "payment-service", "auth-api")
- Concrete error indicators (HTTP codes: 500, 404, timeout, crash)
- Technical identifiers (hostnames, IDs, specific log levels)
- Clear operational context (prod environment, specific incidents)
- Time-bound investigations with technical scope

**MODERATE CONFIDENCE (0.5-0.7):**
- General technical terms but lacking specifics
- Mentions tools but vague about what to search
- Some technical context but needs more details
- Generic operational queries

**LOW CONFIDENCE (0.0-0.3):**
- Extremely vague requests ("help me", "fix this", "something is wrong")
- No technical entities or identifiers
- Ambiguous problems without context
- Requests that need significant clarification

**EXAMPLES:**

Query: "Show me error logs from payment-service in the last 2 hours"
Analysis: Specific service (payment-service), clear intent (error logs), defined timeframe
Confidence: 0.95

Query: "Check what's happening with the database"
Analysis: General technical area (database), unclear scope, no specific identifiers
Confidence: 0.4

Query: "Something is broken, please help"
Analysis: Completely vague, no technical details, requires full clarification
Confidence: 0.1

**USER QUERY TO ANALYZE:** "{query}"

**SCORING DECISION:**
Ask yourself: "Is this about monitoring, troubleshooting, or investigating live operational systems?"
- YES = Score 0.5 or higher based on specificity
- NO = Score below 0.3

**INSTRUCTIONS:**
1. Analyze the technical specificity and operational clarity
2. Consider if a DevOps engineer could immediately understand what to search for
3. Evaluate the presence of concrete, searchable entities
4. Return ONLY a decimal number between 0.0 and 1.0

**YOUR CONFIDENCE SCORE (0.0-1.0):**
"""

        # Tool-specific field mappings and native query syntax guides
        tool_configs = {
            "datadog": {
                "field_mapping": """
- service_name: Application/service names
- host: Server/container hostnames
- env: Environment (prod, staging, dev)
- status: Log levels (ERROR, WARN, INFO, DEBUG)
- source: Log sources (nginx, java, python)
- http.status_code: HTTP status codes (200, 404, 500)
- level: Log severity levels""",
                "native_query_type": "Datadog Log Query",
                "native_syntax_guide": """
**Datadog Log Query Syntax:**
- Basic search: service:payment-service
- Multiple conditions: service:payment-service status:error
- HTTP status: @http.status_code:500
- Environment: env:production
- Host: host:web-server-01
- Source: source:nginx
- Time range: Use relative time (last_2h, last_1d)
- Example: service:payment-service @http.status_code:>=400 env:production"""
            },
            
            
            "jira": {
                "field_mapping": """
- project_key: Project identifiers (PROJ, DEV)
- issue_type: Issue types (Bug, Story, Task, Epic)
- status: Issue status (To Do, In Progress, Done)
- assignee: Assigned user
- priority: Priority level (Highest, High, Medium, Low)
- labels: Issue labels (array)""",
                "native_query_type": "JQL (JIRA Query Language)",
                "native_syntax_guide": """
**JQL Syntax Guidelines:**
- Project filter: project = "DEV"
- Issue type: issuetype = Bug
- Status: status = "In Progress"
- Assignee: assignee = currentUser() OR assignee = "john.doe"
- Priority: priority = High
- Labels: labels = "backend"
- Date ranges: created >= -7d OR created >= -3h OR updated >= -1w OR created >= -3M
- Time formats: -1h (hour), -1d (day), -1w (week), -1M (month)
- Combine with AND/OR: project = DEV AND assignee = currentUser() AND status != Done
- With time: project = "PAYMENT" AND issuetype = Bug AND created >= -3h
- Example: project = "PAYMENT" AND issuetype = Bug AND status = "To Do" AND priority = High AND created >= -2d"""
            }
        }
        
        prompts = {}
        
        # Create tool-specific prompts
        for tool in ["datadog", "jira"]:
            config = tool_configs[tool]
            prompts[tool] = ChatPromptTemplate.from_template(
                tool_template.format(
                    tool_name=tool.upper(),
                    tool_lowercase=tool,
                    native_query_type=config["native_query_type"],
                    field_mapping=config["field_mapping"],
                    native_syntax_guide=config["native_syntax_guide"],
                    query="{query}",
                    has_time="{has_time}",
                    time_context="""{time_range_context}""",
                    time_range_template=""",
 "time_range": {{{{
    "start": "{start_time}",
    "end": "{end_time}"
  }}}}"""
                )
            )

        prompts["confidence"] = ChatPromptTemplate.from_template(confidence_template)
        
        return prompts
    
    # DETECT TIME

    def _detect_time_mentions(self, query: str) -> Tuple[bool, Optional[Dict[str, str]]]:
        """Detect if user mentioned time and extract time range if present"""
        query_lower = query.lower()
        now = datetime.now(timezone.utc)
        
        time_patterns = {
            r'last\s+(\d+)\s+hour[s]?': lambda m: (now - timedelta(hours=int(m.group(1))), now),
            r'last\s+(\d+)\s+day[s]?': lambda m: (now - timedelta(days=int(m.group(1))), now),
            r'past\s+(\d+)\s+hour[s]?': lambda m: (now - timedelta(hours=int(m.group(1))), now),
            r'last\s+(\d+)\s+month[s]?': lambda m: (now - timedelta(days=int(m.group(1)) * 30), now),  # ADD THIS LINE
            r'past\s+(\d+)\s+month[s]?': lambda m: (now - timedelta(days=int(m.group(1)) * 30), now),  # ADD THIS LINE
            r'(\d+)\s+month[s]?\s+ago': lambda m: (now - timedelta(days=int(m.group(1)) * 30), now),  #
            r'yesterday': lambda m: (now - timedelta(days=1), now - timedelta(days=1) + timedelta(hours=24)),
            r'today': lambda m: (now.replace(hour=0, minute=0, second=0), now),
            r'(\d+)\s+hours?\s+ago': lambda m: (now - timedelta(hours=int(m.group(1))), now),
            r'(\d+)\s+minutes?\s+ago': lambda m: (now - timedelta(minutes=int(m.group(1))), now),
            r'this\s+week': lambda m: (now - timedelta(days=now.weekday()), now),
            r'last\s+week': lambda m: (now - timedelta(weeks=1), now - timedelta(days=now.weekday())),
            r'this\s+month': lambda m: (now.replace(day=1, hour=0, minute=0, second=0), now),  # ADD THIS LINE
        }
        
        for pattern, time_func in time_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                start_time, end_time = time_func(match)
                return True, {
                    "start": start_time.isoformat().replace('+00:00', 'Z'),
                    "end": end_time.isoformat().replace('+00:00', 'Z'),
                    "start_epoch": int(start_time.timestamp()),
                    "end_epoch": int(end_time.timestamp())
                }
        
        return False, None

    def _calculate_confidence(self, query: str) -> float:
        """Calculate confidence score based on technical indicators"""
        try:
            chain = self.prompts["confidence"] | self.llm
            response = chain.invoke({"query": query})
            
            # Extract confidence score from response
            score_match = re.search(r'(\d*\.?\d+)', response.content)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
            
            
        except Exception as e:
            logger.warning(f"LLM confidence calculation failed: {e}")


    def _determine_mentioned_tools(self, query: str) -> List[str]:
        """Determine which tools are specifically mentioned in the query"""
        query_lower = query.lower()
        mentioned_tools = []
        
        # Only look for explicit tool names, not generic keywords
        tool_names = {
            "jira": ["jira"],
            "datadog": ["datadog", "dd"]
        }
        
        for tool, tool_keywords in tool_names.items():
            for keyword in tool_keywords:
                if keyword in query_lower:
                    mentioned_tools.append(tool)
                    break
        
        return list(set(mentioned_tools))  # Remove duplicates
    


    def _format_query_params(self, tool: str, parsed_json: Dict, has_time: bool, time_range: Optional[Dict]) -> Dict:
        """Format query parameters based on tool type"""
        native_query = parsed_json.get("native_query", "")
        
        if tool == "datadog":
            datadog_params = {
                "logs": {
                    "query": native_query,
                    "limit": 100,
                    "sort": "desc",
                    "indexes": ["*"]
                }
            }
            
            # Add time_range only if time is mentioned, using epochs
            if has_time and time_range:
                datadog_params["logs"]["time_range"] = {
                    "start": time_range.get('start_epoch'),
                    "end": time_range.get('end_epoch')
                }
            else:
                datadog_params["logs"]["time_range"] = None
                
            return datadog_params
            
        elif tool == "jira":
            # For JIRA, time is handled directly in JQL, no separate time_range
            return {
                "jql": {
                    "query": native_query,
                    "fields": ["summary", "status", "assignee", "priority", "created"],
                    "max_results": 50
                }
            }
        
        return {"query": native_query}



    def _generate_tool_query(self, tool: str, query: str, has_time: bool, time_range: Optional[Dict]) -> Optional[ToolQuery]:
        """Generate a query for a specific tool with native query"""
        try:
            prompt = self.prompts[tool]
            
            # Prepare context
            context = {
                "query": query,
                "has_time": str(has_time).lower(),
                "time_range_context": "",
                "start_time": "",
                "end_time": ""
            }
            
            # In _generate_tool_query(), modify the context update section:
            if has_time and time_range:
                # Add this condition for datadog-specific epoch handling
                if tool == "datadog":
                    context.update({
                        "time_range_context": f"Time Range: {time_range['start']} to {time_range['end']} (Epochs: {time_range.get('start_epoch', '')} to {time_range.get('end_epoch', '')})",
                        "start_time": str(time_range.get('start_epoch', '')),
                        "end_time": str(time_range.get('end_epoch', ''))
                    })
                else:
                    context.update({
                        "time_range_context": f"Time Range: {time_range['start']} to {time_range['end']}",
                        "start_time": time_range['start'],
                        "end_time": time_range['end']
                    })
            
            chain = prompt | self.llm
            response = chain.invoke(context)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_json = json.loads(json_str)
                
                # Format output based on tool type
                query_params = self._format_query_params(tool, parsed_json, has_time, time_range)
                
                return ToolQuery(
                    service_type=tool,
                    query_params=query_params,
                    search_context=f"Generated for {tool} from: {query}"
                )
        
        except Exception as e:
            logger.error(f"❌ Failed to generate query for {tool}: {e}")
        
        return None
    

    def process(self, user_query: str) -> Dict:
        """
        Main processing method that returns structured JSON with tool-specific queries
        """
        try:
            
            # Step 1: Calculate overall confidence
            confidence = self._calculate_confidence(user_query)
            
            # Step 2: Check if query is searchable
            is_searchable = confidence > self.confidence_threshold
            
            if not is_searchable:
                return {
                    "dataCollectionNeeded": False
                }
            
            # Step 3: Detect time mentions
            has_time, time_range = self._detect_time_mentions(user_query)
            
            # Step 4: Determine mentioned tools
            mentioned_tools = self._determine_mentioned_tools(user_query)
            
            # Step 5: Determine target tools for query generation
            if mentioned_tools:
                target_tools = mentioned_tools
            else:
                # If no specific tools mentioned, use all available tools
                target_tools = list(self.tool_config.CONFIG.keys())

            
            # Step 6: Generate tool-specific queries
            tool_queries = []
            for tool in target_tools:
                tool_query = self._generate_tool_query(tool, user_query, has_time, time_range)
                if tool_query:
                    tool_queries.append(tool_query)
            
            # Step 7: Create final result
            result = {
                "dataCollectionNeeded": len(tool_queries) > 0,
                "answer": {
                    "service_queries": [
                        {
                            "service_type": tq.service_type,
                            "query_params": tq.query_params
                        }
                        for tq in tool_queries
                    ]
                },
            }
            
            print("final result")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return {
                "overall_confidence": 0.0,
                "is_searchable": False,
                "has_time_mention": False,
                "mentioned_tools": [],
                "tool_queries": [],
                "raw_query": user_query,
                "error": f"Processing failed: {str(e)}"
            }
