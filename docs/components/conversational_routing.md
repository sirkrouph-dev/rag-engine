# Advanced Conversational Routing System

The RAG Engine features an advanced conversational routing system that addresses one of the biggest challenges in building human-like conversational AI: knowing when and how to use RAG retrieval versus other response strategies.

## Overview

Traditional RAG systems often struggle with:
- **Over-retrieval**: Trying to use RAG for greetings, small talk, or out-of-context queries
- **Under-retrieval**: Missing opportunities to provide factual information from documents
- **Context blindness**: Not maintaining conversation flow and context
- **One-size-fits-all responses**: Using the same approach for all query types

Our conversational routing system solves these problems with a sophisticated **3-stage LLM routing pipeline**:

## 🧠 Three-Stage Routing Pipeline

### Stage 1: Topic Analysis
- **Detects the main topic** of the user's query
- **Assesses breadth**: Is the query too general/broad to answer specifically?
- **Evaluates ambiguity**: Could the query be interpreted multiple ways?
- **Confidence scoring**: How certain are we about the analysis?

### Stage 2: Query Classification & Metadata Extraction
- **Classifies into categories**: RAG-requiring, conversational, out-of-context, etc.
- **Extracts entities**: People, places, technical terms, concepts
- **Identifies intent**: What is the user really trying to accomplish?
- **Gathers metadata**: Domain, expertise level, urgency, context requirements

### Stage 3: Response Strategy Selection & Generation
- **Routes to appropriate strategy** based on classification
- **Maintains conversation context** and chain-of-thought reasoning
- **Generates human-like responses** tailored to the query type
- **Provides polite rejections** for out-of-scope requests

## 📋 Query Categories

The system classifies queries into these categories:

### RAG-Requiring Categories
- **`rag_factual`**: Needs factual information lookup from documents
- **`rag_analytical`**: Needs analysis/reasoning based on retrieved information

### Conversational Categories
- **`greeting`**: Hello, hi, good morning, etc.
- **`goodbye`**: Bye, farewell, see you later, etc.
- **`gratitude`**: Thank you, thanks, I appreciate it, etc.
- **`clarification`**: Asking for clarification on previous responses
- **`follow_up`**: Following up on previous conversation topics
- **`small_talk`**: Casual conversation, weather, personal topics
- **`help_request`**: Asking for help, guidance, or instructions

### Boundary Categories
- **`out_of_context`**: Outside the domain/scope of available knowledge

## 🎯 Response Strategies

Based on classification, the system selects from these strategies:

### RAG Retrieval
- Full RAG pipeline with document retrieval
- Enhanced with routing context and reasoning chain
- Smart citation and source attribution

### Contextual Chat
- Conversational responses using conversation history
- No document retrieval needed
- Maintains rapport and engagement

### Simple Response
- Direct responses for greetings, gratitude, etc.
- Quick, friendly acknowledgments

### Polite Rejection
- Diplomatic responses for out-of-scope queries
- Suggests alternatives when possible
- Maintains helpful tone while setting boundaries

### Clarification Request
- Asks targeted questions to narrow down broad/ambiguous queries
- Helps users refine their questions for better answers

## 🔧 Configuration

### Basic Configuration
```json
{
  "prompting": {
    "type": "conversational_rag",
    "enable_routing": true,
    "fallback_to_simple": true,
    "system_prompt": "You are an intelligent AI assistant.",
    "routing_config": {
      "max_conversation_history": 10,
      "enable_reasoning_chain": true
    }
  }
}
```

### Advanced Configuration
```json
{
  "prompting": {
    "type": "conversational_rag",
    "enable_routing": true,
    "fallback_to_simple": false,
    "routing_config": {
      "topic_analysis_temperature": 0.1,
      "classification_temperature": 0.1,
      "response_temperature": 0.7,
      "max_conversation_history": 20,
      "confidence_threshold": 0.8,
      "enable_clarification_requests": true,
      "context_optimization": {
        "relevance_filtering": true,
        "diversity_enhancement": true,
        "redundancy_removal": true
      }
    },
    "domain_config": {
      "domain_name": "Technical Documentation",
      "allowed_topics": ["software", "technology", "APIs"],
      "rejection_style": "professional",
      "default_expertise": "intermediate"
    }
  }
}
```

## 💬 Conversation Flow Examples

### Example 1: Greeting → Question → Follow-up
```
User: Hi there!
Assistant: Hello! I'm here to help you with any questions. How can I assist you today?
[Strategy: simple_response, Category: greeting]

User: What is machine learning?
Assistant: [Retrieves relevant documents about ML and provides comprehensive answer]
[Strategy: rag_retrieval, Category: rag_factual]

User: Can you give me some examples?
Assistant: [Uses conversation context to provide ML examples, building on previous answer]
[Strategy: contextual_chat, Category: follow_up]
```

### Example 2: Broad Query → Clarification → Focused Answer
```
User: Tell me about AI
Assistant: That's a fascinating and broad topic! To give you the most helpful answer, 
          what aspect of AI interests you most? Are you looking for:
          - Basic concepts and definitions
          - Specific applications or use cases  
          - Technical implementation details
[Strategy: clarification_request, Category: rag_factual, Reason: too_broad]

User: I want to understand how neural networks work
Assistant: [Retrieves documents about neural networks and provides focused explanation]
[Strategy: rag_retrieval, Category: rag_factual]
```

### Example 3: Out of Context → Polite Rejection
```
User: What's the weather like today?
Assistant: I'd love to help, but I don't have access to current weather information. 
          However, I'm great at answering questions about our technical documentation 
          and software topics. Is there anything in those areas I can help you with?
[Strategy: polite_rejection, Category: out_of_context]
```

## 🔄 Conversation Context Management

The system maintains rich conversation context:

### Conversation History
- Stores user queries and assistant responses
- Tracks metadata from each exchange
- Maintains conversation flow and coherence

### Chain of Thought
- Records reasoning steps for each query
- Enables debugging and transparency
- Improves response quality through accumulated insights

### User Profile
- Learns apparent expertise level
- Tracks topics of interest
- Adapts responses over time

### Accumulated Metadata
- Builds understanding of user's domain/context
- Informs future routing decisions
- Enables personalized responses

## 🎛️ Template System

The routing system uses specialized templates for each stage:

### Routing Templates
- `topic_analysis_template.txt` - Analyzes topic and breadth/ambiguity
- `query_classification_template.txt` - Classifies and extracts metadata
- `rag_response_template.txt` - RAG responses with routing context
- `contextual_chat_template.txt` - Conversational responses
- `polite_rejection_template.txt` - Diplomatic out-of-scope responses
- `clarification_request_template.txt` - Targeted clarification questions

### Template Features
- **Variable substitution**: Query, context, metadata, reasoning chain
- **Structured outputs**: JSON format for analysis stages
- **Tone adaptation**: Professional, friendly, diplomatic as needed
- **Context awareness**: Uses conversation history appropriately

## 📊 Monitoring & Insights

### Routing Analytics
- Track routing decision accuracy
- Monitor conversation flow patterns
- Measure response quality and user satisfaction

### Debugging Support
- Access to full reasoning chain
- Confidence scores for each decision
- Category and strategy explanations

### Performance Metrics
- Response time by strategy type
- Classification accuracy
- User engagement patterns

## 🚀 Getting Started

### 1. Basic Setup
```python
from rag_engine.core.conversational_integration import ConversationalRAGPrompter

# Create prompter with routing
prompter = ConversationalRAGPrompter({
    "enable_routing": True,
    "routing_config": {
        "max_conversation_history": 10
    }
})

# Inject RAG components
prompter.set_dependencies(llm=llm, retriever=retriever, vectorstore=vectorstore)
```

### 2. Process Queries
```python
# The prompter automatically routes queries
prompt = prompter.build_prompt({
    "query": "Hello, can you help me understand transformers?",
    "session_id": "user123"
})
```

### 3. Get Routing Insights
```python
# Analyze without generating response
insights = prompter.get_routing_insights(
    "What is the best approach?", 
    session_id="user123"
)
print(f"Category: {insights['category']}")
print(f"Strategy: {insights['strategy']}")
print(f"Confidence: {insights['confidence']}")
```

## 🎯 Benefits

### For Users
- **Natural conversation flow** that feels human-like
- **Appropriate responses** for different types of queries
- **Helpful guidance** when questions are unclear
- **Polite handling** of out-of-scope requests

### for Developers
- **Reduced prompt engineering** burden
- **Automatic query routing** based on content
- **Rich debugging information** and analytics
- **Flexible configuration** for different domains

### For Organizations
- **Professional user experience** that maintains boundaries
- **Efficient resource usage** (no unnecessary RAG calls)
- **Consistent brand voice** across interaction types
- **Scalable conversation management**

## 🔮 Advanced Features

### Multi-domain Support
- Configure different routing rules for different domains
- Domain-specific rejection messages and redirection
- Cross-domain query handling

### Learning and Adaptation
- User preference learning over time
- Conversation pattern recognition
- Automatic routing improvement

### Integration with External Systems
- Connect to live chat handoff systems
- Integration with ticketing systems for complex queries
- API endpoints for routing analytics

---

The conversational routing system transforms your RAG engine from a simple document Q&A system into a sophisticated conversational AI that knows how to handle the full spectrum of user interactions professionally and intelligently.
