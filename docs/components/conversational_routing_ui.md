# Conversational Routing UI Integration Guide

## Overview

The RAG Engine now features a comprehensive web-based interface for managing and configuring the advanced conversational routing system. This UI allows users to modify prompt templates, configure routing parameters, test routing decisions, and monitor analytics.

## Features

### 🎛️ Template Management
- **View all routing templates** in an organized interface
- **Edit templates in real-time** with syntax highlighting
- **Backup and restore** templates automatically
- **Preview template content** before making changes
- **Validate template syntax** and structure

### ⚙️ Configuration Management
- **Enable/disable routing** with toggle switches
- **Configure temperature settings** for each routing stage
- **Set conversation history limits** and thresholds
- **Customize system prompts** and domain settings
- **Real-time validation** of configuration changes

### 🧪 Testing Interface
- **Test routing decisions** with sample queries
- **View detailed routing insights** including confidence scores
- **Quick test examples** for common scenarios
- **Routing strategy explanations** and recommendations
- **Performance metrics** and timing information

### 📊 Analytics Dashboard
- **Monitor routing performance** with visual charts
- **Track strategy distribution** across queries
- **Analyze category patterns** and trends
- **Template usage statistics** and optimization insights
- **Confidence score tracking** over time

## UI Structure

### Main Navigation
The routing interface is accessible via the main navigation menu:
- **Dashboard** → **Routing** tab
- Direct URL: `/routing`

### Tab-Based Interface
The routing management interface is organized into four main sections:

#### 1. Configuration Tab
```
📋 Routing Configuration
├── Enable/Disable Routing
├── Fallback Settings
├── System Prompt Editor
├── Temperature Controls
│   ├── Topic Analysis (0.0-2.0)
│   ├── Classification (0.0-2.0)
│   └── Response (0.0-2.0)
└── Advanced Settings
    ├── Max Conversation History
    └── Confidence Threshold
```

#### 2. Templates Tab
```
📝 Template Manager
├── Template List
│   ├── Topic Analysis Template
│   ├── Query Classification Template
│   ├── RAG Response Template
│   ├── Contextual Chat Template
│   ├── Polite Rejection Template
│   └── Clarification Request Template
├── Template Editor (Modal)
│   ├── Syntax Highlighting
│   ├── Variable Preview
│   └── Backup Management
└── Template Preview
```

#### 3. Testing Tab
```
🧪 Routing Tester
├── Query Input Field
├── Quick Test Examples
│   ├── Greetings
│   ├── Factual Questions
│   ├── Out-of-Context Queries
│   └── Ambiguous Requests
├── Test Results Display
│   ├── Classification Results
│   ├── Strategy Selection
│   ├── Confidence Scores
│   └── Reasoning Chain
└── Strategy Explanations
```

#### 4. Analytics Tab
```
📊 Routing Analytics
├── Overview Statistics
│   ├── Total Queries Processed
│   ├── Average Confidence
│   └── Most Common Strategy
├── Strategy Distribution Chart
├── Category Distribution Chart
├── Template Usage Statistics
└── Performance Metrics
```

## API Endpoints

The UI communicates with the backend through the following REST API endpoints:

### Template Management
```
GET    /api/routing/templates           # Get all templates
GET    /api/routing/templates/{name}    # Get specific template
PUT    /api/routing/templates/{name}    # Update template
```

### Configuration Management
```
GET    /api/routing/config              # Get current config
PUT    /api/routing/config              # Update config
```

### Testing Interface
```
POST   /api/routing/test                # Test routing with query
```

### Analytics
```
GET    /api/routing/analytics           # Get analytics data
```

## Usage Instructions

### 1. Accessing the Interface

1. **Navigate to Routing**: Click the "Routing" tab in the main navigation
2. **Check Status**: Verify that conversational routing is enabled
3. **Select Tab**: Choose the appropriate tab for your task

### 2. Configuring Routing

1. **Go to Configuration Tab**
2. **Enable Routing**: Toggle the "Enable Conversational Routing" switch
3. **Set Parameters**:
   - Adjust temperature settings for each stage
   - Configure conversation history limits
   - Set confidence thresholds
4. **Update System Prompt**: Customize the base system prompt
5. **Save Changes**: Click "Save Configuration"

### 3. Managing Templates

1. **Go to Templates Tab**
2. **Select Template**: Click "Edit" on any template
3. **Modify Content**: Edit the template in the modal editor
4. **Preview Changes**: Review the template content
5. **Save Template**: Click "Save Template" to apply changes

**Template Variables Available:**
- `{query}` - User's input query
- `{context}` - Conversation context
- `{metadata}` - Extracted metadata
- `{reasoning}` - Chain-of-thought reasoning
- `{confidence}` - Confidence scores

### 4. Testing Routing

1. **Go to Testing Tab**
2. **Enter Query**: Type a test query or select from examples
3. **Run Test**: Click "Test Routing"
4. **Review Results**:
   - Check the selected category and strategy
   - Review confidence scores
   - Read the reasoning explanation
5. **Iterate**: Try different queries to test various scenarios

**Recommended Test Queries:**
- "Hello!" (Expected: Simple Response)
- "What is machine learning?" (Expected: RAG Retrieval)
- "What's the weather today?" (Expected: Polite Rejection)
- "Tell me about AI" (Expected: Clarification Request)

### 5. Monitoring Analytics

1. **Go to Analytics Tab**
2. **Review Overview**: Check total queries and average confidence
3. **Analyze Distribution**: Review strategy and category charts
4. **Monitor Performance**: Track template usage and patterns
5. **Refresh Data**: Click "Refresh" for latest statistics

## Best Practices

### Template Editing
- **Always backup**: Templates are automatically backed up before editing
- **Test changes**: Use the testing interface after template modifications
- **Maintain consistency**: Keep similar formatting across templates
- **Document variables**: Include comments for custom variables

### Configuration Tuning
- **Start conservative**: Begin with default temperature settings
- **Monitor confidence**: Aim for average confidence > 80%
- **Gradual adjustments**: Make small incremental changes
- **Test thoroughly**: Validate changes with comprehensive testing

### Performance Optimization
- **Monitor analytics**: Regular check routing patterns
- **Balance strategies**: Ensure appropriate distribution across strategies
- **Optimize templates**: Remove unused variables and streamline content
- **Track confidence**: Investigate low-confidence classifications

## Troubleshooting

### Common Issues

#### Templates Not Loading
- **Check file permissions**: Ensure templates directory is readable
- **Verify file format**: Templates must be `.txt` files
- **Refresh interface**: Click the "Refresh" button in templates tab

#### Configuration Not Saving
- **Check validation**: Ensure all required fields are filled
- **Verify permissions**: Check write permissions for config files
- **Restart service**: Configuration changes may require server restart

#### Testing Failures
- **Check LLM connection**: Ensure LLM service is available
- **Verify configuration**: Confirm routing is enabled
- **Review logs**: Check server logs for error details

#### Analytics Not Updating
- **Process queries**: Analytics require actual query processing
- **Refresh data**: Click refresh button to update statistics
- **Check tracking**: Verify analytics collection is enabled

## Advanced Features

### Custom Template Variables
Add custom variables to templates for specific use cases:

```javascript
// In template content
You are working in the {domain} domain.
Current user expertise: {user_expertise}
Conversation context: {conversation_summary}
```

### Configuration Profiles
Save and load different configuration profiles:

```json
{
  "profiles": {
    "conservative": {
      "topic_analysis_temperature": 0.1,
      "classification_temperature": 0.1,
      "confidence_threshold": 0.9
    },
    "balanced": {
      "topic_analysis_temperature": 0.3,
      "classification_temperature": 0.2,
      "confidence_threshold": 0.8
    },
    "exploratory": {
      "topic_analysis_temperature": 0.5,
      "classification_temperature": 0.4,
      "confidence_threshold": 0.7
    }
  }
}
```

### Batch Testing
Test multiple queries simultaneously:

```javascript
const testQueries = [
  "Hello!",
  "What is machine learning?",
  "How's the weather?",
  "Tell me about AI"
];

// Process all queries and analyze patterns
```

## Integration with Existing Workflows

### Development Workflow
1. **Design templates** using the template manager
2. **Configure parameters** based on domain requirements
3. **Test thoroughly** with representative queries
4. **Monitor analytics** in production
5. **Iterate and optimize** based on performance data

### Production Monitoring
1. **Set up alerts** for low confidence scores
2. **Regular analytics review** (weekly/monthly)
3. **Template performance tracking**
4. **User feedback integration**

## Security Considerations

### Access Control
- **Authentication required**: All routing endpoints require authentication
- **Role-based permissions**: Restrict template editing to authorized users
- **Audit logging**: Track all configuration changes

### Data Privacy
- **Query logging**: Be mindful of sensitive data in test queries
- **Template content**: Avoid hardcoding sensitive information
- **Analytics anonymization**: Ensure user privacy in analytics data

## Performance Optimization

### UI Performance
- **Lazy loading**: Components load only when needed
- **Caching**: Template and configuration data cached locally
- **Debounced updates**: Prevent excessive API calls during editing

### Backend Performance
- **Template caching**: Templates cached in memory for quick access
- **Async processing**: Non-blocking operations for better responsiveness
- **Connection pooling**: Efficient database connections

## Future Enhancements

### Planned Features
- **Visual flow diagrams** showing routing decision trees
- **A/B testing capabilities** for different configurations
- **Machine learning insights** for automatic optimization
- **Integration with external monitoring tools**
- **Collaborative editing** for team-based template management

### API Extensions
- **Webhooks** for real-time notifications
- **Bulk operations** for batch template updates
- **Export/import** functionality for configuration backup
- **Advanced analytics** with custom metrics

## Support and Documentation

### Getting Help
- **In-app help**: Tooltips and contextual help throughout the interface
- **API documentation**: Detailed endpoint documentation
- **Example configurations**: Pre-built templates for common use cases
- **Community support**: Forums and discussion channels

### Additional Resources
- **Video tutorials**: Step-by-step configuration guides
- **Best practices guide**: Industry-specific recommendations
- **Troubleshooting FAQ**: Common issues and solutions
- **Performance tuning guide**: Optimization strategies

---

The conversational routing UI provides a powerful, user-friendly interface for managing one of the most sophisticated features of the RAG Engine. With comprehensive template management, real-time testing, and detailed analytics, users can fine-tune their conversational AI system for optimal performance and user experience.
