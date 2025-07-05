import axios from 'axios'

// Create axios instance with base configuration
const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // You can add auth headers here if needed
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    // Handle common errors
    if (error.response?.status === 500) {
      console.error('Server error:', error.response.data)
    } else if (error.code === 'ECONNABORTED') {
      console.error('Request timeout')
    }
    return Promise.reject(error)
  }
)

export default {
  // System endpoints
  async getHealth() {
    const response = await api.get('/health')
    return response.data
  },

  async getStatus() {
    const response = await api.get('/status')
    return response.data
  },

  async getConfig() {
    const response = await api.get('/config')
    return response.data
  },

  // Chat endpoints
  async sendMessage(query, sessionId = null) {
    const response = await api.post('/chat', {
      query,
      session_id: sessionId
    })
    return response.data
  },

  // Pipeline endpoints
  async buildPipeline() {
    const response = await api.post('/build')
    return response.data
  },

  // Document endpoints
  async getDocuments() {
    const response = await api.get('/documents')
    return response.data
  },

  async getChunks() {
    const response = await api.get('/chunks')
    return response.data
  },

  // Orchestrator endpoints
  async getOrchestratorStatus() {
    const response = await api.get('/orchestrator/status')
    return response.data
  },

  async getComponents() {
    const response = await api.get('/orchestrator/components')
    return response.data
  },

  async rebuildOrchestrator() {
    const response = await api.post('/orchestrator/rebuild')
    return response.data
  },

  async getComponentStatus(componentType) {
    const response = await api.get(`/orchestrator/components/${componentType}`)
    return response.data
  },

  // AI Assistant endpoints
  async askAssistant(question, context = null, model = 'phi3.5:latest') {
    const response = await api.post('/ai-assistant', {
      question,
      context,
      model
    })
    return response.data
  },

  // Stack Management endpoints
  async configureStack(stackType, customRequirements = null, configOverrides = null) {
    const response = await api.post('/stack/configure', {
      stack_type: stackType,
      custom_requirements: customRequirements,
      config_overrides: configOverrides
    })
    return response.data
  },

  async analyzeStack() {
    const response = await api.get('/stack/analyze')
    return response.data
  },

  async auditDependencies() {
    const response = await api.get('/stack/audit')
    return response.data
  },

  // Conversational Routing endpoints
  async getRoutingTemplates() {
    const response = await api.get('/routing/templates')
    return response.data
  },

  async getRoutingTemplate(templateName) {
    const response = await api.get(`/routing/templates/${templateName}`)
    return response.data
  },

  async updateRoutingTemplate(templateName, templateData) {
    const response = await api.put(`/routing/templates/${templateName}`, templateData)
    return response.data
  },

  async getRoutingConfig() {
    const response = await api.get('/routing/config')
    return response.data
  },

  async updateRoutingConfig(configData) {
    const response = await api.put('/routing/config', configData)
    return response.data
  },

  async testRouting(testData) {
    const response = await api.post('/routing/test', testData)
    return response.data
  },

  async getRoutingAnalytics() {
    const response = await api.get('/routing/analytics')
    return response.data
  }
}
