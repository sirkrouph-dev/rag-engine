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
  }
}
