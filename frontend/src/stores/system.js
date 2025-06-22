import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '../services/api'

export const useSystemStore = defineStore('system', () => {
  // State
  const status = ref(null)
  const config = ref(null)
  const health = ref(null)
  const orchestratorStatus = ref(null)
  const components = ref(null)
  const isLoading = ref(false)
  const error = ref(null)

  // Computed
  const isHealthy = computed(() => {
    return health.value?.status === 'ok' || health.value?.status === 'healthy'
  })

  const isPipelineBuilt = computed(() => {
    return status.value?.pipeline_built === true
  })

  const systemHealth = computed(() => {
    if (!health.value) return 'unknown'
    return isHealthy.value ? 'healthy' : 'unhealthy'
  })

  // Actions
  async function fetchHealth() {
    try {
      isLoading.value = true
      error.value = null
      health.value = await api.getHealth()
    } catch (err) {
      error.value = err.message
      health.value = { status: 'error' }
    } finally {
      isLoading.value = false
    }
  }

  async function fetchStatus() {
    try {
      isLoading.value = true
      error.value = null
      status.value = await api.getStatus()
    } catch (err) {
      error.value = err.message
      status.value = null
    } finally {
      isLoading.value = false
    }
  }

  async function fetchConfig() {
    try {
      isLoading.value = true
      error.value = null
      config.value = await api.getConfig()
    } catch (err) {
      error.value = err.message
      config.value = null
    } finally {
      isLoading.value = false
    }
  }

  async function fetchOrchestratorStatus() {
    try {
      orchestratorStatus.value = await api.getOrchestratorStatus()
    } catch (err) {
      error.value = err.message
      orchestratorStatus.value = null
    }
  }

  async function fetchComponents() {
    try {
      components.value = await api.getComponents()
    } catch (err) {
      error.value = err.message
      components.value = null
    }
  }

  async function buildPipeline() {
    try {
      isLoading.value = true
      error.value = null
      const result = await api.buildPipeline()
      await fetchStatus() // Refresh status after build
      return result
    } catch (err) {
      error.value = err.message
      throw err
    } finally {
      isLoading.value = false
    }
  }

  async function rebuildOrchestrator() {
    try {
      isLoading.value = true
      error.value = null
      const result = await api.rebuildOrchestrator()
      await Promise.all([
        fetchStatus(),
        fetchOrchestratorStatus(),
        fetchComponents()
      ])
      return result
    } catch (err) {
      error.value = err.message
      throw err
    } finally {
      isLoading.value = false
    }
  }

  async function initializeStore() {
    await Promise.all([
      fetchHealth(),
      fetchStatus(),
      fetchConfig(),
      fetchOrchestratorStatus(),
      fetchComponents()
    ])
  }

  return {
    // State
    status,
    config,
    health,
    orchestratorStatus,
    components,
    isLoading,
    error,
    
    // Computed
    isHealthy,
    isPipelineBuilt,
    systemHealth,
    
    // Actions
    fetchHealth,
    fetchStatus,
    fetchConfig,
    fetchOrchestratorStatus,
    fetchComponents,
    buildPipeline,
    rebuildOrchestrator,
    initializeStore
  }
})
