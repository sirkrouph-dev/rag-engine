<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h1 class="text-3xl font-bold text-gray-900">Pipeline Management</h1>
      <p class="mt-2 text-gray-600">
        Build and manage your RAG pipeline and orchestrator
      </p>
    </div>

    <!-- Pipeline Status -->
    <div class="card p-6">
      <div class="flex items-center justify-between mb-4">
        <h2 class="text-lg font-semibold text-gray-900">Pipeline Status</h2>
        <button 
          @click="refreshStatus"
          :disabled="systemStore.isLoading"
          class="btn btn-sm btn-secondary"
        >
          <ArrowPathIcon class="w-4 h-4 mr-2" />
          Refresh
        </button>
      </div>
      
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="text-center p-4 rounded-lg border">
          <div 
            :class="[
              'w-12 h-12 mx-auto rounded-full flex items-center justify-center mb-2',
              systemStore.isPipelineBuilt ? 'bg-success-100' : 'bg-yellow-100'
            ]"
          >
            <CogIcon 
              :class="[
                'w-6 h-6',
                systemStore.isPipelineBuilt ? 'text-success-600' : 'text-yellow-600'
              ]" 
            />
          </div>
          <div class="text-sm font-medium text-gray-900">Pipeline</div>
          <div 
            :class="[
              'text-xs mt-1',
              systemStore.isPipelineBuilt ? 'text-success-600' : 'text-yellow-600'
            ]"
          >
            {{ systemStore.isPipelineBuilt ? 'Built' : 'Not Built' }}
          </div>
        </div>
        
        <div class="text-center p-4 rounded-lg border">
          <div 
            :class="[
              'w-12 h-12 mx-auto rounded-full flex items-center justify-center mb-2',
              systemStore.orchestratorStatus?.status === 'active' ? 'bg-success-100' : 'bg-gray-100'
            ]"
          >
            <ServerIcon 
              :class="[
                'w-6 h-6',
                systemStore.orchestratorStatus?.status === 'active' ? 'text-success-600' : 'text-gray-600'
              ]" 
            />
          </div>
          <div class="text-sm font-medium text-gray-900">Orchestrator</div>
          <div 
            :class="[
              'text-xs mt-1',
              systemStore.orchestratorStatus?.status === 'active' ? 'text-success-600' : 'text-gray-600'
            ]"
          >
            {{ systemStore.orchestratorStatus?.status || 'Unknown' }}
          </div>
        </div>
        
        <div class="text-center p-4 rounded-lg border">
          <div class="w-12 h-12 mx-auto rounded-full bg-blue-100 flex items-center justify-center mb-2">
            <Squares2X2Icon class="w-6 h-6 text-blue-600" />
          </div>
          <div class="text-sm font-medium text-gray-900">Components</div>
          <div class="text-xs mt-1 text-blue-600">
            {{ componentsCount }} available
          </div>
        </div>
      </div>
    </div>

    <!-- Actions -->
    <div class="card p-6">
      <h2 class="text-lg font-semibold text-gray-900 mb-4">Actions</h2>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div class="space-y-4">
          <div>
            <h3 class="text-sm font-medium text-gray-900 mb-2">Pipeline Operations</h3>
            <div class="space-y-2">
              <button 
                @click="buildPipeline"
                :disabled="systemStore.isLoading"
                class="w-full btn btn-primary justify-start"
              >
                <PlayIcon class="w-4 h-4 mr-2" />
                Build Pipeline
              </button>
              <p class="text-xs text-gray-600">
                Build the RAG pipeline with current configuration
              </p>
            </div>
          </div>
        </div>
        
        <div class="space-y-4">
          <div>
            <h3 class="text-sm font-medium text-gray-900 mb-2">Orchestrator Operations</h3>
            <div class="space-y-2">
              <button 
                @click="rebuildOrchestrator"
                :disabled="systemStore.isLoading"
                class="w-full btn btn-secondary justify-start"
              >
                <ArrowPathIcon class="w-4 h-4 mr-2" />
                Rebuild Orchestrator
              </button>
              <p class="text-xs text-gray-600">
                Rebuild the orchestrator with current configuration
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Components Details -->
    <div class="card p-6" v-if="systemStore.components">
      <h2 class="text-lg font-semibold text-gray-900 mb-4">Available Components</h2>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div 
          v-for="(componentList, type) in systemStore.components" 
          :key="type"
          class="border rounded-lg p-4"
        >
          <h3 class="font-medium text-gray-900 capitalize mb-2">{{ type }}</h3>
          <div class="space-y-1">
            <div 
              v-for="component in componentList" 
              :key="component"
              class="text-sm text-gray-600 font-mono bg-gray-50 px-2 py-1 rounded"
            >
              {{ component }}
            </div>
          </div>
          <div class="mt-2 text-xs text-gray-500">
            {{ componentList.length }} available
          </div>
        </div>
      </div>
    </div>

    <!-- Build Logs -->
    <div v-if="buildLogs.length > 0" class="card p-6">
      <h2 class="text-lg font-semibold text-gray-900 mb-4">Build Logs</h2>
      <div class="bg-gray-900 text-green-400 font-mono text-sm p-4 rounded-lg max-h-64 overflow-y-auto">
        <div v-for="(log, index) in buildLogs" :key="index" class="whitespace-pre-wrap">
          {{ log }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { 
  CogIcon, 
  ServerIcon, 
  Squares2X2Icon,
  PlayIcon, 
  ArrowPathIcon 
} from '@heroicons/vue/24/outline'
import { useSystemStore } from '../stores/system'

const systemStore = useSystemStore()

const buildLogs = ref([])

const componentsCount = computed(() => {
  if (!systemStore.components) return 0
  return Object.values(systemStore.components).reduce((total, list) => total + list.length, 0)
})

async function refreshStatus() {
  await systemStore.initializeStore()
}

async function buildPipeline() {
  buildLogs.value = []
  addLog('Starting pipeline build...')
  
  try {
    const result = await systemStore.buildPipeline()
    addLog('✅ Pipeline built successfully')
    if (result.documents) {
      addLog(`📚 Processed ${result.documents} documents`)
    }
    if (result.chunks) {
      addLog(`📄 Created ${result.chunks} chunks`)
    }
  } catch (error) {
    addLog(`❌ Build failed: ${error.message}`)
  }
}

async function rebuildOrchestrator() {
  buildLogs.value = []
  addLog('Rebuilding orchestrator...')
  
  try {
    const result = await systemStore.rebuildOrchestrator()
    addLog('✅ Orchestrator rebuilt successfully')
  } catch (error) {
    addLog(`❌ Rebuild failed: ${error.message}`)
  }
}

function addLog(message) {
  const timestamp = new Date().toLocaleTimeString()
  buildLogs.value.push(`[${timestamp}] ${message}`)
}

onMounted(() => {
  refreshStatus()
})
</script>
