<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h1 class="text-3xl font-bold text-gray-900">System Information</h1>
      <p class="mt-2 text-gray-600">
        Detailed system status and configuration information
      </p>
    </div>

    <!-- System Health -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <StatCard
        title="API Health"
        :value="systemStore.health?.status || 'Unknown'"
        :status="systemStore.systemHealth"
        icon="ServerIcon"
      />
      <StatCard
        title="Pipeline"
        :value="systemStore.isPipelineBuilt ? 'Built' : 'Not Built'"
        :status="systemStore.isPipelineBuilt ? 'success' : 'warning'"
        icon="CogIcon"
      />
      <StatCard
        title="Orchestrator"
        :value="systemStore.orchestratorStatus?.status || 'Unknown'"
        :status="systemStore.orchestratorStatus?.status === 'active' ? 'success' : 'warning'"
        icon="Squares2X2Icon"
      />
      <StatCard
        title="Components"
        :value="componentsCount"
        status="info"
        icon="PuzzlePieceIcon"
      />
    </div>

    <!-- Configuration Details -->
    <div class="card p-6">
      <h2 class="text-lg font-semibold text-gray-900 mb-4">Configuration</h2>
      <div v-if="systemStore.config" class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 class="text-sm font-medium text-gray-900 mb-3">Core Components</h3>
          <div class="space-y-2">
            <ConfigItem 
              label="Embedding Provider" 
              :value="systemStore.config.embedding_provider" 
            />
            <ConfigItem 
              label="Vector Store" 
              :value="systemStore.config.vectorstore_provider" 
            />
            <ConfigItem 
              label="LLM Provider" 
              :value="systemStore.config.llm_provider" 
            />
            <ConfigItem 
              label="Chunking Method" 
              :value="systemStore.config.chunking_method" 
            />
          </div>
        </div>
        <div>
          <h3 class="text-sm font-medium text-gray-900 mb-3">Settings</h3>
          <div class="space-y-2">
            <ConfigItem 
              label="Document Count" 
              :value="systemStore.config.documents" 
            />
            <ConfigItem 
              label="Retrieval Top-K" 
              :value="systemStore.config.retrieval_top_k" 
            />
          </div>
        </div>
      </div>
      <div v-else class="text-center py-8 text-gray-500">
        No configuration data available
      </div>
    </div>

    <!-- Orchestrator Details -->
    <div class="card p-6">
      <h2 class="text-lg font-semibold text-gray-900 mb-4">Orchestrator Status</h2>
      <div v-if="systemStore.orchestratorStatus" class="space-y-4">
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="text-center p-4 bg-gray-50 rounded-lg">
            <div class="text-2xl font-bold text-gray-900">
              {{ systemStore.orchestratorStatus.type || 'Default' }}
            </div>
            <div class="text-sm text-gray-600">Orchestrator Type</div>
          </div>
          <div class="text-center p-4 bg-gray-50 rounded-lg">
            <div 
              :class="[
                'text-2xl font-bold capitalize',
                systemStore.orchestratorStatus.status === 'active' ? 'text-success-600' : 'text-yellow-600'
              ]"
            >
              {{ systemStore.orchestratorStatus.status }}
            </div>
            <div class="text-sm text-gray-600">Status</div>
          </div>
          <div class="text-center p-4 bg-gray-50 rounded-lg">
            <div class="text-2xl font-bold text-gray-900">
              {{ componentsCount }}
            </div>
            <div class="text-sm text-gray-600">Available Components</div>
          </div>
        </div>
      </div>
      <div v-else class="text-center py-8 text-gray-500">
        Orchestrator not initialized
      </div>
    </div>

    <!-- Components Registry -->
    <div class="card p-6" v-if="systemStore.components">
      <h2 class="text-lg font-semibold text-gray-900 mb-4">Component Registry</h2>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div 
          v-for="(componentList, type) in systemStore.components" 
          :key="type"
          class="border rounded-lg p-4 bg-gray-50"
        >
          <div class="flex items-center space-x-2 mb-3">
            <component :is="getComponentIcon(type)" class="w-5 h-5 text-gray-600" />
            <h3 class="font-medium text-gray-900 capitalize">{{ type }}</h3>
          </div>
          
          <div class="space-y-1 mb-3">
            <div 
              v-for="component in componentList" 
              :key="component"
              class="text-sm text-gray-700 font-mono bg-white px-2 py-1 rounded border"
            >
              {{ component }}
            </div>
          </div>
          
          <div class="text-xs text-gray-500">
            {{ componentList.length }} {{ componentList.length === 1 ? 'component' : 'components' }}
          </div>
        </div>
      </div>
    </div>

    <!-- API Endpoints -->
    <div class="card p-6">
      <h2 class="text-lg font-semibold text-gray-900 mb-4">API Endpoints</h2>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div 
          v-for="endpoint in apiEndpoints" 
          :key="endpoint.path"
          class="flex items-center justify-between p-3 border rounded-lg"
        >
          <div>
            <div class="flex items-center space-x-2">
              <span 
                :class="[
                  'px-2 py-1 text-xs font-medium rounded',
                  endpoint.method === 'GET' ? 'bg-blue-100 text-blue-800' :
                  endpoint.method === 'POST' ? 'bg-green-100 text-green-800' :
                  'bg-gray-100 text-gray-800'
                ]"
              >
                {{ endpoint.method }}
              </span>
              <span class="text-sm font-mono text-gray-900">{{ endpoint.path }}</span>
            </div>
            <div class="text-xs text-gray-600 mt-1">{{ endpoint.description }}</div>
          </div>
          <button 
            @click="testEndpoint(endpoint)"
            class="btn btn-sm btn-secondary"
            :disabled="endpoint.method !== 'GET'"
          >
            Test
          </button>
        </div>
      </div>
    </div>

    <!-- Test Results -->
    <div v-if="testResults.length > 0" class="card p-6">
      <h2 class="text-lg font-semibold text-gray-900 mb-4">Test Results</h2>
      <div class="space-y-2 max-h-64 overflow-y-auto">
        <div 
          v-for="(result, index) in testResults" 
          :key="index"
          :class="[
            'p-3 rounded text-sm font-mono',
            result.success ? 'bg-green-50 text-green-900' : 'bg-red-50 text-red-900'
          ]"
        >
          <div class="font-medium">{{ result.endpoint }}</div>
          <div class="text-xs mt-1">{{ result.message }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { 
  ServerIcon, 
  CogIcon, 
  Squares2X2Icon,
  PuzzlePieceIcon,
  DocumentTextIcon,
  CircleStackIcon,
  CpuChipIcon,
  ChatBubbleLeftRightIcon
} from '@heroicons/vue/24/outline'
import { useSystemStore } from '../stores/system'
import StatCard from '../components/StatCard.vue'
import ConfigItem from '../components/ConfigItem.vue'
import api from '../services/api'

const systemStore = useSystemStore()
const testResults = ref([])

const componentsCount = computed(() => {
  if (!systemStore.components) return 0
  return Object.values(systemStore.components).reduce((total, list) => total + list.length, 0)
})

const apiEndpoints = [
  { path: '/health', method: 'GET', description: 'Health check' },
  { path: '/status', method: 'GET', description: 'System status' },
  { path: '/config', method: 'GET', description: 'Configuration' },
  { path: '/documents', method: 'GET', description: 'List documents' },
  { path: '/chunks', method: 'GET', description: 'List chunks' },
  { path: '/orchestrator/status', method: 'GET', description: 'Orchestrator status' },
  { path: '/orchestrator/components', method: 'GET', description: 'Available components' },
  { path: '/build', method: 'POST', description: 'Build pipeline' },
  { path: '/chat', method: 'POST', description: 'Chat endpoint' },
  { path: '/orchestrator/rebuild', method: 'POST', description: 'Rebuild orchestrator' }
]

function getComponentIcon(type) {
  const iconMap = {
    'chunker': DocumentTextIcon,
    'embedder': CircleStackIcon,
    'vectorstore': Squares2X2Icon,
    'llm': CpuChipIcon,
    'retriever': ChatBubbleLeftRightIcon,
    'loader': DocumentTextIcon
  }
  return iconMap[type] || PuzzlePieceIcon
}

async function testEndpoint(endpoint) {
  if (endpoint.method !== 'GET') return
  
  try {
    const response = await fetch(`/api${endpoint.path}`)
    const data = await response.json()
    
    testResults.value.unshift({
      endpoint: `${endpoint.method} ${endpoint.path}`,
      success: response.ok,
      message: response.ok ? 'Success' : `Error: ${response.status}`,
      timestamp: new Date()
    })
  } catch (error) {
    testResults.value.unshift({
      endpoint: `${endpoint.method} ${endpoint.path}`,
      success: false,
      message: `Network error: ${error.message}`,
      timestamp: new Date()
    })
  }
  
  // Keep only last 10 results
  if (testResults.value.length > 10) {
    testResults.value = testResults.value.slice(0, 10)
  }
}

onMounted(() => {
  systemStore.initializeStore()
})
</script>
