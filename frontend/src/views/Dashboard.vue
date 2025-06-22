<template>  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h1 class="text-3xl font-bold text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">Dashboard</h1>
      <p class="mt-2 text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">
        Welcome to RAG Engine - An experimental modular retrieval-augmented generation framework
      </p>
    </div>

    <!-- Quick Stats -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <StatCard
        title="System Health"
        :value="systemStore.systemHealth"
        :status="systemStore.systemHealth"
        icon="ServerIcon"
      />
      <StatCard
        title="Pipeline Status"
        :value="systemStore.isPipelineBuilt ? 'Built' : 'Not Built'"
        :status="systemStore.isPipelineBuilt ? 'success' : 'warning'"
        icon="CogIcon"
      />
      <StatCard
        title="Documents"
        :value="documentsCount"
        status="info"
        icon="DocumentTextIcon"
      />
      <StatCard
        title="Chunks"
        :value="chunksCount"
        status="info"
        icon="Squares2X2Icon"
      />
    </div>

    <!-- System Overview -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Configuration Card -->
      <div class="card p-6">
        <h2 class="text-lg font-semibold mb-4 text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">Configuration</h2>
        <div v-if="systemStore.config" class="space-y-3">
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
          <ConfigItem 
            label="Retrieval Top-K" 
            :value="systemStore.config.retrieval_top_k" 
          />
        </div>        <div v-else class="text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary text-center py-4">
          No configuration loaded
        </div>
      </div>

      <!-- Orchestrator Status -->
      <div class="card p-6">
        <h2 class="text-lg font-semibold mb-4 text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">Orchestrator</h2>
        <div v-if="systemStore.orchestratorStatus" class="space-y-3">
          <div class="text-sm">
            <span class="font-medium text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">Type:</span>
            <span class="ml-2 text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">{{ systemStore.orchestratorStatus.type || 'Default' }}</span>
          </div>
          <div class="text-sm">
            <span class="font-medium text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">Status:</span>
            <span 
              :class="[
                'ml-2 badge',
                systemStore.orchestratorStatus.status === 'active' ? 'badge-success' : 'badge-warning'
              ]"
            >
              {{ systemStore.orchestratorStatus.status }}
            </span>
          </div>
          <div v-if="systemStore.components" class="mt-4">
            <span class="font-medium text-sm text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">Components:</span>
            <div class="mt-2 grid grid-cols-2 gap-2">
              <div 
                v-for="(component, type) in systemStore.components" 
                :key="type"
                class="text-xs p-2 rounded bg-dark-bg border border-dark-border dark:bg-dark-bg dark:border-dark-border light:bg-light-bg light:border-light-border"
              >
                <div class="font-medium capitalize text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">{{ type }}</div>
                <div class="text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">{{ component.length || 0 }} available</div>
              </div>
            </div>
          </div>
        </div>
        <div v-else class="text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary text-center py-4">
          Orchestrator not initialized
        </div>
      </div>
    </div>

    <!-- Quick Actions -->
    <div class="card p-6">
      <h2 class="text-lg font-semibold mb-4 text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">Quick Actions</h2>
      <div class="flex flex-wrap gap-4">
        <router-link to="/chat" class="btn btn-primary">
          <ChatBubbleLeftRightIcon class="w-4 h-4 mr-2" />
          Start Chatting
        </router-link>
        <button 
          @click="buildPipeline" 
          :disabled="systemStore.isLoading"
          class="btn btn-secondary"
        >
          <CogIcon class="w-4 h-4 mr-2" />
          Build Pipeline
        </button>
        <router-link to="/documents" class="btn btn-secondary">
          <DocumentTextIcon class="w-4 h-4 mr-2" />
          View Documents
        </router-link>
        <router-link to="/system" class="btn btn-secondary">
          <ServerIcon class="w-4 h-4 mr-2" />
          System Details
        </router-link>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { 
  ChatBubbleLeftRightIcon, 
  CogIcon, 
  DocumentTextIcon, 
  ServerIcon,
  Squares2X2Icon
} from '@heroicons/vue/24/outline'
import { useSystemStore } from '../stores/system'
import StatCard from '../components/StatCard.vue'
import ConfigItem from '../components/ConfigItem.vue'
import api from '../services/api'

const systemStore = useSystemStore()

const documentsData = ref(null)
const chunksData = ref(null)

const documentsCount = computed(() => {
  return documentsData.value?.total || 0
})

const chunksCount = computed(() => {
  return chunksData.value?.total || 0
})

async function loadData() {
  try {
    const [docs, chunks] = await Promise.all([
      api.getDocuments(),
      api.getChunks()
    ])
    documentsData.value = docs
    chunksData.value = chunks
  } catch (error) {
    console.error('Failed to load dashboard data:', error)
  }
}

async function buildPipeline() {
  try {
    await systemStore.buildPipeline()
    await loadData() // Refresh data after build
  } catch (error) {
    console.error('Failed to build pipeline:', error)
  }
}

onMounted(() => {
  loadData()
})
</script>
