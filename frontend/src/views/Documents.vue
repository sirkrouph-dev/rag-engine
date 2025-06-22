<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h1 class="text-3xl font-bold text-gray-900">Documents</h1>
      <p class="mt-2 text-gray-600">
        View and manage documents in your RAG pipeline
      </p>
    </div>

    <!-- Stats -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
      <StatCard
        title="Total Documents"
        :value="documentsData?.total || 0"
        status="info"
        icon="DocumentTextIcon"
      />
      <StatCard
        title="Total Chunks"
        :value="chunksData?.total || 0"
        status="info"
        icon="Squares2X2Icon"
      />
      <StatCard
        title="Average Chunk Size"
        :value="averageChunkSize"
        status="info"
        icon="ChartBarIcon"
      />
    </div>

    <!-- Documents List -->
    <div class="card">
      <div class="px-6 py-4 border-b border-gray-200">
        <div class="flex items-center justify-between">
          <h2 class="text-lg font-semibold text-gray-900">Documents</h2>
          <button 
            @click="refreshData"
            :disabled="isLoading"
            class="btn btn-sm btn-secondary"
          >
            <ArrowPathIcon class="w-4 h-4 mr-2" />
            Refresh
          </button>
        </div>
      </div>
      
      <div v-if="isLoading" class="p-6 text-center">
        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
        <p class="mt-2 text-gray-600">Loading documents...</p>
      </div>
      
      <div v-else-if="documentsData?.documents?.length > 0" class="divide-y divide-gray-200">
        <div 
          v-for="(doc, index) in documentsData.documents" 
          :key="index"
          class="p-6 hover:bg-gray-50 transition-colors"
        >
          <div class="flex items-start justify-between">
            <div class="flex-1">
              <div class="flex items-center space-x-2">
                <DocumentTextIcon class="w-5 h-5 text-gray-400" />
                <h3 class="text-sm font-medium text-gray-900">
                  {{ getFileName(doc.path) }}
                </h3>
                <span class="badge badge-info">{{ doc.type }}</span>
              </div>
              <p class="mt-1 text-sm text-gray-600">{{ doc.path }}</p>
              <div class="mt-2 flex items-center space-x-4 text-xs text-gray-500">
                <span>Size: {{ formatSize(doc.size) }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div v-else class="p-6 text-center text-gray-500">
        <DocumentTextIcon class="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <h3 class="text-lg font-medium text-gray-900 mb-2">No documents found</h3>
        <p>No documents have been loaded into the pipeline yet.</p>
      </div>
    </div>

    <!-- Chunks List -->
    <div class="card">
      <div class="px-6 py-4 border-b border-gray-200">
        <div class="flex items-center justify-between">
          <h2 class="text-lg font-semibold text-gray-900">Document Chunks</h2>
          <div class="flex items-center space-x-2">
            <span class="text-sm text-gray-600">
              Showing {{ Math.min(chunksLimit, chunksData?.total || 0) }} of {{ chunksData?.total || 0 }}
            </span>
            <button 
              v-if="chunksData?.total > chunksLimit"
              @click="chunksLimit += 20"
              class="btn btn-sm btn-secondary"
            >
              Load More
            </button>
          </div>
        </div>
      </div>
      
      <div v-if="chunksData?.chunks?.length > 0" class="divide-y divide-gray-200">
        <div 
          v-for="(chunk, index) in displayedChunks" 
          :key="chunk.id"
          class="p-6 hover:bg-gray-50 transition-colors"
        >
          <div class="flex items-start justify-between mb-3">
            <div class="flex items-center space-x-2">
              <Squares2X2Icon class="w-4 h-4 text-gray-400" />
              <span class="text-sm font-medium text-gray-900">Chunk {{ chunk.id + 1 }}</span>
            </div>
            <div class="text-xs text-gray-500">
              {{ chunk.content_preview?.length || 0 }} characters
            </div>
          </div>
          
          <div class="text-sm text-gray-900 bg-gray-50 p-3 rounded-lg font-mono">
            {{ chunk.content_preview }}
          </div>
          
          <div v-if="chunk.metadata && Object.keys(chunk.metadata).length > 0" class="mt-3">
            <h4 class="text-xs font-medium text-gray-700 mb-2">Metadata:</h4>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
              <div 
                v-for="(value, key) in chunk.metadata" 
                :key="key"
                class="text-xs"
              >
                <span class="text-gray-600">{{ key }}:</span>
                <span class="ml-1 text-gray-900">{{ value }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div v-else class="p-6 text-center text-gray-500">
        <Squares2X2Icon class="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <h3 class="text-lg font-medium text-gray-900 mb-2">No chunks found</h3>
        <p>Document chunks will appear here after building the pipeline.</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { 
  DocumentTextIcon, 
  Squares2X2Icon,
  ArrowPathIcon,
  ChartBarIcon
} from '@heroicons/vue/24/outline'
import StatCard from '../components/StatCard.vue'
import api from '../services/api'

const documentsData = ref(null)
const chunksData = ref(null)
const isLoading = ref(false)
const chunksLimit = ref(20)

const averageChunkSize = computed(() => {
  if (!chunksData.value?.chunks?.length) return 0
  const totalSize = chunksData.value.chunks.reduce((sum, chunk) => {
    return sum + (chunk.content_preview?.length || 0)
  }, 0)
  return Math.round(totalSize / chunksData.value.chunks.length)
})

const displayedChunks = computed(() => {
  if (!chunksData.value?.chunks) return []
  return chunksData.value.chunks.slice(0, chunksLimit.value)
})

async function refreshData() {
  isLoading.value = true
  try {
    const [docs, chunks] = await Promise.all([
      api.getDocuments(),
      api.getChunks()
    ])
    documentsData.value = docs
    chunksData.value = chunks
  } catch (error) {
    console.error('Failed to load documents data:', error)
  } finally {
    isLoading.value = false
  }
}

function getFileName(path) {
  if (!path) return 'Unknown'
  return path.split('/').pop() || path.split('\\').pop() || path
}

function formatSize(size) {
  if (!size) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let unitIndex = 0
  let fileSize = size
  
  while (fileSize >= 1024 && unitIndex < units.length - 1) {
    fileSize /= 1024
    unitIndex++
  }
  
  return `${fileSize.toFixed(1)} ${units[unitIndex]}`
}

onMounted(() => {
  refreshData()
})
</script>
