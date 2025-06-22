<template>
  <div class="flex items-center space-x-3">
    <!-- Health Status -->
    <div class="flex items-center space-x-2">
      <div 
        :class="[
          'w-2 h-2 rounded-full',
          systemStore.systemHealth === 'healthy' ? 'bg-success-500' :
          systemStore.systemHealth === 'unhealthy' ? 'bg-danger-500' :
          'bg-yellow-500'
        ]"
      />
      <span class="text-sm text-gray-600 capitalize">
        {{ systemStore.systemHealth }}
      </span>
    </div>

    <!-- Pipeline Status -->
    <div v-if="systemStore.status" class="flex items-center space-x-2">
      <span 
        :class="[
          'badge',
          systemStore.isPipelineBuilt ? 'badge-success' : 'badge-warning'
        ]"
      >
        {{ systemStore.isPipelineBuilt ? 'Pipeline Ready' : 'Pipeline Not Built' }}
      </span>
    </div>

    <!-- Loading Indicator -->
    <div v-if="systemStore.isLoading" class="flex items-center">
      <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-600"></div>
    </div>
  </div>
</template>

<script setup>
import { useSystemStore } from '../stores/system'

const systemStore = useSystemStore()
</script>
