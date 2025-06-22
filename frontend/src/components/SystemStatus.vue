<template>
  <div class="flex items-center space-x-3">
    <!-- Health Status -->
    <div class="flex items-center space-x-2">
      <div 
        :class="[
          'status-dot',
          systemStore.systemHealth === 'healthy' ? 'status-healthy' :
          systemStore.systemHealth === 'unhealthy' ? 'status-unhealthy' :
          'status-unknown'
        ]"
      />
      <span class="text-sm capitalize text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">
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
      <div class="animate-spin rounded-full h-4 w-4 border-2 border-t-transparent border-accent-primary-dark dark:border-accent-primary-dark light:border-accent-primary-light"></div>
    </div>
  </div>
</template>

<script setup>
import { useSystemStore } from '../stores/system'

const systemStore = useSystemStore()
</script>
