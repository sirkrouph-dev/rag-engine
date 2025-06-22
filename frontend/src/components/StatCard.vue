<template>
  <div class="card p-6">
    <div class="flex items-center">
      <div class="flex-shrink-0">
        <div 
          :class="[
            'w-8 h-8 rounded-lg flex items-center justify-center',
            status === 'success' || status === 'healthy' ? 'bg-success-100' :
            status === 'warning' ? 'bg-yellow-100' :
            status === 'error' || status === 'unhealthy' ? 'bg-danger-100' :
            'bg-blue-100'
          ]"
        >
          <component 
            :is="iconComponent" 
            :class="[
              'w-5 h-5',
              status === 'success' || status === 'healthy' ? 'text-success-600' :
              status === 'warning' ? 'text-yellow-600' :
              status === 'error' || status === 'unhealthy' ? 'text-danger-600' :
              'text-blue-600'
            ]"
          />
        </div>
      </div>
      <div class="ml-4">
        <h3 class="text-sm font-medium text-gray-500">{{ title }}</h3>
        <p 
          :class="[
            'text-2xl font-semibold capitalize',
            status === 'success' || status === 'healthy' ? 'text-success-900' :
            status === 'warning' ? 'text-yellow-900' :
            status === 'error' || status === 'unhealthy' ? 'text-danger-900' :
            'text-gray-900'
          ]"
        >
          {{ value }}
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { 
  ServerIcon, 
  CogIcon, 
  DocumentTextIcon, 
  Squares2X2Icon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon
} from '@heroicons/vue/24/outline'

const props = defineProps({
  title: {
    type: String,
    required: true
  },
  value: {
    type: [String, Number],
    required: true
  },
  status: {
    type: String,
    default: 'info'
  },
  icon: {
    type: String,
    default: 'ServerIcon'
  }
})

const iconComponent = computed(() => {
  const iconMap = {
    ServerIcon,
    CogIcon,
    DocumentTextIcon,
    Squares2X2Icon,
    CheckCircleIcon,
    ExclamationTriangleIcon,
    XCircleIcon
  }
  return iconMap[props.icon] || ServerIcon
})
</script>
