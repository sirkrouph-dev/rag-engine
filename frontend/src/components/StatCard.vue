<template>  <div class="card card-body">
    <div class="flex items-center justify-between">
      <div>
        <p class="text-sm font-medium text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">{{ title }}</p>
        <p class="text-2xl font-bold text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">{{ value }}</p>
      </div>
      <div class="flex-shrink-0">
        <div 
          :class="[
            'w-12 h-12 rounded-lg flex items-center justify-center',
            statusClass
          ]"
        >
          <component :is="iconComponent" class="w-6 h-6" />
        </div>
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
  ChartBarIcon,
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
    default: 'info' // 'success', 'warning', 'error', 'info'
  },
  icon: {
    type: String,
    default: 'ChartBarIcon'
  }
})

const statusClass = computed(() => {
  const classes = {
    'success': 'text-white',
    'warning': 'text-white', 
    'error': 'text-white',
    'info': 'text-white',
    'healthy': 'text-white',
    'unhealthy': 'text-white',
    'unknown': 'text-white'
  }
  
  const bgClasses = {
    'success': 'bg-accent-secondary-dark dark:bg-accent-secondary-dark light:bg-accent-secondary-light',
    'warning': 'bg-yellow-500',
    'error': 'bg-accent-error-dark dark:bg-accent-error-dark light:bg-accent-error-light',
    'info': 'bg-accent-primary-dark dark:bg-accent-primary-dark light:bg-accent-primary-light',
    'healthy': 'bg-accent-secondary-dark dark:bg-accent-secondary-dark light:bg-accent-secondary-light',
    'unhealthy': 'bg-accent-error-dark dark:bg-accent-error-dark light:bg-accent-error-light',
    'unknown': 'bg-yellow-500'
  }
  
  return `${classes[props.status] || classes.info} ${bgClasses[props.status] || bgClasses.info}`
})

const iconComponent = computed(() => {
  const iconMap = {
    ServerIcon,
    CogIcon,
    DocumentTextIcon,
    Squares2X2Icon,
    ChartBarIcon,
    CheckCircleIcon,
    ExclamationTriangleIcon,
    XCircleIcon
  }
  return iconMap[props.icon] || ChartBarIcon
})
</script>
