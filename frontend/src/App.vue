<template>
  <div id="app" class="min-h-screen bg-gray-50">
    <!-- Navigation -->
    <nav class="bg-white shadow-sm border-b border-gray-200">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex">
            <!-- Logo -->
            <div class="flex-shrink-0 flex items-center">
              <router-link to="/" class="flex items-center space-x-3">
                <div class="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                  <span class="text-white font-bold text-sm">RAG</span>
                </div>
                <span class="text-xl font-semibold text-gray-900">RAG Engine</span>
                <span class="px-2 py-1 text-xs font-medium bg-yellow-100 text-yellow-800 rounded-full">
                  Experimental
                </span>
              </router-link>
            </div>
            
            <!-- Navigation Links -->
            <div class="hidden sm:ml-6 sm:flex sm:space-x-8">
              <router-link
                v-for="item in navigation"
                :key="item.name"
                :to="item.href"
                :class="[
                  $route.name === item.name
                    ? 'border-primary-500 text-gray-900'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700',
                  'inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors'
                ]"
              >
                <component :is="item.icon" class="w-4 h-4 mr-2" />
                {{ item.label }}
              </router-link>
            </div>
          </div>
          
          <!-- System Status -->
          <div class="flex items-center space-x-4">
            <SystemStatus />
          </div>
        </div>
      </div>
    </nav>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
      <router-view />
    </main>

    <!-- Error Toast -->
    <ErrorToast />
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import { 
  HomeIcon, 
  ChatBubbleLeftRightIcon, 
  CogIcon, 
  DocumentTextIcon, 
  ServerIcon 
} from '@heroicons/vue/24/outline'
import SystemStatus from './components/SystemStatus.vue'
import ErrorToast from './components/ErrorToast.vue'
import { useSystemStore } from './stores/system'

const systemStore = useSystemStore()

const navigation = [
  { name: 'dashboard', label: 'Dashboard', href: '/', icon: HomeIcon },
  { name: 'chat', label: 'Chat', href: '/chat', icon: ChatBubbleLeftRightIcon },
  { name: 'pipeline', label: 'Pipeline', href: '/pipeline', icon: CogIcon },
  { name: 'documents', label: 'Documents', href: '/documents', icon: DocumentTextIcon },
  { name: 'system', label: 'System', href: '/system', icon: ServerIcon }
]

onMounted(() => {
  systemStore.initializeStore()
})
</script>
