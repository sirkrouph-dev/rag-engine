<template>
  <div id="app" class="min-h-screen transition-colors duration-300 bg-dark-bg text-dark-text-primary dark:bg-dark-bg dark:text-dark-text-primary light:bg-light-bg light:text-light-text-primary">
    <!-- Navigation -->
    <nav class="shadow-xl border-b backdrop-blur-md bg-dark-surface border-dark-border dark:bg-dark-surface dark:border-dark-border light:bg-light-surface light:border-light-border">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex">
            <!-- Logo -->
            <div class="flex-shrink-0 flex items-center">
              <router-link to="/" class="flex items-center space-x-3 group">
                <div class="w-8 h-8 bg-gradient-to-br rounded-lg flex items-center justify-center shadow-lg group-hover:shadow-lg transition-all duration-300 from-accent-primary-dark to-accent-secondary-dark dark:from-accent-primary-dark dark:to-accent-secondary-dark light:from-accent-primary-light light:to-accent-secondary-light">
                  <span class="text-white font-bold text-sm">RAG</span>
                </div>
                <span class="text-xl font-semibold text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">RAG Engine</span>
                <span class="px-2 py-1 text-xs font-medium rounded-full animate-pulse bg-accent-error-dark/20 text-accent-error-dark border border-accent-error-dark/30 dark:bg-accent-error-dark/20 dark:text-accent-error-dark dark:border-accent-error-dark/30 light:bg-accent-error-light/20 light:text-accent-error-light light:border-accent-error-light/30">
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
                    ? 'nav-link-active'
                    : 'nav-link-inactive'
                ]"
              >
                <component :is="item.icon" class="w-4 h-4 mr-2" />
                {{ item.label }}
              </router-link>
            </div>
          </div>
          
          <!-- System Status and Theme Toggle -->
          <div class="flex items-center space-x-4">
            <SystemStatus />
            <ThemeToggle />
          </div>
        </div>
      </div>
    </nav>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
      <div class="animate-fade-in">
        <router-view />
      </div>
    </main>

    <!-- Error Toast -->
    <ErrorToast />
    
    <!-- Background Decoration -->
    <div class="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
      <div class="absolute top-0 right-0 w-96 h-96 rounded-full blur-3xl bg-accent-primary-dark/5 dark:bg-accent-primary-dark/5 light:bg-accent-primary-light/5"></div>
      <div class="absolute bottom-0 left-0 w-96 h-96 rounded-full blur-3xl bg-accent-secondary-dark/5 dark:bg-accent-secondary-dark/5 light:bg-accent-secondary-light/5"></div>
    </div>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import { 
  HomeIcon, 
  ChatBubbleLeftRightIcon, 
  CogIcon, 
  DocumentTextIcon, 
  ServerIcon,
  SparklesIcon 
} from '@heroicons/vue/24/outline'
import SystemStatus from './components/SystemStatus.vue'
import ErrorToast from './components/ErrorToast.vue'
import ThemeToggle from './components/ThemeToggle.vue'
import { useSystemStore } from './stores/system'

const systemStore = useSystemStore()

const navigation = [
  { name: 'dashboard', label: 'Dashboard', href: '/', icon: HomeIcon },
  { name: 'chat', label: 'Chat', href: '/chat', icon: ChatBubbleLeftRightIcon },
  { name: 'pipeline', label: 'Pipeline', href: '/pipeline', icon: CogIcon },
  { name: 'documents', label: 'Documents', href: '/documents', icon: DocumentTextIcon },
  { name: 'ai-assistant', label: 'AI Assistant', href: '/ai-assistant', icon: SparklesIcon },
  { name: 'system', label: 'System', href: '/system', icon: ServerIcon }
]

onMounted(() => {
  systemStore.initializeStore()
})
</script>
