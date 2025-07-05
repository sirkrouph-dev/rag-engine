<template>
  <div class="routing-management max-w-7xl mx-auto py-8 px-4">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-3xl font-bold text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-2">
        Conversational Routing
      </h1>
      <p class="text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">
        Configure and customize the advanced conversational routing system for human-like chat responses.
      </p>
    </div>

    <!-- Tab Navigation -->
    <div class="border-b border-dark-border dark:border-dark-border light:border-light-border mb-6">
      <nav class="-mb-px flex space-x-8">
        <button
          v-for="tab in tabs"
          :key="tab.id"
          @click="activeTab = tab.id"
          :class="[
            'py-2 px-1 border-b-2 font-medium text-sm whitespace-nowrap',
            activeTab === tab.id
              ? 'border-accent-primary-dark text-accent-primary-dark dark:border-accent-primary-dark dark:text-accent-primary-dark light:border-accent-primary-light light:text-accent-primary-light'
              : 'border-transparent text-dark-text-secondary hover:text-dark-text-primary hover:border-dark-border dark:text-dark-text-secondary dark:hover:text-dark-text-primary dark:hover:border-dark-border light:text-light-text-secondary light:hover:text-light-text-primary light:hover:border-light-border'
          ]"
        >
          {{ tab.name }}
        </button>
      </nav>
    </div>

    <!-- Tab Content -->
    <div class="tab-content">
      <!-- Configuration Tab -->
      <div v-show="activeTab === 'config'" class="space-y-6">
        <RoutingConfig 
          :config="routingConfig" 
          @update="updateRoutingConfig"
          :loading="configLoading"
        />
      </div>

      <!-- Templates Tab -->
      <div v-show="activeTab === 'templates'" class="space-y-6">
        <TemplateManager 
          :templates="templates" 
          @update="updateTemplate"
          @refresh="loadTemplates"
          :loading="templatesLoading"
        />
      </div>

      <!-- Testing Tab -->
      <div v-show="activeTab === 'testing'" class="space-y-6">
        <RoutingTester 
          @test="testRouting"
          :testResult="testResult"
          :testing="testing"
        />
      </div>

      <!-- Analytics Tab -->
      <div v-show="activeTab === 'analytics'" class="space-y-6">
        <RoutingAnalytics 
          :analytics="analytics"
          @refresh="loadAnalytics"
          :loading="analyticsLoading"
        />
      </div>
    </div>

    <!-- Loading Overlay -->
    <div v-if="globalLoading" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div class="bg-dark-surface dark:bg-dark-surface light:bg-light-surface rounded-lg p-6 shadow-xl">
        <div class="flex items-center space-x-3">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-accent-primary-dark dark:border-accent-primary-dark light:border-accent-primary-light"></div>
          <span class="text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">Loading...</span>
        </div>
      </div>
    </div>

    <!-- Success/Error Messages -->
    <div v-if="message" :class="[
      'fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 max-w-md',
      message.type === 'success' 
        ? 'bg-green-500 text-white' 
        : 'bg-red-500 text-white'
    ]">
      <div class="flex items-center">
        <span>{{ message.text }}</span>
        <button @click="message = null" class="ml-4 text-white hover:text-gray-200">
          ×
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import api from '../services/api'
import RoutingConfig from '../components/routing/RoutingConfig.vue'
import TemplateManager from '../components/routing/TemplateManager.vue'
import RoutingTester from '../components/routing/RoutingTester.vue'
import RoutingAnalytics from '../components/routing/RoutingAnalytics.vue'

export default {
  name: 'RoutingManagement',
  components: {
    RoutingConfig,
    TemplateManager,
    RoutingTester,
    RoutingAnalytics
  },
  data() {
    return {
      activeTab: 'config',
      tabs: [
        { id: 'config', name: 'Configuration' },
        { id: 'templates', name: 'Templates' },
        { id: 'testing', name: 'Testing' },
        { id: 'analytics', name: 'Analytics' }
      ],
      routingConfig: {},
      templates: {},
      analytics: {},
      testResult: null,
      message: null,
      globalLoading: false,
      configLoading: false,
      templatesLoading: false,
      analyticsLoading: false,
      testing: false
    }
  },
  async mounted() {
    await this.loadAllData()
  },
  methods: {
    async loadAllData() {
      this.globalLoading = true
      try {
        await Promise.all([
          this.loadRoutingConfig(),
          this.loadTemplates(),
          this.loadAnalytics()
        ])
      } catch (error) {
        this.showMessage('Error loading data: ' + error.message, 'error')
      } finally {
        this.globalLoading = false
      }
    },

    async loadRoutingConfig() {
      this.configLoading = true
      try {
        const response = await api.getRoutingConfig()
        this.routingConfig = response.routing_config || {}
      } catch (error) {
        console.error('Error loading routing config:', error)
        this.showMessage('Error loading routing configuration', 'error')
      } finally {
        this.configLoading = false
      }
    },

    async updateRoutingConfig(config) {
      try {
        const response = await api.updateRoutingConfig(config)
        this.routingConfig = { ...this.routingConfig, ...config }
        this.showMessage('Routing configuration updated successfully', 'success')
      } catch (error) {
        console.error('Error updating routing config:', error)
        this.showMessage('Error updating routing configuration', 'error')
      }
    },

    async loadTemplates() {
      this.templatesLoading = true
      try {
        const response = await api.getRoutingTemplates()
        this.templates = response.templates || {}
      } catch (error) {
        console.error('Error loading templates:', error)
        this.showMessage('Error loading templates', 'error')
      } finally {
        this.templatesLoading = false
      }
    },

    async updateTemplate(templateName, templateData) {
      try {
        const response = await api.updateRoutingTemplate(templateName, templateData)
        await this.loadTemplates() // Refresh templates
        this.showMessage(`Template '${templateName}' updated successfully`, 'success')
      } catch (error) {
        console.error('Error updating template:', error)
        this.showMessage('Error updating template', 'error')
      }
    },

    async testRouting(testData) {
      this.testing = true
      try {
        const response = await api.testRouting(testData)
        this.testResult = response.insights
        this.showMessage('Routing test completed', 'success')
      } catch (error) {
        console.error('Error testing routing:', error)
        this.showMessage('Error testing routing', 'error')
      } finally {
        this.testing = false
      }
    },

    async loadAnalytics() {
      this.analyticsLoading = true
      try {
        const response = await api.getRoutingAnalytics()
        this.analytics = response.analytics || {}
      } catch (error) {
        console.error('Error loading analytics:', error)
        this.showMessage('Error loading analytics', 'error')
      } finally {
        this.analyticsLoading = false
      }
    },

    showMessage(text, type = 'info') {
      this.message = { text, type }
      setTimeout(() => {
        this.message = null
      }, 5000)
    }
  }
}
</script>

<style scoped>
.routing-management {
  min-height: calc(100vh - 120px);
}

.tab-content {
  animation: fadeIn 0.3s ease-in-out;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
