<template>
  <div class="routing-analytics bg-dark-surface dark:bg-dark-surface light:bg-light-surface rounded-lg p-6 shadow-lg">
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-xl font-semibold text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">
        Routing Analytics
      </h2>
      <button
        @click="refreshAnalytics"
        :disabled="loading"
        class="px-4 py-2 bg-accent-primary-dark text-white rounded-lg hover:bg-accent-primary-dark/80 disabled:opacity-50 disabled:cursor-not-allowed dark:bg-accent-primary-dark dark:hover:bg-accent-primary-dark/80 light:bg-accent-primary-light light:hover:bg-accent-primary-light/80 transition-colors"
      >
        <span v-if="loading" class="flex items-center">
          <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
          Loading...
        </span>
        <span v-else>Refresh</span>
      </button>
    </div>

    <div v-if="loading && !hasData" class="flex items-center justify-center py-8">
      <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-accent-primary-dark dark:border-accent-primary-dark light:border-accent-primary-light"></div>
    </div>

    <div v-else class="space-y-6">
      <!-- Overview Stats -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="bg-dark-bg dark:bg-dark-bg light:bg-light-bg rounded-lg p-4">
          <div class="text-2xl font-bold text-accent-primary-dark dark:text-accent-primary-dark light:text-accent-primary-light">
            {{ analytics.total_queries || 0 }}
          </div>
          <div class="text-sm text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">
            Total Queries Processed
          </div>
        </div>

        <div class="bg-dark-bg dark:bg-dark-bg light:bg-light-bg rounded-lg p-4">
          <div class="text-2xl font-bold text-green-500">
            {{ (analytics.avg_confidence * 100 || 0).toFixed(1) }}%
          </div>
          <div class="text-sm text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">
            Average Confidence
          </div>
        </div>

        <div class="bg-dark-bg dark:bg-dark-bg light:bg-light-bg rounded-lg p-4">
          <div class="text-2xl font-bold text-blue-500">
            {{ getMostCommonStrategy() }}
          </div>
          <div class="text-sm text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">
            Most Common Strategy
          </div>
        </div>
      </div>

      <!-- Routing Decisions Chart -->
      <div class="bg-dark-bg dark:bg-dark-bg light:bg-light-bg rounded-lg p-4">
        <h3 class="text-lg font-semibold text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-4">
          Response Strategy Distribution
        </h3>
        
        <div v-if="hasRoutingData" class="space-y-3">
          <div 
            v-for="(count, strategy) in analytics.routing_decisions" 
            :key="strategy"
            class="flex items-center justify-between"
          >
            <span class="text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary capitalize">
              {{ formatStrategyName(strategy) }}
            </span>
            <div class="flex items-center space-x-3 flex-1 ml-4">
              <div class="flex-1 bg-dark-border dark:bg-dark-border light:bg-light-border rounded-full h-2">
                <div 
                  :class="getStrategyColor(strategy)"
                  class="h-2 rounded-full transition-all duration-500"
                  :style="{ width: getStrategyPercentage(strategy) + '%' }"
                ></div>
              </div>
              <span class="text-sm text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary w-12 text-right">
                {{ count }}
              </span>
            </div>
          </div>
        </div>
        
        <div v-else class="text-center py-8 text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">
          No routing data available yet. Process some queries to see analytics.
        </div>
      </div>

      <!-- Category Distribution -->
      <div class="bg-dark-bg dark:bg-dark-bg light:bg-light-bg rounded-lg p-4">
        <h3 class="text-lg font-semibold text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-4">
          Query Category Distribution
        </h3>
        
        <div v-if="hasCategoryData" class="space-y-3">
          <div 
            v-for="(count, category) in analytics.category_distribution" 
            :key="category"
            class="flex items-center justify-between"
          >
            <span class="text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary capitalize">
              {{ formatCategoryName(category) }}
            </span>
            <div class="flex items-center space-x-3 flex-1 ml-4">
              <div class="flex-1 bg-dark-border dark:bg-dark-border light:bg-light-border rounded-full h-2">
                <div 
                  :class="getCategoryColor(category)"
                  class="h-2 rounded-full transition-all duration-500"
                  :style="{ width: getCategoryPercentage(category) + '%' }"
                ></div>
              </div>
              <span class="text-sm text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary w-12 text-right">
                {{ count }}
              </span>
            </div>
          </div>
        </div>
        
        <div v-else class="text-center py-8 text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">
          No category data available yet. Process some queries to see analytics.
        </div>
      </div>

      <!-- Template Usage -->
      <div v-if="hasTemplateUsage" class="bg-dark-bg dark:bg-dark-bg light:bg-light-bg rounded-lg p-4">
        <h3 class="text-lg font-semibold text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-4">
          Template Usage Statistics
        </h3>
        
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div 
            v-for="(usage, template) in analytics.template_usage" 
            :key="template"
            class="border border-dark-border dark:border-dark-border light:border-light-border rounded-lg p-3"
          >
            <div class="text-lg font-semibold text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">
              {{ usage.count || 0 }}
            </div>
            <div class="text-sm text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">
              {{ formatTemplateName(template) }}
            </div>
          </div>
        </div>
      </div>

      <!-- Info Panel -->
      <div class="bg-blue-50 dark:bg-blue-900/20 light:bg-blue-50 border border-blue-200 dark:border-blue-800 light:border-blue-200 rounded-lg p-4">
        <h4 class="text-sm font-medium text-blue-800 dark:text-blue-200 light:text-blue-800 mb-2">
          About Routing Analytics
        </h4>
        <p class="text-blue-700 dark:text-blue-300 light:text-blue-700 text-sm">
          These analytics show how the conversational routing system is performing. Monitor strategy distribution to ensure queries are being routed appropriately, and track confidence levels to identify areas for improvement.
        </p>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'RoutingAnalytics',
  props: {
    analytics: {
      type: Object,
      default: () => ({})
    },
    loading: {
      type: Boolean,
      default: false
    }
  },
  computed: {
    hasData() {
      return Object.keys(this.analytics).length > 0
    },

    hasRoutingData() {
      return this.analytics.routing_decisions && 
             Object.values(this.analytics.routing_decisions).some(count => count > 0)
    },

    hasCategoryData() {
      return this.analytics.category_distribution && 
             Object.values(this.analytics.category_distribution).some(count => count > 0)
    },

    hasTemplateUsage() {
      return this.analytics.template_usage && 
             Object.keys(this.analytics.template_usage).length > 0
    },

    totalRoutingDecisions() {
      if (!this.analytics.routing_decisions) return 0
      return Object.values(this.analytics.routing_decisions).reduce((sum, count) => sum + count, 0)
    },

    totalCategoryDistribution() {
      if (!this.analytics.category_distribution) return 0
      return Object.values(this.analytics.category_distribution).reduce((sum, count) => sum + count, 0)
    }
  },
  methods: {
    refreshAnalytics() {
      this.$emit('refresh')
    },

    getMostCommonStrategy() {
      if (!this.analytics.routing_decisions) return 'N/A'
      
      const decisions = this.analytics.routing_decisions
      const maxStrategy = Object.keys(decisions).reduce((a, b) => 
        decisions[a] > decisions[b] ? a : b, Object.keys(decisions)[0]
      )
      
      return this.formatStrategyName(maxStrategy) || 'N/A'
    },

    formatStrategyName(strategy) {
      return strategy
        ? strategy.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
          ).join(' ')
        : strategy
    },

    formatCategoryName(category) {
      return category
        ? category.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
          ).join(' ')
        : category
    },

    formatTemplateName(template) {
      return template
        ? template.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
          ).join(' ')
        : template
    },

    getStrategyPercentage(strategy) {
      if (!this.analytics.routing_decisions || this.totalRoutingDecisions === 0) return 0
      return (this.analytics.routing_decisions[strategy] / this.totalRoutingDecisions) * 100
    },

    getCategoryPercentage(category) {
      if (!this.analytics.category_distribution || this.totalCategoryDistribution === 0) return 0
      return (this.analytics.category_distribution[category] / this.totalCategoryDistribution) * 100
    },

    getStrategyColor(strategy) {
      const colors = {
        'rag_retrieval': 'bg-blue-500',
        'contextual_chat': 'bg-green-500',
        'simple_response': 'bg-yellow-500',
        'polite_rejection': 'bg-red-500',
        'clarification_request': 'bg-purple-500'
      }
      return colors[strategy] || 'bg-gray-500'
    },

    getCategoryColor(category) {
      const colors = {
        'rag_factual': 'bg-blue-500',
        'rag_analytical': 'bg-indigo-500',
        'greeting': 'bg-green-500',
        'goodbye': 'bg-green-400',
        'gratitude': 'bg-yellow-500',
        'out_of_context': 'bg-red-500',
        'follow_up': 'bg-purple-500',
        'small_talk': 'bg-pink-500'
      }
      return colors[category] || 'bg-gray-500'
    }
  }
}
</script>
