<template>
  <div class="routing-tester bg-dark-surface dark:bg-dark-surface light:bg-light-surface rounded-lg p-6 shadow-lg">
    <h2 class="text-xl font-semibold text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-4">
      Test Routing System
    </h2>
    <p class="text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary mb-6">
      Test how queries are routed and classified by the conversational routing system.
    </p>

    <form @submit.prevent="runTest" class="space-y-6">
      <!-- Test Query Input -->
      <div>
        <label class="block text-sm font-medium text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-2">
          Test Query
        </label>
        <textarea
          v-model="testQuery"
          rows="3"
          class="w-full px-3 py-2 bg-dark-bg dark:bg-dark-bg light:bg-light-bg border border-dark-border dark:border-dark-border light:border-light-border rounded-lg text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary placeholder-dark-text-secondary dark:placeholder-dark-text-secondary light:placeholder-light-text-secondary focus:ring-2 focus:ring-accent-primary-dark dark:focus:ring-accent-primary-dark light:focus:ring-accent-primary-light focus:border-transparent"
          placeholder="Enter a query to test routing (e.g., 'Hello!', 'What is machine learning?', 'What's the weather?')"
        ></textarea>
      </div>

      <!-- Quick Test Examples -->
      <div>
        <label class="block text-sm font-medium text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-2">
          Quick Test Examples
        </label>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
          <button
            v-for="example in testExamples"
            :key="example.query"
            @click="testQuery = example.query"
            type="button"
            class="text-left p-3 bg-dark-bg dark:bg-dark-bg light:bg-light-bg border border-dark-border dark:border-dark-border light:border-light-border rounded-lg hover:border-accent-primary-dark dark:hover:border-accent-primary-dark light:hover:border-accent-primary-light transition-colors"
          >
            <div class="text-sm font-medium text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">
              {{ example.query }}
            </div>
            <div class="text-xs text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary mt-1">
              Expected: {{ example.expected }}
            </div>
          </button>
        </div>
      </div>

      <!-- Test Button -->
      <div class="flex justify-center">
        <button
          type="submit"
          :disabled="!testQuery.trim() || testing"
          :class="[
            'px-8 py-3 rounded-lg font-medium transition-colors',
            testQuery.trim() && !testing
              ? 'bg-accent-primary-dark text-white hover:bg-accent-primary-dark/80 dark:bg-accent-primary-dark dark:text-white dark:hover:bg-accent-primary-dark/80 light:bg-accent-primary-light light:text-white light:hover:bg-accent-primary-light/80'
              : 'bg-dark-border text-dark-text-secondary cursor-not-allowed dark:bg-dark-border dark:text-dark-text-secondary light:bg-light-border light:text-light-text-secondary'
          ]"
        >
          <span v-if="testing" class="flex items-center">
            <div class="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-3"></div>
            Testing Routing...
          </span>
          <span v-else>Test Routing</span>
        </button>
      </div>
    </form>

    <!-- Test Results -->
    <div v-if="testResult" class="mt-8 space-y-6">
      <div class="border-t border-dark-border dark:border-dark-border light:border-light-border pt-6">
        <h3 class="text-lg font-semibold text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-4">
          Routing Results
        </h3>

        <!-- Query Info -->
        <div class="bg-dark-bg dark:bg-dark-bg light:bg-light-bg rounded-lg p-4 mb-4">
          <div class="text-sm text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary mb-2">
            Tested Query:
          </div>
          <div class="text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary font-medium">
            "{{ testResult.query }}"
          </div>
        </div>

        <!-- Routing Insights -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <!-- Category & Strategy -->
          <div class="bg-dark-bg dark:bg-dark-bg light:bg-light-bg rounded-lg p-4">
            <h4 class="text-sm font-medium text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-3">
              Classification
            </h4>
            <div class="space-y-2">
              <div class="flex justify-between">
                <span class="text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">Category:</span>
                <span class="text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary font-medium">
                  {{ formatCategory(testResult.estimated_category) }}
                </span>
              </div>
              <div class="flex justify-between">
                <span class="text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">Strategy:</span>
                <span class="text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary font-medium">
                  {{ formatStrategy(testResult.estimated_strategy) }}
                </span>
              </div>
              <div class="flex justify-between">
                <span class="text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">Confidence:</span>
                <span :class="[
                  'font-medium',
                  testResult.confidence >= 0.8 
                    ? 'text-green-500' 
                    : testResult.confidence >= 0.6 
                      ? 'text-yellow-500' 
                      : 'text-red-500'
                ]">
                  {{ (testResult.confidence * 100).toFixed(1) }}%
                </span>
              </div>
            </div>
          </div>

          <!-- Status -->
          <div class="bg-dark-bg dark:bg-dark-bg light:bg-light-bg rounded-lg p-4">
            <h4 class="text-sm font-medium text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-3">
              Status
            </h4>
            <div class="space-y-2">
              <div class="flex items-center">
                <div :class="[
                  'w-3 h-3 rounded-full mr-2',
                  testResult.routing_enabled ? 'bg-green-500' : 'bg-red-500'
                ]"></div>
                <span class="text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">
                  {{ testResult.routing_enabled ? 'Routing Enabled' : 'Routing Disabled' }}
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- Reasoning -->
        <div v-if="testResult.reasoning" class="mt-4 bg-dark-bg dark:bg-dark-bg light:bg-light-bg rounded-lg p-4">
          <h4 class="text-sm font-medium text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-2">
            Routing Reasoning
          </h4>
          <p class="text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary text-sm">
            {{ testResult.reasoning }}
          </p>
        </div>

        <!-- Strategy Explanation -->
        <div class="mt-4 bg-blue-50 dark:bg-blue-900/20 light:bg-blue-50 border border-blue-200 dark:border-blue-800 light:border-blue-200 rounded-lg p-4">
          <h4 class="text-sm font-medium text-blue-800 dark:text-blue-200 light:text-blue-800 mb-2">
            What This Means
          </h4>
          <p class="text-blue-700 dark:text-blue-300 light:text-blue-700 text-sm">
            {{ getStrategyExplanation(testResult.estimated_strategy) }}
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'RoutingTester',
  props: {
    testResult: {
      type: Object,
      default: null
    },
    testing: {
      type: Boolean,
      default: false
    }
  },
  data() {
    return {
      testQuery: '',
      testExamples: [
        { query: 'Hello!', expected: 'Simple Response' },
        { query: 'What is machine learning?', expected: 'RAG Retrieval' },
        { query: 'Thank you for your help', expected: 'Simple Response' },
        { query: 'How do I implement neural networks?', expected: 'RAG Retrieval' },
        { query: 'What\'s the weather like today?', expected: 'Polite Rejection' },
        { query: 'Tell me about AI', expected: 'Clarification Request' },
        { query: 'Can you help me debug this error?', expected: 'RAG Retrieval' },
        { query: 'Goodbye!', expected: 'Simple Response' }
      ]
    }
  },
  methods: {
    runTest() {
      if (!this.testQuery.trim()) return
      
      this.$emit('test', {
        query: this.testQuery.trim(),
        config: {} // Could include test-specific config
      })
    },

    formatCategory(category) {
      return category
        ? category.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
          ).join(' ')
        : 'Unknown'
    },

    formatStrategy(strategy) {
      return strategy
        ? strategy.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
          ).join(' ')
        : 'Unknown'
    },

    getStrategyExplanation(strategy) {
      const explanations = {
        'rag_retrieval': 'The query requires factual information that should be retrieved from documents using the RAG system.',
        'contextual_chat': 'The query can be answered using conversation context without needing document retrieval.',
        'simple_response': 'The query is a greeting, gratitude, or simple interaction that needs a direct response.',
        'polite_rejection': 'The query is outside the system\'s domain and should be politely declined with helpful guidance.',
        'clarification_request': 'The query is too broad or ambiguous and needs clarification to provide a helpful response.'
      }
      return explanations[strategy] || 'The routing strategy determines how the system will respond to this type of query.'
    }
  }
}
</script>
