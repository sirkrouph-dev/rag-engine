<template>
  <div class="template-manager bg-dark-surface dark:bg-dark-surface light:bg-light-surface rounded-lg p-6 shadow-lg">
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-xl font-semibold text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">
        Routing Templates
      </h2>
      <button
        @click="refreshTemplates"
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

    <div v-if="loading && Object.keys(templates).length === 0" class="flex items-center justify-center py-8">
      <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-accent-primary-dark dark:border-accent-primary-dark light:border-accent-primary-light"></div>
    </div>

    <div v-else-if="Object.keys(templates).length === 0" class="text-center py-8">
      <p class="text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">
        No routing templates found. Templates should be located in the templates/routing/ directory.
      </p>
    </div>

    <div v-else class="space-y-4">
      <!-- Template List -->
      <div class="grid gap-4">
        <div 
          v-for="(template, name) in templates" 
          :key="name"
          class="border border-dark-border dark:border-dark-border light:border-light-border rounded-lg p-4 hover:border-accent-primary-dark dark:hover:border-accent-primary-dark light:hover:border-accent-primary-light transition-colors"
        >
          <div class="flex items-center justify-between mb-2">
            <h3 class="text-lg font-medium text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">
              {{ formatTemplateName(name) }}
            </h3>
            <div class="flex items-center space-x-2">
              <span class="text-xs text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary bg-dark-bg dark:bg-dark-bg light:bg-light-bg px-2 py-1 rounded">
                {{ template.filename }}
              </span>
              <button
                @click="editTemplate(name)"
                class="px-3 py-1 text-sm bg-accent-primary-dark text-white rounded hover:bg-accent-primary-dark/80 dark:bg-accent-primary-dark dark:hover:bg-accent-primary-dark/80 light:bg-accent-primary-light light:hover:bg-accent-primary-light/80 transition-colors"
              >
                Edit
              </button>
            </div>
          </div>
          
          <div class="text-sm text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">
            {{ getTemplateDescription(name) }}
          </div>
          
          <!-- Preview -->
          <div v-if="!template.collapsed" class="mt-3 p-3 bg-dark-bg dark:bg-dark-bg light:bg-light-bg rounded text-xs font-mono">
            <pre class="whitespace-pre-wrap text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">{{ template.content.substring(0, 200) }}{{ template.content.length > 200 ? '...' : '' }}</pre>
          </div>
          
          <button
            @click="template.collapsed = !template.collapsed"
            class="mt-2 text-xs text-accent-primary-dark hover:text-accent-primary-dark/80 dark:text-accent-primary-dark dark:hover:text-accent-primary-dark/80 light:text-accent-primary-light light:hover:text-accent-primary-light/80"
          >
            {{ template.collapsed ? 'Show Preview' : 'Hide Preview' }}
          </button>
        </div>
      </div>
    </div>

    <!-- Template Editor Modal -->
    <div v-if="editingTemplate" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div class="bg-dark-surface dark:bg-dark-surface light:bg-light-surface rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] flex flex-col">
        <!-- Modal Header -->
        <div class="flex items-center justify-between p-6 border-b border-dark-border dark:border-dark-border light:border-light-border">
          <h3 class="text-xl font-semibold text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">
            Edit Template: {{ formatTemplateName(editingTemplate.name) }}
          </h3>
          <button
            @click="cancelEdit"
            class="text-dark-text-secondary hover:text-dark-text-primary dark:text-dark-text-secondary dark:hover:text-dark-text-primary light:text-light-text-secondary light:hover:text-light-text-primary"
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
            </svg>
          </button>
        </div>

        <!-- Modal Body -->
        <div class="flex-1 p-6 overflow-hidden">
          <div class="mb-4">
            <p class="text-sm text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary mb-2">
              {{ getTemplateDescription(editingTemplate.name) }}
            </p>
          </div>
          
          <div class="h-96">
            <label class="block text-sm font-medium text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-2">
              Template Content
            </label>
            <textarea
              v-model="editingTemplate.content"
              class="w-full h-full px-3 py-2 bg-dark-bg dark:bg-dark-bg light:bg-light-bg border border-dark-border dark:border-dark-border light:border-light-border rounded-lg text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary font-mono text-sm focus:ring-2 focus:ring-accent-primary-dark dark:focus:ring-accent-primary-dark light:focus:ring-accent-primary-light focus:border-transparent resize-none"
              placeholder="Enter template content..."
            ></textarea>
          </div>
        </div>

        <!-- Modal Footer -->
        <div class="flex items-center justify-end space-x-3 p-6 border-t border-dark-border dark:border-dark-border light:border-light-border">
          <button
            @click="cancelEdit"
            class="px-4 py-2 text-dark-text-secondary border border-dark-border rounded-lg hover:bg-dark-bg dark:text-dark-text-secondary dark:border-dark-border dark:hover:bg-dark-bg light:text-light-text-secondary light:border-light-border light:hover:bg-light-bg transition-colors"
          >
            Cancel
          </button>
          <button
            @click="saveTemplate"
            :disabled="saving"
            class="px-4 py-2 bg-accent-primary-dark text-white rounded-lg hover:bg-accent-primary-dark/80 disabled:opacity-50 disabled:cursor-not-allowed dark:bg-accent-primary-dark dark:hover:bg-accent-primary-dark/80 light:bg-accent-primary-light light:hover:bg-accent-primary-light/80 transition-colors"
          >
            <span v-if="saving" class="flex items-center">
              <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              Saving...
            </span>
            <span v-else>Save Template</span>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'TemplateManager',
  props: {
    templates: {
      type: Object,
      default: () => ({})
    },
    loading: {
      type: Boolean,
      default: false
    }
  },
  data() {
    return {
      editingTemplate: null,
      originalContent: '',
      saving: false
    }
  },
  watch: {
    templates: {
      handler(templates) {
        // Add collapsed state to templates for UI
        Object.keys(templates).forEach(key => {
          if (!templates[key].hasOwnProperty('collapsed')) {
            this.$set(templates[key], 'collapsed', true)
          }
        })
      },
      immediate: true,
      deep: true
    }
  },
  methods: {
    formatTemplateName(name) {
      return name
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ')
    },

    getTemplateDescription(name) {
      const descriptions = {
        'topic_analysis': 'Analyzes user queries to identify topic, breadth, and ambiguity',
        'query_classification': 'Classifies queries and extracts metadata for routing decisions',
        'rag_response': 'Generates responses using RAG retrieval with routing context',
        'contextual_chat': 'Provides conversational responses using conversation history',
        'polite_rejection': 'Diplomatically handles out-of-scope or inappropriate queries',
        'clarification_request': 'Asks targeted questions to clarify ambiguous queries'
      }
      return descriptions[name] || 'Custom routing template'
    },

    editTemplate(name) {
      const template = this.templates[name]
      this.editingTemplate = {
        name,
        content: template.content,
        filename: template.filename,
        path: template.path
      }
      this.originalContent = template.content
    },

    cancelEdit() {
      this.editingTemplate = null
      this.originalContent = ''
    },

    async saveTemplate() {
      if (!this.editingTemplate) return

      this.saving = true
      try {
        await this.$emit('update', this.editingTemplate.name, {
          content: this.editingTemplate.content
        })
        this.editingTemplate = null
        this.originalContent = ''
      } catch (error) {
        console.error('Error saving template:', error)
      } finally {
        this.saving = false
      }
    },

    refreshTemplates() {
      this.$emit('refresh')
    }
  }
}
</script>

<style scoped>
pre {
  white-space: pre-wrap;
  word-wrap: break-word;
}
</style>
