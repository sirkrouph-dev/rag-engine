<template>
  <div class="routing-config bg-dark-surface dark:bg-dark-surface light:bg-light-surface rounded-lg p-6 shadow-lg">
    <h2 class="text-xl font-semibold text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-4">
      Routing Configuration
    </h2>

    <div v-if="loading" class="flex items-center justify-center py-8">
      <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-accent-primary-dark dark:border-accent-primary-dark light:border-accent-primary-light"></div>
    </div>

    <form v-else @submit.prevent="saveConfig" class="space-y-6">
      <!-- Enable Routing -->
      <div class="flex items-center justify-between">
        <div>
          <label class="text-sm font-medium text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">
            Enable Conversational Routing
          </label>
          <p class="text-xs text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">
            Use multi-stage LLM routing for better response strategy selection
          </p>
        </div>
        <label class="switch">
          <input 
            type="checkbox" 
            v-model="localConfig.enabled"
            @change="markDirty"
          >
          <span class="slider round"></span>
        </label>
      </div>

      <!-- Fallback to Simple -->
      <div class="flex items-center justify-between">
        <div>
          <label class="text-sm font-medium text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">
            Fallback to Simple Response
          </label>
          <p class="text-xs text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">
            Fall back to simple responses when routing fails
          </p>
        </div>
        <label class="switch">
          <input 
            type="checkbox" 
            v-model="localConfig.fallback_to_simple"
            @change="markDirty"
          >
          <span class="slider round"></span>
        </label>
      </div>

      <!-- System Prompt -->
      <div>
        <label class="block text-sm font-medium text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-2">
          System Prompt
        </label>
        <textarea
          v-model="localConfig.system_prompt"
          @input="markDirty"
          rows="3"
          class="w-full px-3 py-2 bg-dark-bg dark:bg-dark-bg light:bg-light-bg border border-dark-border dark:border-dark-border light:border-light-border rounded-lg text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary placeholder-dark-text-secondary dark:placeholder-dark-text-secondary light:placeholder-light-text-secondary focus:ring-2 focus:ring-accent-primary-dark dark:focus:ring-accent-primary-dark light:focus:ring-accent-primary-light focus:border-transparent"
          placeholder="You are an intelligent AI assistant..."
        ></textarea>
      </div>

      <!-- Temperature Settings -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label class="block text-sm font-medium text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-2">
            Topic Analysis Temperature
          </label>
          <input
            type="number"
            step="0.1"
            min="0"
            max="2"
            v-model.number="localConfig.routing_config.topic_analysis_temperature"
            @input="markDirty"
            class="w-full px-3 py-2 bg-dark-bg dark:bg-dark-bg light:bg-light-bg border border-dark-border dark:border-dark-border light:border-light-border rounded-lg text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary focus:ring-2 focus:ring-accent-primary-dark dark:focus:ring-accent-primary-dark light:focus:ring-accent-primary-light focus:border-transparent"
          >
        </div>

        <div>
          <label class="block text-sm font-medium text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-2">
            Classification Temperature
          </label>
          <input
            type="number"
            step="0.1"
            min="0"
            max="2"
            v-model.number="localConfig.routing_config.classification_temperature"
            @input="markDirty"
            class="w-full px-3 py-2 bg-dark-bg dark:bg-dark-bg light:bg-light-bg border border-dark-border dark:border-dark-border light:border-light-border rounded-lg text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary focus:ring-2 focus:ring-accent-primary-dark dark:focus:ring-accent-primary-dark light:focus:ring-accent-primary-light focus:border-transparent"
          >
        </div>

        <div>
          <label class="block text-sm font-medium text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-2">
            Response Temperature
          </label>
          <input
            type="number"
            step="0.1"
            min="0"
            max="2"
            v-model.number="localConfig.routing_config.response_temperature"
            @input="markDirty"
            class="w-full px-3 py-2 bg-dark-bg dark:bg-dark-bg light:bg-light-bg border border-dark-border dark:border-dark-border light:border-light-border rounded-lg text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary focus:ring-2 focus:ring-accent-primary-dark dark:focus:ring-accent-primary-dark light:focus:ring-accent-primary-light focus:border-transparent"
          >
        </div>
      </div>

      <!-- Advanced Settings -->
      <div class="border-t border-dark-border dark:border-dark-border light:border-light-border pt-6">
        <h3 class="text-lg font-medium text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-4">
          Advanced Settings
        </h3>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label class="block text-sm font-medium text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-2">
              Max Conversation History
            </label>
            <input
              type="number"
              min="1"
              max="50"
              v-model.number="localConfig.routing_config.max_conversation_history"
              @input="markDirty"
              class="w-full px-3 py-2 bg-dark-bg dark:bg-dark-bg light:bg-light-bg border border-dark-border dark:border-dark-border light:border-light-border rounded-lg text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary focus:ring-2 focus:ring-accent-primary-dark dark:focus:ring-accent-primary-dark light:focus:ring-accent-primary-light focus:border-transparent"
            >
          </div>

          <div>
            <label class="block text-sm font-medium text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary mb-2">
              Confidence Threshold
            </label>
            <input
              type="number"
              step="0.1"
              min="0"
              max="1"
              v-model.number="localConfig.routing_config.confidence_threshold"
              @input="markDirty"
              class="w-full px-3 py-2 bg-dark-bg dark:bg-dark-bg light:bg-light-bg border border-dark-border dark:border-dark-border light:border-light-border rounded-lg text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary focus:ring-2 focus:ring-accent-primary-dark dark:focus:ring-accent-primary-dark light:focus:ring-accent-primary-light focus:border-transparent"
            >
          </div>
        </div>
      </div>

      <!-- Save Button -->
      <div class="flex justify-end pt-4">
        <button
          type="submit"
          :disabled="!isDirty || saving"
          :class="[
            'px-6 py-2 rounded-lg font-medium transition-colors',
            isDirty && !saving
              ? 'bg-accent-primary-dark text-white hover:bg-accent-primary-dark/80 dark:bg-accent-primary-dark dark:text-white dark:hover:bg-accent-primary-dark/80 light:bg-accent-primary-light light:text-white light:hover:bg-accent-primary-light/80'
              : 'bg-dark-border text-dark-text-secondary cursor-not-allowed dark:bg-dark-border dark:text-dark-text-secondary light:bg-light-border light:text-light-text-secondary'
          ]"
        >
          <span v-if="saving" class="flex items-center">
            <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
            Saving...
          </span>
          <span v-else>Save Configuration</span>
        </button>
      </div>
    </form>
  </div>
</template>

<script>
export default {
  name: 'RoutingConfig',
  props: {
    config: {
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
      localConfig: {},
      isDirty: false,
      saving: false
    }
  },
  watch: {
    config: {
      handler(newConfig) {
        this.localConfig = this.initializeConfig(newConfig)
        this.isDirty = false
      },
      immediate: true,
      deep: true
    }
  },
  methods: {
    initializeConfig(config) {
      return {
        enabled: config.enabled ?? true,
        fallback_to_simple: config.fallback_to_simple ?? true,
        system_prompt: config.system_prompt ?? 'You are an intelligent AI assistant that knows when and how to use different response strategies.',
        routing_config: {
          topic_analysis_temperature: config.routing_config?.topic_analysis_temperature ?? 0.1,
          classification_temperature: config.routing_config?.classification_temperature ?? 0.1,
          response_temperature: config.routing_config?.response_temperature ?? 0.7,
          max_conversation_history: config.routing_config?.max_conversation_history ?? 10,
          confidence_threshold: config.routing_config?.confidence_threshold ?? 0.8,
          enable_reasoning_chain: config.routing_config?.enable_reasoning_chain ?? true,
          enable_clarification_requests: config.routing_config?.enable_clarification_requests ?? true,
          ...config.routing_config
        },
        domain_config: {
          domain_name: config.domain_config?.domain_name ?? 'General Assistant',
          rejection_style: config.domain_config?.rejection_style ?? 'professional',
          default_expertise: config.domain_config?.default_expertise ?? 'intermediate',
          ...config.domain_config
        }
      }
    },

    markDirty() {
      this.isDirty = true
    },

    async saveConfig() {
      this.saving = true
      try {
        await this.$emit('update', this.localConfig)
        this.isDirty = false
      } catch (error) {
        console.error('Error saving config:', error)
      } finally {
        this.saving = false
      }
    }
  }
}
</script>

<style scoped>
/* Toggle Switch Styles */
.switch {
  position: relative;
  display: inline-block;
  width: 60px;
  height: 34px;
}

.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #ccc;
  transition: .4s;
}

.slider:before {
  position: absolute;
  content: "";
  height: 26px;
  width: 26px;
  left: 4px;
  bottom: 4px;
  background-color: white;
  transition: .4s;
}

input:checked + .slider {
  background-color: #2196F3;
}

input:checked + .slider:before {
  transform: translateX(26px);
}

.slider.round {
  border-radius: 34px;
}

.slider.round:before {
  border-radius: 50%;
}
</style>
