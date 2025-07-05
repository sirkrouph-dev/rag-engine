<template>
  <div class="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
    <div class="container mx-auto px-6 py-8">
      <!-- Header -->
      <div class="flex items-center justify-between mb-8">
        <div>
          <h1 class="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
            AI Assistant
          </h1>
          <p class="text-slate-300 mt-2">Get help with stack configuration and bloat management</p>
        </div>
        <div class="flex items-center space-x-4">
          <select 
            v-model="selectedModel" 
            class="bg-slate-800 border border-slate-600 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-purple-400 focus:border-transparent"
          >
            <option value="phi3.5:latest">Phi 3.5</option>
            <option value="llama3.1:latest">Llama 3.1</option>
            <option value="mistral:latest">Mistral</option>
          </select>
        </div>
      </div>

      <!-- Stack Configuration Panel -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        <!-- Stack Selector -->
        <div class="bg-slate-800/50 backdrop-blur rounded-xl border border-slate-700 p-6">
          <h3 class="text-xl font-semibold mb-4 text-blue-400">Quick Stack Setup</h3>
          <div class="space-y-3">
            <button
              v-for="stack in availableStacks"
              :key="stack.name"
              @click="configureStack(stack.name)"
              :class="[
                'w-full text-left p-4 rounded-lg border transition-all duration-200',
                selectedStack === stack.name
                  ? 'border-purple-400 bg-purple-900/30'
                  : 'border-slate-600 bg-slate-700/30 hover:border-slate-500 hover:bg-slate-700/50'
              ]"
            >
              <div class="font-medium text-white">{{ stack.name }}</div>
              <div class="text-sm text-slate-300 mt-1">{{ stack.description }}</div>
              <div class="text-xs text-slate-400 mt-1">{{ stack.size }}</div>
            </button>
          </div>
        </div>

        <!-- Stack Analysis -->
        <div class="bg-slate-800/50 backdrop-blur rounded-xl border border-slate-700 p-6">
          <h3 class="text-xl font-semibold mb-4 text-green-400">Current Stack</h3>
          <div v-if="stackAnalysis">
            <div class="space-y-3">
              <div>
                <span class="text-slate-300">Type:</span>
                <span class="ml-2 font-medium">{{ stackAnalysis.current_stack }}</span>
              </div>
              <div>
                <span class="text-slate-300">Packages:</span>
                <span class="ml-2 font-medium">{{ stackAnalysis.installed_packages?.length || 0 }}</span>
              </div>
              <div>
                <span class="text-slate-300">Size:</span>
                <span class="ml-2 font-medium">{{ stackAnalysis.total_size }}</span>
              </div>
            </div>
            <button
              @click="refreshStackAnalysis"
              class="mt-4 w-full bg-green-600 hover:bg-green-700 text-white py-2 px-4 rounded-lg transition-colors"
            >
              Refresh Analysis
            </button>
          </div>
          <div v-else class="text-center">
            <button
              @click="refreshStackAnalysis"
              class="bg-green-600 hover:bg-green-700 text-white py-2 px-4 rounded-lg transition-colors"
            >
              Analyze Current Stack
            </button>
          </div>
        </div>

        <!-- Dependency Audit -->
        <div class="bg-slate-800/50 backdrop-blur rounded-xl border border-slate-700 p-6">
          <h3 class="text-xl font-semibold mb-4 text-orange-400">Optimization</h3>
          <div v-if="dependencyAudit">
            <div class="space-y-3">
              <div>
                <span class="text-slate-300">Heavy packages:</span>
                <span class="ml-2 font-medium">{{ dependencyAudit.heavy_packages?.length || 0 }}</span>
              </div>
              <div>
                <span class="text-slate-300">Unused:</span>
                <span class="ml-2 font-medium">{{ dependencyAudit.unused_packages?.length || 0 }}</span>
              </div>
              <div>
                <span class="text-slate-300">Total size:</span>
                <span class="ml-2 font-medium">{{ dependencyAudit.total_size }}</span>
              </div>
            </div>
            <button
              @click="refreshDependencyAudit"
              class="mt-4 w-full bg-orange-600 hover:bg-orange-700 text-white py-2 px-4 rounded-lg transition-colors"
            >
              Re-audit
            </button>
          </div>
          <div v-else class="text-center">
            <button
              @click="refreshDependencyAudit"
              class="bg-orange-600 hover:bg-orange-700 text-white py-2 px-4 rounded-lg transition-colors"
            >
              Audit Dependencies
            </button>
          </div>
        </div>
      </div>

      <!-- Chat Interface -->
      <div class="bg-slate-800/50 backdrop-blur rounded-xl border border-slate-700">
        <!-- Chat Messages -->
        <div class="h-96 overflow-y-auto p-6 space-y-4" ref="chatContainer">
          <div v-if="messages.length === 0" class="text-center text-slate-400 mt-16">
            <div class="text-6xl mb-4">🤖</div>
            <h3 class="text-xl font-semibold mb-2">RAG Engine AI Assistant</h3>
            <p>Ask me anything about stack configuration, package optimization, or RAG setup!</p>
            <div class="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto">
              <button
                v-for="suggestion in quickQuestions"
                :key="suggestion"
                @click="askQuestion(suggestion)"
                class="text-left p-3 bg-slate-700/50 hover:bg-slate-700 rounded-lg border border-slate-600 transition-colors"
              >
                <div class="text-sm text-slate-300">{{ suggestion }}</div>
              </button>
            </div>
          </div>
          
          <div
            v-for="message in messages"
            :key="message.id"
            :class="[
              'flex',
              message.type === 'user' ? 'justify-end' : 'justify-start'
            ]"
          >
            <div
              :class="[
                'max-w-3xl px-4 py-3 rounded-lg',
                message.type === 'user'
                  ? 'bg-purple-600 text-white ml-16'
                  : 'bg-slate-700 text-slate-100 mr-16'
              ]"
            >
              <div v-if="message.type === 'assistant'" class="flex items-center mb-2">
                <span class="text-blue-400 mr-2">🤖</span>
                <span class="text-sm text-slate-300">AI Assistant</span>
              </div>
              <div class="whitespace-pre-wrap">{{ message.content }}</div>
              <div v-if="message.context" class="mt-2 p-2 bg-slate-600/50 rounded text-xs">
                <div class="font-medium text-slate-300 mb-1">Context applied:</div>
                <div class="text-slate-400">{{ JSON.stringify(message.context, null, 2) }}</div>
              </div>
            </div>
          </div>
          
          <div v-if="isLoading" class="flex justify-start">
            <div class="bg-slate-700 text-slate-100 px-4 py-3 rounded-lg mr-16">
              <div class="flex items-center">
                <span class="text-blue-400 mr-2">🤖</span>
                <span class="text-sm text-slate-300">AI Assistant is thinking...</span>
              </div>
              <div class="flex space-x-1 mt-2">
                <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
              </div>
            </div>
          </div>
        </div>

        <!-- Chat Input -->
        <div class="border-t border-slate-700 p-6">
          <div class="flex space-x-4">
            <input
              v-model="currentQuestion"
              @keyup.enter="sendMessage"
              placeholder="Ask about stack configuration, optimization, or any RAG Engine question..."
              class="flex-1 bg-slate-700 border border-slate-600 rounded-lg px-4 py-3 text-white placeholder-slate-400 focus:ring-2 focus:ring-purple-400 focus:border-transparent"
              :disabled="isLoading"
            />
            <button
              @click="sendMessage"
              :disabled="isLoading || !currentQuestion.trim()"
              class="bg-purple-600 hover:bg-purple-700 disabled:bg-slate-600 text-white px-6 py-3 rounded-lg font-medium transition-colors"
            >
              Send
            </button>
          </div>
        </div>
      </div>

      <!-- Status Messages -->
      <div v-if="statusMessage" class="mt-6">
        <div
          :class="[
            'p-4 rounded-lg border',
            statusMessage.type === 'success'
              ? 'bg-green-900/50 border-green-600 text-green-200'
              : statusMessage.type === 'error'
              ? 'bg-red-900/50 border-red-600 text-red-200'
              : 'bg-blue-900/50 border-blue-600 text-blue-200'
          ]"
        >
          {{ statusMessage.text }}
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import api from '../services/api'

export default {
  name: 'AIAssistant',
  data() {
    return {
      selectedModel: 'phi3.5:latest',
      selectedStack: null,
      currentQuestion: '',
      messages: [],
      isLoading: false,
      statusMessage: null,
      stackAnalysis: null,
      dependencyAudit: null,
      availableStacks: [
        {
          name: 'DEMO',
          description: 'Quick demos with minimal dependencies',
          size: '~200MB'
        },
        {
          name: 'LOCAL',
          description: 'Local development with multiple vector stores',
          size: '~500MB'
        },
        {
          name: 'CLOUD',
          description: 'Cloud APIs only, minimal local processing',
          size: '~100MB'
        },
        {
          name: 'MINI',
          description: 'Ultra minimal for embedded systems',
          size: '~50MB'
        },
        {
          name: 'FULL',
          description: 'All features for production use',
          size: '~1GB'
        },
        {
          name: 'RESEARCH',
          description: 'Cutting-edge research with experimental models',
          size: '~1GB+'
        }
      ],
      quickQuestions: [
        "Which stack should I choose for my use case?",
        "How can I reduce my dependency bloat?",
        "What's the difference between LOCAL and CLOUD stacks?",
        "How do I optimize my current setup?"
      ]
    }
  },
  async mounted() {
    // Initialize with a welcome message
    this.addMessage('assistant', 'Hello! I\'m your RAG Engine AI assistant. I can help you choose the right stack, optimize dependencies, and configure your setup. What would you like to know?')
  },
  methods: {
    addMessage(type, content, context = null) {
      this.messages.push({
        id: Date.now(),
        type,
        content,
        context,
        timestamp: new Date()
      })
      this.$nextTick(() => {
        this.scrollToBottom()
      })
    },

    scrollToBottom() {
      if (this.$refs.chatContainer) {
        this.$refs.chatContainer.scrollTop = this.$refs.chatContainer.scrollHeight
      }
    },

    async sendMessage() {
      if (!this.currentQuestion.trim() || this.isLoading) return

      const question = this.currentQuestion.trim()
      this.currentQuestion = ''

      this.addMessage('user', question)
      await this.askAssistant(question)
    },

    async askQuestion(question) {
      this.currentQuestion = question
      await this.sendMessage()
    },

    async askAssistant(question) {
      this.isLoading = true
      try {
        // Gather context
        const context = {
          stackAnalysis: this.stackAnalysis,
          dependencyAudit: this.dependencyAudit,
          selectedStack: this.selectedStack,
          timestamp: new Date().toISOString()
        }

        const response = await api.askAssistant(question, context, this.selectedModel)
        
        if (response.status === 'success') {
          this.addMessage('assistant', response.response, response.context)
        } else {
          this.addMessage('assistant', `Error: ${response.response}`)
        }
      } catch (error) {
        console.error('Error asking assistant:', error)
        this.addMessage('assistant', 'Sorry, I encountered an error. Please try again.')
      } finally {
        this.isLoading = false
      }
    },

    async configureStack(stackType) {
      this.selectedStack = stackType
      this.showStatus(`Configuring ${stackType} stack...`, 'info')
      
      try {
        const response = await api.configureStack(stackType)
        
        if (response.status === 'success') {
          this.showStatus(response.message, 'success')
          this.addMessage('assistant', `I've configured the ${stackType} stack for you! ${response.message}`)
          
          // Refresh analysis after configuration
          await this.refreshStackAnalysis()
        } else {
          this.showStatus(`Error: ${response.message}`, 'error')
        }
      } catch (error) {
        console.error('Error configuring stack:', error)
        this.showStatus('Failed to configure stack', 'error')
      }
    },

    async refreshStackAnalysis() {
      try {
        this.stackAnalysis = await api.analyzeStack()
      } catch (error) {
        console.error('Error analyzing stack:', error)
        this.showStatus('Failed to analyze stack', 'error')
      }
    },

    async refreshDependencyAudit() {
      try {
        this.dependencyAudit = await api.auditDependencies()
      } catch (error) {
        console.error('Error auditing dependencies:', error)
        this.showStatus('Failed to audit dependencies', 'error')
      }
    },

    showStatus(text, type = 'info') {
      this.statusMessage = { text, type }
      setTimeout(() => {
        this.statusMessage = null
      }, 5000)
    }
  }
}
</script>
