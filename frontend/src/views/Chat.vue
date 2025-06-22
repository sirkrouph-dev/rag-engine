<template>  <div class="h-[calc(100vh-8rem)] flex flex-col">
    <!-- Header -->
    <div class="mb-6">
      <h1 class="text-3xl font-bold text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">Chat</h1>
      <p class="mt-2 text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">
        Ask questions about your documents using the RAG Engine
      </p>
    </div>

    <!-- Chat Container -->
    <div class="flex-1 card flex flex-col">
      <!-- Pipeline Status Check -->
      <div v-if="!systemStore.isPipelineBuilt" class="p-6 border-b border-dark-border dark:border-dark-border light:border-light-border">
        <div class="rounded-md p-4 bg-yellow-500/10 border border-yellow-500/20">
          <div class="flex">
            <div class="flex-shrink-0">
              <ExclamationTriangleIcon class="h-5 w-5 text-yellow-400" />
            </div>
            <div class="ml-3">
              <h3 class="text-sm font-medium text-yellow-300">
                Pipeline Not Ready
              </h3>
              <div class="mt-2 text-sm text-yellow-400/80">
                <p>The RAG pipeline needs to be built before you can start chatting.</p>
              </div>
              <div class="mt-4">
                <button 
                  @click="buildPipeline" 
                  :disabled="systemStore.isLoading"
                  class="btn btn-sm"
                  style="background-color: rgba(251, 191, 36, 0.2); color: #fbbf24; border-color: rgba(251, 191, 36, 0.3);"
                >
                  <CogIcon class="w-4 h-4 mr-2" />
                  Build Pipeline
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Messages Area -->
      <div 
        ref="messagesContainer"
        class="flex-1 overflow-y-auto p-6 space-y-4"
        :class="{ 'opacity-50': !systemStore.isPipelineBuilt }"
      >
        <!-- Welcome Message -->
        <div v-if="chatStore.messages.length === 0" class="text-center py-8">
          <ChatBubbleLeftRightIcon class="mx-auto h-12 w-12 mb-4 text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary" />          <h3 class="text-lg font-medium mb-2 text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">Start a conversation</h3>
          <p class="mb-6 max-w-sm mx-auto text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">
            Ask questions about your documents and I'll help you find answers using the RAG Engine.
          </p>
          <div class="flex flex-wrap gap-2 justify-center">
            <button 
              v-for="suggestion in suggestions" 
              :key="suggestion"
              @click="sendSuggestion(suggestion)"
              :disabled="!systemStore.isPipelineBuilt"
              class="btn btn-sm btn-secondary"
            >
              {{ suggestion }}
            </button>
          </div>
        </div>

        <!-- Messages -->
        <div v-for="message in chatStore.messages" :key="message.id" class="animate-fade-in">
          <ChatMessage :message="message" @delete="chatStore.deleteMessage" />
        </div>

        <!-- Loading Indicator -->
        <div v-if="chatStore.isLoading" class="flex justify-start">
          <div class="max-w-xs lg:max-w-md px-4 py-3 rounded-lg bg-dark-surface dark:bg-dark-surface light:bg-light-surface">
            <div class="flex items-center space-x-2">
              <div class="flex space-x-1">
                <div class="w-2 h-2 rounded-full animate-bounce bg-dark-text-secondary dark:bg-dark-text-secondary light:bg-light-text-secondary" style="animation-delay: 0ms"></div>
                <div class="w-2 h-2 rounded-full animate-bounce bg-dark-text-secondary dark:bg-dark-text-secondary light:bg-light-text-secondary" style="animation-delay: 150ms"></div>
                <div class="w-2 h-2 rounded-full animate-bounce bg-dark-text-secondary dark:bg-dark-text-secondary light:bg-light-text-secondary" style="animation-delay: 300ms"></div>
              </div>
              <span class="text-sm text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">Thinking...</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Input Area -->
      <div class="border-t p-6 border-dark-border dark:border-dark-border light:border-light-border">
        <form @submit.prevent="sendMessage" class="flex space-x-4">
          <div class="flex-1">
            <input
              v-model="currentMessage"
              type="text"
              placeholder="Ask a question about your documents..."
              :disabled="!systemStore.isPipelineBuilt || chatStore.isLoading"
              class="input"
              @keydown.enter.prevent="sendMessage"
            />
          </div>
          <button
            type="submit"
            :disabled="!currentMessage.trim() || !systemStore.isPipelineBuilt || chatStore.isLoading"
            class="btn btn-primary px-6"
          >
            <PaperAirplaneIcon class="w-4 h-4 mr-2" />
            Send
          </button>
        </form>
        
        <!-- Chat Actions -->
        <div class="flex justify-between items-center mt-4">
          <div class="flex space-x-2">
            <button 
              @click="chatStore.clearChat"
              :disabled="chatStore.messages.length === 0"
              class="btn btn-sm btn-secondary"
            >
              <TrashIcon class="w-4 h-4 mr-1" />
              Clear Chat
            </button>
          </div>          <div class="text-xs text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">
            {{ chatStore.messages.length }} messages
            <span v-if="chatStore.sessionId">
              • Session: {{ chatStore.sessionId.slice(-8) }}
            </span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, watch } from 'vue'
import { 
  ChatBubbleLeftRightIcon, 
  PaperAirplaneIcon, 
  TrashIcon,
  ExclamationTriangleIcon,
  CogIcon
} from '@heroicons/vue/24/outline'
import { useChatStore } from '../stores/chat'
import { useSystemStore } from '../stores/system'
import ChatMessage from '../components/ChatMessage.vue'

const chatStore = useChatStore()
const systemStore = useSystemStore()

const currentMessage = ref('')
const messagesContainer = ref(null)

const suggestions = [
  "What is this document about?",
  "Summarize the main points",
  "What are the key findings?",
  "Tell me about the methodology"
]

async function sendMessage() {
  if (!currentMessage.value.trim() || !systemStore.isPipelineBuilt) return
  
  const message = currentMessage.value
  currentMessage.value = ''
  
  await chatStore.sendMessage(message)
  await scrollToBottom()
}

async function sendSuggestion(suggestion) {
  if (!systemStore.isPipelineBuilt) return
  
  currentMessage.value = suggestion
  await sendMessage()
}

async function buildPipeline() {
  try {
    await systemStore.buildPipeline()
  } catch (error) {
    console.error('Failed to build pipeline:', error)
  }
}

async function scrollToBottom() {
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

// Auto-scroll when new messages are added
watch(() => chatStore.messages.length, () => {
  scrollToBottom()
})
</script>
