import { defineStore } from 'pinia'
import { ref } from 'vue'
import api from '../services/api'

export const useChatStore = defineStore('chat', () => {
  // State
  const messages = ref([])
  const isLoading = ref(false)
  const sessionId = ref(null)
  const error = ref(null)

  // Actions
  async function sendMessage(query) {
    if (!query.trim()) return

    // Add user message
    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: query,
      timestamp: new Date()
    }
    messages.value.push(userMessage)

    try {
      isLoading.value = true
      error.value = null

      const response = await api.sendMessage(query, sessionId.value)
      
      // Update session ID if provided
      if (response.session_id) {
        sessionId.value = response.session_id
      }

      // Add assistant response
      const assistantMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: response.answer || response.response,
        sources: response.sources || [],
        timestamp: new Date(),
        status: response.status
      }
      messages.value.push(assistantMessage)

    } catch (err) {
      error.value = err.message
      
      // Add error message
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: 'Sorry, I encountered an error while processing your question. Please try again.',
        timestamp: new Date()
      }
      messages.value.push(errorMessage)
    } finally {
      isLoading.value = false
    }
  }

  function clearChat() {
    messages.value = []
    sessionId.value = null
    error.value = null
  }

  function deleteMessage(messageId) {
    const index = messages.value.findIndex(msg => msg.id === messageId)
    if (index > -1) {
      messages.value.splice(index, 1)
    }
  }

  return {
    // State
    messages,
    isLoading,
    sessionId,
    error,
    
    // Actions
    sendMessage,
    clearChat,
    deleteMessage
  }
})
