<template>
  <div 
    :class="[
      'flex',
      message.type === 'user' ? 'justify-end' : 'justify-start'
    ]"
  >    <div 
      :class="[
        'max-w-xs lg:max-w-md px-4 py-3 rounded-lg relative group',
        message.type === 'user' 
          ? 'text-white ml-auto bg-accent-primary-dark dark:bg-accent-primary-dark light:bg-accent-primary-light' 
          : message.type === 'error'
            ? 'border bg-accent-error-dark/10 text-accent-error-dark border-accent-error-dark/20 dark:bg-accent-error-dark/10 dark:text-accent-error-dark dark:border-accent-error-dark/20 light:bg-accent-error-light/10 light:text-accent-error-light light:border-accent-error-light/20'
            : 'shadow-sm border bg-dark-surface border-dark-border text-dark-text-primary dark:bg-dark-surface dark:border-dark-border dark:text-dark-text-primary light:bg-light-surface light:border-light-border light:text-light-text-primary'
      ]"
    >
      <!-- Message Content -->
      <div class="text-sm whitespace-pre-wrap">{{ message.content }}</div>
      
      <!-- Sources (for assistant messages) -->
      <div v-if="message.sources && message.sources.length > 0" class="mt-3 pt-3 border-t border-dark-border/20 dark:border-dark-border/20 light:border-light-border/20">
        <div class="text-xs mb-2 text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">Sources:</div>
        <div class="space-y-1">
          <div 
            v-for="(source, index) in message.sources" 
            :key="index"
            class="text-xs rounded px-2 py-1 bg-dark-bg dark:bg-dark-bg light:bg-light-bg"
          >
            {{ source.title || `Source ${index + 1}` }}
            <span v-if="source.page" class="text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">
              (Page {{ source.page }})
            </span>
          </div>
        </div>
      </div>
      
      <!-- Timestamp and Actions -->
      <div class="flex items-center justify-between mt-2 pt-2 border-t border-opacity-20 border-dark-border/20 dark:border-dark-border/20 light:border-light-border/20">
        <div 
          :class="[
            'text-xs',
            message.type === 'user' ? 'text-white/70' : 'text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary'
          ]"
        >
          {{ formatTime(message.timestamp) }}
        </div>
        <div class="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button 
            @click="copyMessage"
            :class="[
              'p-1 rounded hover:bg-gray-100 transition-colors',
              message.type === 'user' ? 'hover:bg-primary-500 hover:bg-opacity-20' : ''
            ]"
            title="Copy message"
          >
            <ClipboardIcon 
              :class="[
                'w-3 h-3',
                message.type === 'user' ? 'text-primary-200' : 'text-gray-400'
              ]" 
            />
          </button>
          <button 
            @click="$emit('delete', message.id)"
            :class="[
              'p-1 rounded hover:bg-red-100 transition-colors',
              message.type === 'user' ? 'hover:bg-red-500 hover:bg-opacity-20' : ''
            ]"
            title="Delete message"
          >
            <TrashIcon 
              :class="[
                'w-3 h-3',
                message.type === 'user' ? 'text-primary-200' : 'text-gray-400'
              ]" 
            />
          </button>
        </div>
      </div>
      
      <!-- Status indicator for assistant messages -->
      <div v-if="message.type === 'assistant' && message.status" class="absolute -bottom-1 -right-1">
        <div 
          :class="[
            'w-3 h-3 rounded-full border-2 border-white',
            message.status === 'success' ? 'bg-success-500' :
            message.status === 'error' ? 'bg-danger-500' :
            'bg-yellow-500'
          ]"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ClipboardIcon, TrashIcon } from '@heroicons/vue/24/outline'

defineProps({
  message: {
    type: Object,
    required: true
  }
})

defineEmits(['delete'])

function formatTime(timestamp) {
  return new Date(timestamp).toLocaleTimeString([], { 
    hour: '2-digit', 
    minute: '2-digit' 
  })
}

async function copyMessage() {
  try {
    await navigator.clipboard.writeText(message.content)
  } catch (error) {
    console.error('Failed to copy message:', error)
  }
}
</script>
