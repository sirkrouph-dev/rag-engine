<template>
  <div 
    :class="[
      'flex',
      message.type === 'user' ? 'justify-end' : 'justify-start'
    ]"
  >
    <div 
      :class="[
        'max-w-xs lg:max-w-md px-4 py-3 rounded-lg relative group',
        message.type === 'user' 
          ? 'bg-primary-600 text-white ml-auto' 
          : message.type === 'error'
            ? 'bg-red-50 text-red-900 border border-red-200'
            : 'bg-white border border-gray-200 shadow-sm'
      ]"
    >
      <!-- Message Content -->
      <div class="text-sm whitespace-pre-wrap">{{ message.content }}</div>
      
      <!-- Sources (for assistant messages) -->
      <div v-if="message.sources && message.sources.length > 0" class="mt-3 pt-3 border-t border-gray-200">
        <div class="text-xs text-gray-600 mb-2">Sources:</div>
        <div class="space-y-1">
          <div 
            v-for="(source, index) in message.sources" 
            :key="index"
            class="text-xs bg-gray-50 rounded px-2 py-1"
          >
            {{ source.title || `Source ${index + 1}` }}
            <span v-if="source.page" class="text-gray-500">
              (Page {{ source.page }})
            </span>
          </div>
        </div>
      </div>
      
      <!-- Timestamp and Actions -->
      <div class="flex items-center justify-between mt-2 pt-2 border-t border-gray-200 border-opacity-20">
        <div 
          :class="[
            'text-xs',
            message.type === 'user' ? 'text-primary-200' : 'text-gray-500'
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
