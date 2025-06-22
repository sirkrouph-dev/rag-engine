<template>
  <Teleport to="body">
    <div 
      v-if="systemStore.error" 
      class="fixed top-4 right-4 z-50 max-w-sm w-full"
    >
      <div class="bg-white rounded-lg shadow-lg border border-red-200 p-4 animate-slide-up">
        <div class="flex">
          <div class="flex-shrink-0">
            <ExclamationTriangleIcon class="h-5 w-5 text-red-400" />
          </div>
          <div class="ml-3">
            <h3 class="text-sm font-medium text-red-800">
              Error
            </h3>
            <div class="mt-1 text-sm text-red-700">
              {{ systemStore.error }}
            </div>
          </div>
          <div class="ml-auto pl-3">
            <div class="-mx-1.5 -my-1.5">
              <button
                @click="systemStore.error = null"
                class="inline-flex rounded-md bg-red-50 p-1.5 text-red-500 hover:bg-red-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-red-50 focus:ring-red-600"
              >
                <XMarkIcon class="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup>
import { ExclamationTriangleIcon, XMarkIcon } from '@heroicons/vue/24/outline'
import { useSystemStore } from '../stores/system'

const systemStore = useSystemStore()
</script>
