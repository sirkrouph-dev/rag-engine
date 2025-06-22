<template>
  <div class="flex justify-between py-2">
    <span class="text-sm font-medium text-gray-700">{{ label }}</span>
    <span class="text-sm text-gray-900 font-mono">{{ value }}</span>
  </div>
</template>

<script setup>
defineProps({
  label: {
    type: String,
    required: true
  },
  value: {
    type: [String, Number],
    required: true
  }
})
</script>
