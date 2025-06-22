<template>  <div class="flex justify-between py-2">
    <span class="text-sm font-medium text-dark-text-secondary dark:text-dark-text-secondary light:text-light-text-secondary">{{ label }}</span>
    <span class="text-sm font-mono text-dark-text-primary dark:text-dark-text-primary light:text-light-text-primary">{{ value }}</span>
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
