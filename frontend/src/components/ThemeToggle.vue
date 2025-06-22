<template>
  <div class="relative">
    <button
      @click="toggleTheme"
      class="p-2 rounded-lg transition-all duration-200 group border bg-dark-surface border-dark-border hover:bg-dark-border dark:bg-dark-surface dark:border-dark-border dark:hover:bg-dark-border light:bg-light-surface light:border-light-border light:hover:bg-light-border"
      :title="isDark ? 'Switch to light mode' : 'Switch to dark mode'"
    >
      <SunIcon 
        v-if="isDark" 
        class="w-5 h-5 transition-colors text-dark-text-secondary group-hover:text-dark-text-primary dark:text-dark-text-secondary dark:group-hover:text-dark-text-primary light:text-light-text-secondary light:group-hover:text-light-text-primary" 
      />
      <MoonIcon 
        v-else 
        class="w-5 h-5 transition-colors text-light-text-secondary group-hover:text-light-text-primary dark:text-dark-text-secondary dark:group-hover:text-dark-text-primary light:text-light-text-secondary light:group-hover:text-light-text-primary" 
      />
    </button>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { SunIcon, MoonIcon } from '@heroicons/vue/24/outline'

const isDark = ref(true) // Default to dark mode

function toggleTheme() {
  isDark.value = !isDark.value
  updateTheme()
}

function updateTheme() {
  const html = document.documentElement
  
  if (isDark.value) {
    html.classList.add('dark')
    html.classList.remove('light')
    localStorage.setItem('theme', 'dark')
  } else {
    html.classList.remove('dark')
    html.classList.add('light')
    localStorage.setItem('theme', 'light')
  }
}

onMounted(() => {
  // Check for saved theme preference or default to dark
  const savedTheme = localStorage.getItem('theme')
  
  if (savedTheme) {
    isDark.value = savedTheme === 'dark'
  } else {
    // Default to dark mode for engineers
    isDark.value = true
  }
  
  updateTheme()
})
</script>
