import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import './style.css'

// Import views
import Dashboard from './views/Dashboard.vue'
import Chat from './views/Chat.vue'
import Pipeline from './views/Pipeline.vue'
import Documents from './views/Documents.vue'
import System from './views/System.vue'
import AIAssistant from './views/AIAssistant.vue'
import Routing from './views/Routing.vue'

// Router configuration
const routes = [
  { path: '/', component: Dashboard, name: 'dashboard' },
  { path: '/chat', component: Chat, name: 'chat' },
  { path: '/pipeline', component: Pipeline, name: 'pipeline' },
  { path: '/documents', component: Documents, name: 'documents' },
  { path: '/routing', component: Routing, name: 'routing' },
  { path: '/system', component: System, name: 'system' },
  { path: '/ai-assistant', component: AIAssistant, name: 'ai-assistant' }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// Create app
const app = createApp(App)
app.use(createPinia())
app.use(router)
app.mount('#app')
