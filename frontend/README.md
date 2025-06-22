# RAG Engine Frontend

> **⚠️ EXPERIMENTAL PROJECT ⚠️**
> 
> **This is an experimental Vue.js frontend for the RAG Engine project.**

A modern, responsive Vue.js frontend for the RAG Engine with excellent UX and clean design.

## Features

- 🎯 **Modern Dashboard** - Overview of system status and quick actions
- 💬 **Interactive Chat** - Chat with your documents using the RAG pipeline
- ⚙️ **Pipeline Management** - Build and manage your RAG pipeline
- 📄 **Document Viewer** - Browse documents and chunks
- 🔧 **System Monitor** - Detailed system information and health

## Tech Stack

- **Vue 3** with Composition API
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Pinia** for state management
- **Vue Router** for navigation
- **Heroicons** for beautiful icons
- **Axios** for API communication

## Development

### Prerequisites

- Node.js 16+ and npm
- RAG Engine backend running on port 8000

### Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

The frontend will be available at `http://localhost:3000` and will proxy API requests to the backend at `http://localhost:8000`.

### Project Structure

```
frontend/
├── src/
│   ├── components/     # Reusable Vue components
│   ├── views/          # Page components
│   ├── stores/         # Pinia stores for state management
│   ├── services/       # API service layer
│   └── style.css       # Global styles and Tailwind
├── package.json        # Dependencies and scripts
└── vite.config.js      # Vite configuration
```

## API Integration

The frontend integrates with the RAG Engine FastAPI backend via:

- `/api/health` - Health checks
- `/api/status` - System status
- `/api/chat` - Chat with RAG system
- `/api/build` - Build pipeline
- `/api/documents` - Document management
- `/api/orchestrator/*` - Orchestrator management

## UI/UX Features

- **Responsive Design** - Works on desktop, tablet, and mobile
- **Real-time Updates** - Live system status and health indicators
- **Interactive Chat** - Message history, sources, and error handling
- **Loading States** - Smooth loading indicators and transitions
- **Error Handling** - User-friendly error messages and recovery
- **Accessibility** - Semantic HTML and keyboard navigation

## Development Notes

- Uses Vite proxy for API requests during development
- Tailwind CSS for rapid UI development
- Pinia stores for centralized state management
- Component composition for reusability
- Modern ES6+ JavaScript with Vue 3 Composition API
