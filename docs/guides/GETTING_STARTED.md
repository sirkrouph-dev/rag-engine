# Getting Started with RAG Engine Frontend

> **⚠️ EXPERIMENTAL PROJECT ⚠️**
> 
> **This is a complete getting started guide for using the RAG Engine with its Vue.js frontend.**

## Prerequisites

- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **Git** for cloning the repository

## Setup Instructions

### 1. Clone and Setup Backend

```bash
# Clone the repository
git clone <repository-url>
cd rag_engine

# Install Python dependencies
pip install -r requirements.txt

# Test the installation
python -m rag_engine --help
```

### 2. Setup Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Verify installation
npm run dev --help
```

### 3. Start the Services

#### Terminal 1 - Backend
```bash
# From project root directory
python -m rag_engine serve --config examples/configs/example_config.json --port 8000
```

This starts the RAG Engine FastAPI server on `http://localhost:8000`

#### Terminal 2 - Frontend
```bash
# From frontend/ directory
npm run dev
```

This starts the Vue.js development server on `http://localhost:3000`

### 4. Access the Application

1. **Web Interface**: Open `http://localhost:3000` in your browser
2. **API Documentation**: Open `http://localhost:8000/docs` for FastAPI docs
3. **Health Check**: Visit `http://localhost:8000/health` to verify backend

## First Steps

### 1. Check System Status
- Open the web interface at `http://localhost:3000`
- You should see the dashboard with system status indicators
- The system health should show as "healthy" if everything is working

### 2. Build the Pipeline
- Go to the "Pipeline" tab
- Click "Build Pipeline" to initialize the RAG system
- Watch the build logs for progress
- Wait for completion (should show "Pipeline Built" status)

### 3. Try the Chat Interface
- Navigate to the "Chat" tab
- Try asking questions like:
  - "What is this document about?"
  - "Summarize the main points"
  - "What are the key findings?"

### 4. Explore Documents
- Visit the "Documents" tab to see loaded documents
- Browse document chunks and their metadata
- View statistics about your document collection

### 5. Monitor the System
- Check the "System" tab for detailed system information
- Test API endpoints using the built-in testing tools
- Monitor component status and health

## Configuration

### Backend Configuration
The backend uses JSON configuration files located in `examples/configs/`. Key files:

- `example_config.json` - Basic configuration for testing
- `production.json` - More comprehensive setup
- `vertex_ai_example.json` - Google Cloud Vertex AI integration

### Frontend Configuration
The frontend is configured via:

- `frontend/vite.config.js` - Build and development server config
- `frontend/tailwind.config.js` - Styling and theme configuration
- `frontend/package.json` - Dependencies and scripts

## Troubleshooting

### Backend Issues

**"Module not found" errors:**
```bash
# Ensure you're in the project root directory
pip install -r requirements.txt
```

**"Config file not found":**
```bash
# Check the config file path
ls examples/configs/
python -m rag_engine serve --config examples/configs/example_config.json
```

**Port already in use:**
```bash
# Use a different port
python -m rag_engine serve --config examples/configs/example_config.json --port 8001
```

### Frontend Issues

**"npm command not found":**
- Install Node.js from https://nodejs.org/

**Build fails:**
```bash
# Clear npm cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**API connection errors:**
- Ensure backend is running on port 8000
- Check browser network tab for failed requests
- Verify CORS is enabled in backend configuration

### Common Issues

**Pipeline build fails:**
- Check backend logs for specific errors
- Ensure all required environment variables are set
- Verify document paths in configuration are accessible

**Chat not working:**
- Ensure pipeline is built successfully
- Check backend status via API documentation
- Look for error messages in browser console

## Next Steps

### Development
- Explore the code in `rag_engine/` for backend components
- Check `frontend/src/` for frontend components
- Read the comprehensive documentation in `docs/`

### Customization
- Modify configurations in `examples/configs/`
- Customize the frontend theme in `frontend/tailwind.config.js`
- Add new components following existing patterns

### Production Deployment
- See `docs/deployment/` for production setup guides
- Configure environment variables for production
- Set up proper authentication and security measures

## Learn More

- [Frontend Guide](frontend/FRONTEND_GUIDE.md) - Complete frontend documentation
- [Project Structure](PROJECT_STRUCTURE.md) - Understanding the codebase
- [API Documentation](docs/api/) - Backend API details
- [Configuration Guide](docs/configuration.md) - Advanced configuration options

## Support

This is an experimental project with advanced features in testing. For issues:
1. Check the troubleshooting section above
2. Review the comprehensive documentation
3. Check browser console and backend logs for errors
4. Ensure all prerequisites are correctly installed
5. Note that some features are experimental and may have limitations

## Quick Reference

### Useful Commands
```bash
# Backend
python -m rag_engine serve --config CONFIG_FILE --port PORT
python -m rag_engine --help

# Frontend
npm run dev          # Development server
npm run build        # Production build
npm run preview      # Preview production build

# Health checks
curl http://localhost:8000/health
curl http://localhost:8000/status
```

### Default URLs
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
