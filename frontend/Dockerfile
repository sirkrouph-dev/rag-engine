# Frontend Dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Install a simple server to serve the built app
RUN npm install -g serve

# Expose port
EXPOSE 3001

# Serve the built app
CMD ["serve", "-s", "dist", "-l", "3001"]
