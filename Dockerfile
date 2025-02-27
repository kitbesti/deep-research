FROM node:22-alpine

WORKDIR /app

# Set platform for native binaries
ENV npm_config_platform=linux

# Copy package files first
COPY package*.json ./

# Clean install dependencies for Linux
RUN npm ci --force && \
    npm rebuild esbuild --platform=linux --force

# Copy application code
COPY . .
COPY .env.local ./.env.local

CMD ["npm", "run", "docker"]
