import * as dotenv from 'dotenv';
import * as path from 'path';

// Load environment variables from .env.local
const envPath = path.resolve(process.cwd(), '.env.local');
console.log('Loading environment variables from:', envPath);
const result = dotenv.config({ path: envPath });

if (result.error) {
  console.error('Error loading .env.local file:', result.error);
  process.exit(1);
}

// Validate required environment variables
const requiredEnvVars = ['FIRECRAWL_KEY', 'OPENAI_KEY', 'GOOGLE_KEY'];
const missingEnvVars = requiredEnvVars.filter(envVar => !process.env[envVar]);

if (missingEnvVars.length > 0) {
  console.error('Missing required environment variables:', missingEnvVars.join(', '));
  process.exit(1);
}

// Log the first few characters of each key for verification
console.log('Environment variables loaded:');
requiredEnvVars.forEach(key => {
  const value = process.env[key];
  console.log(`- ${key}: ${value ? value.substring(0, 8) + '...' : 'not set'}`);
}); 