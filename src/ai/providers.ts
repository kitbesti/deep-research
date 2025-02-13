import { createOpenAI } from '@ai-sdk/openai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { createAzure } from '@ai-sdk/azure';
import { createMistral } from '@ai-sdk/mistral';
import { getEncoding } from 'js-tiktoken';

import { RecursiveCharacterTextSplitter } from './text-splitter';

interface CustomOpenAIProviderSettings {
  baseURL?: string;
}

// Add delay utility
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Providers
const openai = createOpenAI({
  apiKey: process.env.OPENAI_KEY!,
  baseURL: process.env.OPENAI_ENDPOINT || 'https://api.openai.com/v1',
} as CustomOpenAIProviderSettings);

const google = createGoogleGenerativeAI({
  apiKey: process.env.GOOGLE_KEY!,
});

const azure = createAzure({
  apiKey: process.env.AZURE_KEY!,
  resourceName: process.env.AZURE_RESOURCE_NAME,
});

const mistral = createMistral({
  apiKey: process.env.MISTRAL_KEY!,
});

// Default models for each provider
const customModel = process.env.OPENAI_MODEL || 'o3-mini';
const customGoogleModel = process.env.GOOGLE_MODEL || 'gemini-2.0-pro-exp-02-05';
const customAzureModel = process.env.AZURE_MODEL || 'gpt-4o-mini';
const customMistralModel = process.env.MISTRAL_MODEL || 'mistral-large-latest';

// Models
export const o3MiniModel = openai(customModel, {
  reasoningEffort: customModel.startsWith('o') ? 'medium' : undefined,
  structuredOutputs: true,
});

export const googleModel = google(customGoogleModel, {
  structuredOutputs: true,
});

export const azureModel = azure(customAzureModel, {
  structuredOutputs: true,
});

export const mistralModel = mistral(customMistralModel);

// Export a function to get the selected model
export function getSelectedModel(modelType: 'openai' | 'google' | 'azure' | 'mistral') {
  switch (modelType) {
    case 'openai':
      return o3MiniModel;
    case 'google':
      return googleModel;
    case 'azure':
      return azureModel;
    case 'mistral':
      return mistralModel;
    default:
      return o3MiniModel;
  }
}

const MinChunkSize = 140;
const encoder = getEncoding('o200k_base');

// trim prompt to maximum context size
export function trimPrompt(
  prompt: string,
  contextSize = Number(process.env.CONTEXT_SIZE) || 128_000,
) {
  if (!prompt) {
    return '';
  }

  const length = encoder.encode(prompt).length;
  if (length <= contextSize) {
    return prompt;
  }

  const overflowTokens = length - contextSize;
  // on average it's 3 characters per token, so multiply by 3 to get a rough estimate of the number of characters
  const chunkSize = prompt.length - overflowTokens * 3;
  if (chunkSize < MinChunkSize) {
    return prompt.slice(0, MinChunkSize);
  }

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap: 0,
  });
  const trimmedPrompt = splitter.splitText(prompt)[0] ?? '';

  // last catch, there's a chance that the trimmed prompt is same length as the original prompt, due to how tokens are split & innerworkings of the splitter, handle this case by just doing a hard cut
  if (trimmedPrompt.length === prompt.length) {
    return trimPrompt(prompt.slice(0, chunkSize), contextSize);
  }

  // recursively trim until the prompt is within the context size
  return trimPrompt(trimmedPrompt, contextSize);
}
