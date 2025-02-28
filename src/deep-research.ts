import FirecrawlApp, { SearchResponse } from '@mendable/firecrawl-js';
import search from './duckduckgo-search';
import { generateObject } from 'ai';
import { compact } from 'lodash-es';
import pLimit from 'p-limit';
import { z } from 'zod';
import * as fs from 'fs/promises';

import { trimPrompt } from './ai/providers';
import { systemPrompt } from './prompt';
import { OutputManager } from './output-manager';

// Initialize output manager for coordinated console/progress output
const output = new OutputManager();

// Replace console.log with output.log
function log(...args: any[]) {
  output.log(...args);
}

export type ResearchProgress = {
  currentDepth: number;
  totalDepth: number;
  currentBreadth: number;
  totalBreadth: number;
  currentQuery?: string;
  totalQueries: number;
  completedQueries: number;
};

type ResearchResult = {
  learnings: string[];
  visitedUrls: string[];
};

// increase this if you have higher API rate limits
const ConcurrencyLimit = 2;

// Initialize Firecrawl with optional API key and optional base url
if (!process.env.FIRECRAWL_KEY) {
  throw new Error('FIRECRAWL_KEY environment variable is not set or is empty');
}

log('Initializing Firecrawl with API key:', process.env.FIRECRAWL_KEY.substring(0, 8) + '...');

const firecrawl = new FirecrawlApp({
  apiKey: process.env.FIRECRAWL_KEY,
  apiUrl: process.env.FIRECRAWL_BASE_URL,
});

// take en user query, return a list of SERP queries
async function generateSerpQueries({
  query,
  numQueries = 3,
  learnings,
  researchLanguage,
}: {
  query: string;
  numQueries?: number;

  // optional, if provided, the research will continue from the last learning
  learnings?: string[];
  researchLanguage: string;
}) {
  const res = await generateObject({
    model: global.selectedModel,
    system: systemPrompt(),
    prompt: `
      Given the following prompt from the user, generate a list of SERP queries to research the topic.
      The queries should be in ${researchLanguage}.
      Return a maximum of ${numQueries} queries, but feel free to return less if the original prompt is clear.
      Make sure each query is unique and not similar to each other:

      <prompt>${query}</prompt>

      ${
        learnings
          ? `Here are some learnings from previous research, use them to generate more specific queries:
           ${learnings.join('\n')}`
          : ''
      }
    `,
    schema: z.object({
      queries: z
        .array(
          z.object({
            query: z.string().describe('The SERP query'),
            researchGoal: z
              .string()
              .describe(
                'First talk about the goal of the research that this query is meant to accomplish, then go deeper into how to advance the research once the results are found, mention additional research directions. Be as specific as possible, especially for additional research directions.',
              ),
          }),
        )
        .describe(`List of SERP queries, max of ${numQueries}`),
    }),
  });
  log(`Created ${res.object.queries.length} queries`, res.object.queries);

  return res.object.queries.slice(0, numQueries);
}

import sanitize from 'sanitize-filename';
import path from 'path';
export function urlToFilepath(url: string): string {
  return path.join('downloaded-urls', `${sanitize(url, { replacement: '-' })}.md`);
}

async function processSerpResult({
  query,
  result,
  numLearnings = 3,
  numFollowUpQuestions = 3,
  researchLanguage,
}: {
  query: string;
  result: SearchResponse;
  numLearnings?: number;
  numFollowUpQuestions?: number;
  researchLanguage: string;
}) {
  // Create downloaded-urls directory if it doesn't exist
  await fs.mkdir('downloaded-urls', { recursive: true });

  // Save each document
  for (const doc of result.data) {
    if (doc.markdown && doc.url) {
      const content = [
        doc.title ? `Title: ${doc.title}` : '',
        doc.description ? `Description: ${doc.description}` : '',
        `URL: ${doc.url}`,
        `Accessed at: ${Date()}`,
        '',
        doc.markdown
      ].filter(Boolean).join('\n');

      await fs.writeFile(urlToFilepath(doc.url), content, 'utf-8');
    }
  }

  const contents = compact(result.data.map(item => item.markdown)).map(
    content => trimPrompt(content, 25_000),
  );
  log(`Ran "${query}", found ${contents.length} contents`);

  const res = await generateObject({
    model: global.selectedModel,
    abortSignal: AbortSignal.timeout(60_000),
    system: systemPrompt(),
    prompt: `
      Given the following contents from a SERP search for the query <query>${query}</query>, generate a list of learnings from the contents.
      The learnings and follow-up questions should be in ${researchLanguage}.
      Return a maximum of ${numLearnings} learnings, but feel free to return less if the contents are clear.
      Make sure each learning is unique and not similar to each other.
      The learnings should be concise and to the point, as detailed and information dense as possible.
      Make sure to include any entities like people, places, companies, products, things, etc in the learnings,
      as well as any exact metrics, numbers, or dates. The learnings will be used to research the topic further.

      <contents>${contents.map(content => `<content>\n${content}\n</content>`).join('\n')}</contents>
    `,
    schema: z.object({
      learnings: z
        .array(z.string())
        .describe(`List of learnings, max of ${numLearnings}`),
      followUpQuestions: z
        .array(z.string())
        .describe(
          `List of follow-up questions to research the topic further, max of ${numFollowUpQuestions}`,
        ),
    }),
  });
  log(`Created ${res.object.learnings.length} learnings`, res.object.learnings);

  return res.object;
}

export async function writeFinalReport({
  prompt,
  learnings,
  visitedUrls,
  language = 'English',
}: {
  prompt: string;
  learnings: string[];
  visitedUrls: string[];
  language: string;
}) {
  const learningsString = trimPrompt(
    learnings
      .map(learning => `<learning>\n${learning}\n</learning>`)
      .join('\n'),
    150_000,
  );

  const res = await generateObject({
    model: global.selectedModel,
    system: systemPrompt(),
    prompt: `Given the following prompt from the user, write a final report on the topic using the learnings from research. Make it as detailed as possible, aim for 3 or more pages, include ALL the learnings from research. The report should be written in ${language}:\n\n<prompt>${prompt}</prompt>\n\nHere are all the learnings from previous research:\n\n<learnings>\n${learningsString}\n</learnings>`,
    schema: z.object({
      reportMarkdown: z
        .string()
        .describe(
          `Final report on the topic in Markdown, written in ${language}`,
        ),
    }),
  });

  // Append the visited URLs section to the report
  const urlsSection = `\n\n## Sources\n\n${visitedUrls.map(url => `- ${url}, saved at ${urlToFilepath(url)}`).join('\n')}`;
  return res.object.reportMarkdown + urlsSection;
}

export async function deepResearch({
  query,
  breadth,
  depth,
  learnings = [],
  visitedUrls = [],
  researchLanguage,
  onProgress,
}: {
  query: string;
  breadth: number;
  depth: number;
  learnings?: string[];
  visitedUrls?: string[];
  researchLanguage: string;
  onProgress?: (progress: ResearchProgress) => void;
}): Promise<ResearchResult> {
  const progress: ResearchProgress = {
    currentDepth: depth,
    totalDepth: depth,
    currentBreadth: breadth,
    totalBreadth: breadth,
    totalQueries: 0,
    completedQueries: 0,
  };

  const reportProgress = (update: Partial<ResearchProgress>) => {
    Object.assign(progress, update);
    onProgress?.(progress);
  };

  const serpQueries = await generateSerpQueries({
    query,
    learnings,
    numQueries: breadth,
    researchLanguage,
  });

  reportProgress({
    totalQueries: serpQueries.length,
    currentQuery: serpQueries[0]?.query,
  });

  const limit = pLimit(ConcurrencyLimit);

  const results = await Promise.all(
    serpQueries.map(serpQuery =>
      limit(async () => {
        try {
          let result = await firecrawl.search(serpQuery.query, {
            timeout: 15000,
            limit: 5,
            scrapeOptions: { formats: ['markdown'] },
          });

          // If the result is empty, use the DuckDuckGo search as a fallback
          if (result.data.length === 0) {
            log(
              `No results found for ${serpQuery.query}, falling back to DuckDuckGo search`,
            );
            result = await search(serpQuery.query, {
              timeout: 15000,
              limit: 5,
              scrapeOptions: { formats: ['markdown'] },
            });
          }

          // Collect URLs from this search
          const newUrls = compact(result.data.map(item => item.url));
          const newBreadth = Math.ceil(breadth / 2);
          const newDepth = depth - 1;

          const newLearnings = await processSerpResult({
            query: serpQuery.query,
            result,
            numFollowUpQuestions: newBreadth,
            researchLanguage,
          });
          const allLearnings = [...learnings, ...newLearnings.learnings];
          const allUrls = [...visitedUrls, ...newUrls];

          if (newDepth > 0) {
            log(
              `Researching deeper, breadth: ${newBreadth}, depth: ${newDepth}`,
            );

            reportProgress({
              currentDepth: newDepth,
              currentBreadth: newBreadth,
              completedQueries: progress.completedQueries + 1,
              currentQuery: serpQuery.query,
            });

            const nextQuery = `
              Previous research goal: ${serpQuery.researchGoal}
              Follow-up research directions: ${newLearnings.followUpQuestions.map(q => `\n${q}`).join('')}
            `.trim();

            return deepResearch({
              query: nextQuery,
              breadth: newBreadth,
              depth: newDepth,
              learnings: allLearnings,
              visitedUrls: allUrls,
              researchLanguage,
              onProgress,
            });
          } else {
            reportProgress({
              currentDepth: 0,
              completedQueries: progress.completedQueries + 1,
              currentQuery: serpQuery.query,
            });
            return {
              learnings: allLearnings,
              visitedUrls: allUrls,
            };
          }
        } catch (e: any) {
          if (e.message && e.message.includes('Timeout')) {
            log(`Timeout error running query: "${serpQuery.query}": `, e);
          } else {
            log(`Error running query: "${serpQuery.query}": `, e);
          }
          return {
            learnings: [],
            visitedUrls: [],
          };
        }
      }),
    ),
  );

  return {
    learnings: [...new Set(results.flatMap(r => r.learnings))],
    visitedUrls: [...new Set(results.flatMap(r => r.visitedUrls))],
  };
}
