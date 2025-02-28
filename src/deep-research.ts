import FirecrawlApp, { SearchResponse } from '@mendable/firecrawl-js';
import { generateObject, generateText, NoSuchToolError, InvalidToolArgumentsError, ToolExecutionError, type ToolCallRepairFunction, type ToolSet } from 'ai';
import { compact } from 'lodash-es';
import pLimit from 'p-limit';
import { z } from 'zod';

import { o3MiniModel, trimPrompt } from './ai/providers';
import { OutputManager } from './output-manager';
import { systemPrompt } from './prompt';

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

const firecrawl = new FirecrawlApp({
  apiKey: process.env.FIRECRAWL_KEY ?? '',
  apiUrl: process.env.FIRECRAWL_BASE_URL,
});

/**
 * Common tool call repair function that can be used across different generateObject calls
 */
const repairToolCall: ToolCallRepairFunction<ToolSet> = async ({
  toolCall,
  tools,
  parameterSchema,
  error,
  messages,
  system,
}) => {
  if (NoSuchToolError.isInstance(error)) {
    return null; // don't attempt to fix invalid tool names
  }

  try {
    // Try to repair the tool call using the same model
    const result = await generateText({
      model: o3MiniModel,
      system,
      messages,
      tools,
      output: 'no-schema',
    });

    const newToolCall = result.toolCalls.find(
      newToolCall => newToolCall.toolName === toolCall.toolName,
    );

    return newToolCall != null
      ? {
          toolCallType: 'function' as const,
          toolCallId: toolCall.toolCallId,
          toolName: toolCall.toolName,
          args: JSON.stringify(newToolCall.args),
        }
      : null;
  } catch (repairError) {
    log('Error during tool call repair:', repairError);
    return null;
  }
}

// take en user query, return a list of SERP queries
async function generateSerpQueries({
  query,
  numQueries = 3,
  learnings,
  researchLanguage,
}: {
  query: string;
  numQueries?: number;
  learnings?: string[];
  researchLanguage: string;
}) {

  try {
    const result = await generateText({
      model: o3MiniModel,
      system: systemPrompt(),
      prompt: `Given the following prompt from the user, generate a list of SERP queries to research the topic.
      The queries should be in ${researchLanguage}. Return a maximum of ${numQueries} queries in a structured format. 
      Each query should be on a new line starting with a number and include both the query and its research goal. 
      Make sure each query is unique and not similar to each other: 
      
      <prompt>${query}</prompt>\n\n${
        learnings
          ? `Here are some learnings from previous research, use them to generate more specific queries: ${learnings.join(
              '\n',
            )}`
          : ''
      }`,
      experimental_repairToolCall: repairToolCall,
    });

    // Parse the text output into structured queries
    const lines = result.text.split('\n').filter(line => line.trim());
    const queries = [];
    let currentQuery = null;

    for (const line of lines) {
      // Match lines that start with a number followed by a dot or parenthesis
      const queryMatch = line.match(/^\d+[\.\)]?\s*\*?\*?"?([^"]+)"?\*?\*?/);
      if (queryMatch) {
        if (currentQuery) {
          queries.push(currentQuery);
        }
        currentQuery = {
          query: queryMatch[1].trim(),
          researchGoal: '',
        };
      } else if (currentQuery && line.toLowerCase().includes('focus:')) {
        // Extract research goal after "Focus:"
        currentQuery.researchGoal = line.split('Focus:')[1].trim().replace(/^\*|\*$/g, '');
        queries.push(currentQuery);
        currentQuery = null;
      } else if (currentQuery && !line.startsWith('*') && !line.startsWith('-')) {
        // If no explicit "Focus:" but there's additional text, use it as the research goal
        currentQuery.researchGoal = line.trim().replace(/^\*|\*$/g, '');
        queries.push(currentQuery);
        currentQuery = null;
      }
    }

    // Add the last query if it exists
    if (currentQuery) {
      queries.push(currentQuery);
    }
    
    log(
      `Created ${queries.length} queries`,
      queries,
    );

    return queries.slice(0, numQueries);
  } catch (error) {
    log('Error generating SERP queries:', error);
    return [];
  }
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

  try {
    const contents = compact(result.data.map(item => item.markdown)).map(
      content => trimPrompt(content, 25_000),
    );
    log(`Ran ${query}, found ${contents.length} contents`);

    const res = await generateObject({
      model: o3MiniModel,
      abortSignal: AbortSignal.timeout(60_000_000),
      system: systemPrompt(),
      prompt: `Given the following contents from a SERP search for the query <query>${query}</query>, generate a list of learnings from the contents. 
      IMPORTANT RULES:
      1. ONLY use information that is explicitly stated in the provided contents.
      2. DO NOT make assumptions or add information from your own knowledge.
      3. Each learning MUST be directly supported by the search contents.
      4. If you're unsure about any information, exclude it.
      5. Use exact quotes and numbers from the contents when possible.
      6. Cite the specific content section for each learning using [Content X] at the end of each learning.

      The learnings and follow-up questions should be in ${researchLanguage}.
      Return a maximum of ${numLearnings} learnings, but feel free to return less if the contents are clear. 
      Make sure each learning is unique and not similar to each other. 
      The learnings should be concise and to the point, as detailed and information dense as possible. 
      Make sure to include any entities like people, places, companies, products, things, etc in the learnings, 
      as well as any exact metrics, numbers, or dates. The learnings will be used to research the topic further.\n\n<contents>${contents
        .map((content, idx) => `<content${idx + 1}>\n${content}\n</content${idx + 1}>`)
        .join('\n')}</contents>`,
      schema: z.object({
        followUpQuestions: z
          .array(z.string())
          .describe(
            `List of follow-up questions to research the topic further, max of ${numFollowUpQuestions}`,
          ),
        learnings: z
          .array(z.string())
          .describe(`List of learnings, max of ${numLearnings}, each with a content citation`),
      }),
      experimental_repairToolCall: repairToolCall,
    });

    try {
      return process.env.OPENAI_MODEL?.includes('deepseek') 
        ? JSON.parse(res.choices[0].message.content)
        : res.object;
    } catch (e) {
      return res.object;
    }
  } catch (error) {
    log('Error processing SERP result:', error);
    return { learnings: [], followUpQuestions: [] };
  }
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
  try {
    const learningsString = trimPrompt(
      learnings
        .map(learning => `<learning>\n${learning}\n</learning>`)
        .join('\n'),
      150_000,
    );

    // Use generateText instead of generateObject since we want markdown output
    const result = await generateText({
      model: o3MiniModel,
      system: systemPrompt(),
      prompt: `Given the following prompt from the user, write a final report on the topic using the learnings from research. Make it as as detailed as possible, aim for 3 or more pages, include ALL the learnings from research.

      IMPORTANT RULES:
      1. The report MUST be based ONLY on the provided learnings.
      2. DO NOT add any new information that is not in the learnings.
      3. DO NOT make assumptions or add information from your own knowledge.
      4. Maintain all citations from the learnings in your report.
      5. If you find any conflicts in the learnings, highlight them clearly.
      
      The report should be written in ${language}, and return the report in markdown format:\n\n<prompt>${prompt}</prompt>\n\nHere are all the learnings from previous research:\n\n<learnings>\n${learningsString}\n</learnings>`,
      schema: z.object({
        reportMarkdown: z
          .string()
          .describe(
            `Final report on the topic in Markdown, written in ${language}, strictly based on provided learnings`,
          ),
      }),
      experimental_repairToolCall: repairToolCall,
    });

    try {
      const report = process.env.OPENAI_MODEL?.includes('deepseek')
        ? JSON.parse(result.choices[0].message.content).reportMarkdown
        : result.text;

      // Append the visited URLs section to the report
      const urlsSection = `\n\n## Sources\n\n${visitedUrls.map(url => `- ${url}`).join('\n')}`;
      return report + urlsSection;
    } catch (e) {
      // 如果解析失败，回退到原有逻辑
      const urlsSection = `\n\n## Sources\n\n${visitedUrls.map(url => `- ${url}`).join('\n')}`;
      return result.text + urlsSection;
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    log('Error writing final report:', error);
    return `Error generating report: ${errorMessage}\n\n## Sources\n\n${visitedUrls.map(url => `- ${url}`).join('\n')}`;
  }
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
          const result = await firecrawl.search(serpQuery.query, {
            timeout: 15000,
            limit: 5,
            scrapeOptions: { formats: ['markdown'] },
          });

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
