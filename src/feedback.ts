import { generateObject, generateText, NoSuchToolError, InvalidToolArgumentsError, ToolExecutionError } from 'ai';
import { z } from 'zod';

import { o3MiniModel } from './ai/providers';
import { systemPrompt } from './prompt';

/**
 * Generates feedback questions based on a user query
 * @param query - The user's research query
 * @param numQuestions - Maximum number of questions to generate (default: 3)
 * @returns Array of follow-up questions
 */
export async function generateFeedback({
  query,
  numQuestions = 3,
  researchLanguage,
}: {
  query: string;
  numQuestions?: number;

}): Promise<string[]> {
  try {
    const result = await generateText({
      model: o3MiniModel,
      system: systemPrompt(),
      prompt: `Given the following query from the user, ask some follow up questions to clarify the research direction. 
      The questions should be in ${researchLanguage}.
      Return a maximum of ${numQuestions} questions, but feel free to return less if the original query is clear: <query>${query}</query>`,
      experimental_repairToolCall: async ({
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


        // Try to repair the tool call using a stronger model
        const result = await generateText({
          model: o3MiniModel,
          system,
          messages: [
            ...messages,
            {
              role: 'assistant',
              content: [
                {
                  type: 'tool-call',
                  toolCallId: toolCall.toolCallId,
                  toolName: toolCall.toolName,
                  args: toolCall.args,
                },
              ],
            },
            {
              role: 'tool' as const,
              content: [
                {
                  type: 'tool-result',
                  toolCallId: toolCall.toolCallId,
                  toolName: toolCall.toolName,
                  result: error.message,
                },
              ],
            },
          ],
          tools,
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
      },
    });

    // Extract questions from the result
    const questions = result.text
      .split('\n')
      .filter(line => line.trim().length > 0)
      .slice(0, numQuestions);

    return questions;
  } catch (error) {
    if (NoSuchToolError.isInstance(error)) {
      console.error('Tool not found:', error.message);
      return [`Could not generate questions due to missing tool: ${error.message}`];
    } else if (InvalidToolArgumentsError.isInstance(error)) {
      console.error('Invalid tool arguments:', error.message);
      return [`Could not generate questions due to invalid arguments: ${error.message}`];
    } else if (ToolExecutionError.isInstance(error)) {
      console.error('Tool execution error:', error.message);
      return [`Could not generate questions due to execution error: ${error.message}`];
    } else {
      console.error('Unexpected error:', error);
      return [`Could not generate questions due to an unexpected error`];
    }
  }
}
