import { generateObject } from 'ai';
import { z } from 'zod';

import { systemPrompt } from './prompt';

export async function generateFeedback({
  query,
  numQuestions = 3,
  researchLanguage,
}: {
  query: string;
  numQuestions?: number;
  researchLanguage: string;
}) {
  const options = {
    model: global.selectedModel,
    system: systemPrompt(),
    prompt: `
      Given the following query from the user, ask some follow up questions to clarify the research direction.
      The questions should be in ${researchLanguage}.
      Return a maximum of ${numQuestions} questions, but feel free to return less if the original query is clear:

      <query>${query}</query>
    `,
    schema: z.object({
      questions: z
        .array(z.string())
        .describe(
          `Follow up questions to clarify the research direction, max of ${numQuestions}`,
        ),
    }),
  };

  // Special handling for deepseek models which don't support JSON output
  if (global.selectedModel?.includes('deepseek')) {
    delete options.schema;
    const response = await generateObject({
      ...options,
      prompt: `${options.prompt}\nPlease format your response as a JSON array of strings containing the questions.`,
    });
    try {
      const parsed = JSON.parse(response as string);
      return { questions: Array.isArray(parsed) ? parsed : parsed.questions };
    } catch (e) {
      throw new Error('Failed to parse deepseek response as JSON');
    }
  }

  const userFeedback = await generateObject(options);

  return userFeedback.object.questions.slice(0, numQuestions);
}
