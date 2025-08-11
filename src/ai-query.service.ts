import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { MilvusService } from './milvus.service';
import { EmbeddingsService } from './embeddings.service';
import axios from 'axios';

export interface QueryResponse {
  answer: string;
  relevantChunks: string[];
  confidence: number;
}

@Injectable()
export class AiQueryService {
  private readonly logger = new Logger(AiQueryService.name);
  private readonly geminiApiKey: string;
  private readonly geminiApiUrl: string;

  constructor(
    private milvusService: MilvusService,
    private embeddingsService: EmbeddingsService,
    private configService: ConfigService,
  ) {
    this.geminiApiKey = this.configService.get<string>('GEMINI_API_KEY');
    this.geminiApiUrl = this.configService.get<string>('GEMINI_API_URL');
  }

  async processQuery(query: string): Promise<QueryResponse> {
    try {
      // Step 1: Generate embedding for the user query
      this.logger.log(`Processing query: ${query}`);
      const queryEmbedding = await this.embeddingsService.generateEmbedding(query);

      // Step 2: Search for relevant chunks in Milvus
      const searchResults = await this.milvusService.searchSimilar(queryEmbedding, 5);

      const relevantChunks = searchResults.map(result => result.text);
      this.logger.log(`Found ${relevantChunks.length} relevant chunks`);

      // Step 3: Prepare context for Gemini
      const context = relevantChunks.join('\n\n');

      // Step 4: Generate response using Gemini AI
      const answer = await this.generateGeminiResponse(query, context);

      // Step 5: Calculate confidence based on search scores
      const confidence = this.calculateConfidence(searchResults);

      return {
        answer,
        relevantChunks,
        confidence,
      };
    } catch (error) {
      this.logger.error('Error processing query:', error.message);
      throw new Error(`Failed to process query: ${error.message}`);
    }
  }

  private async generateGeminiResponse(query: string, context: string): Promise<string> {
    const prompt = `Based on the following context about a person, please answer the user's question accurately and concisely.

Context:
${context}

Question: ${query}

Please provide a helpful and accurate answer based only on the information provided in the context. If the context doesn't contain enough information to answer the question, please say so.

Answer:`;

    try {
      const response = await axios.post(
        `${this.geminiApiUrl}?key=${this.geminiApiKey}`,
        {
          contents: [
            {
              parts: [
                {
                  text: prompt,
                },
              ],
            },
          ],
          generationConfig: {
            temperature: 0.7,
            topK: 40,
            topP: 0.95,
            maxOutputTokens: 1024,
          },
        },
        {
          headers: {
            'Content-Type': 'application/json',
          },
        },
      );

      return response.data.candidates[0].content.parts[0].text;
    } catch (error) {
      this.logger.error('Error calling Gemini API:', error.message);
      throw new Error(`Failed to generate AI response: ${error.message}`);
    }
  }

  private calculateConfidence(searchResults: any[]): number {
    if (!searchResults || searchResults.length === 0) {
      return 0;
    }

    // Calculate confidence based on similarity scores
    const scores = searchResults.map(result => result.score || 0);
    const avgScore = scores.reduce((sum, score) => sum + score, 0) / scores.length;

    // Normalize score to percentage (assuming scores are between 0 and 1)
    return Math.min(Math.max(avgScore * 100, 0), 100);
  }
}
