import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios from 'axios';
import { MilvusService } from './milvus.service';
import { EmbeddingsService } from './embeddings.service';

export interface QueryResponse {
  answer: string;
  sources: string[];
  confidence: number;
}

@Injectable()
export class AiQueryService {
  private readonly logger = new Logger(AiQueryService.name);
  private readonly localLlmHost: string;
  private readonly localLlmPort: string;
  private readonly localLlmModel: string;
  private readonly baseUrl: string;

  constructor(
    private configService: ConfigService,
    private milvusService: MilvusService,
    private embeddingsService: EmbeddingsService,
  ) {
    this.localLlmHost = this.configService.get<string>(
      'LOCAL_LLM_HOST',
      'localhost',
    );
    this.localLlmPort = this.configService.get<string>(
      'LOCAL_LLM_PORT',
      '11434',
    );
    this.localLlmModel = this.configService.get<string>(
      'LOCAL_LLM_MODEL',
      'llama3.1',
    );
    this.baseUrl = `http://${this.localLlmHost}:${this.localLlmPort}`;
  }

  async processQuery(question: string): Promise<QueryResponse> {
    try {
      this.logger.log(`Processing query: ${question}`);

      // Generate embedding for the question
      const queryEmbedding =
        await this.embeddingsService.generateEmbedding(question);

      // Search for relevant context in Milvus
      const searchResults = await this.milvusService.searchSimilar(
        queryEmbedding,
        5,
      );

      // Extract context from search results
      const contextTexts = searchResults.map(
        (result) => result.text || result.entity?.text || '',
      );
      const context = contextTexts
        .filter((text) => text.length > 0)
        .join('\n\n');

      // Generate response using local LLaMA model
      const answer = await this.generateResponse(question, context);

      return {
        answer,
        sources: contextTexts.filter((text) => text.length > 0),
        confidence: this.calculateConfidence(searchResults),
      };
    } catch (error) {
      this.logger.error('Error processing query:', error.message);
      throw new Error(`Failed to process query: ${error.message}`);
    }
  }

  private async generateResponse(
    question: string,
    context: string,
  ): Promise<string> {
    const prompt = this.buildPrompt(question, context);

    try {
      // Try Ollama API format first
      const response = await axios.post(
        `${this.baseUrl}/api/generate`,
        {
          model: this.localLlmModel,
          prompt: prompt,
          stream: false,
          options: {
            temperature: 0.7,
            max_tokens: 500,
          },
        },
        {
          timeout: 60000,
          headers: {
            'Content-Type': 'application/json',
          },
        },
      );

      if (response.data?.response) {
        this.logger.log(
          'Successfully generated response using local LLaMA model',
        );
        return response.data.response.trim();
      }

      throw new Error('No response from local model');
    } catch (error) {
      this.logger.warn(`Local LLaMA model error: ${error.message}`);

      // Fallback to simple context-based response
      return this.generateFallbackResponse(question, context);
    }
  }

  private buildPrompt(question: string, context: string): string {
    return `You are a helpful assistant. Use the following context to answer the user's question. If the context doesn't contain relevant information, say so clearly.

Context:
${context}

Question: ${question}

Answer:`;
  }

  private generateFallbackResponse(question: string, context: string): string {
    if (!context || context.trim().length === 0) {
      return "I don't have enough context information to answer your question accurately.";
    }

    // Simple keyword matching fallback
    const questionWords = question.toLowerCase().split(/\s+/);
    const contextSentences = context
      .split(/[.!?]+/)
      .filter((s) => s.trim().length > 0);

    const relevantSentences = contextSentences.filter((sentence) => {
      const sentenceLower = sentence.toLowerCase();
      return questionWords.some(
        (word) => word.length > 3 && sentenceLower.includes(word),
      );
    });

    if (relevantSentences.length > 0) {
      return `Based on the available context: ${relevantSentences.slice(0, 3).join('. ')}.`;
    }

    return `I found some context information, but it may not directly answer your question: ${context.substring(0, 200)}...`;
  }

  private calculateConfidence(searchResults: any[]): number {
    if (!searchResults || searchResults.length === 0) {
      return 0;
    }

    // Calculate confidence based on search scores
    const avgScore =
      searchResults.reduce((sum, result) => {
        return sum + (result.score || result.distance || 0);
      }, 0) / searchResults.length;

    // Normalize to 0-1 range (this is a simple heuristic)
    return Math.min(1, Math.max(0, avgScore));
  }
}
