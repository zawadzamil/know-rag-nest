import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { MilvusService } from './milvus.service';
import { EmbeddingsService } from './embeddings.service';
import axios from 'axios';

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
      'llama3.1:latest',
    );
    this.baseUrl = `http://${this.localLlmHost}:${this.localLlmPort}`;
  }

  async queryWithContext(query: string): Promise<string> {
    try {
      this.logger.log(`Processing query: "${query}"`);

      // Generate embedding for the query
      const queryEmbedding =
        await this.embeddingsService.generateEmbedding(query);

      // Search for similar chunks in Milvus
      const similarChunks = await this.milvusService.searchSimilar(
        queryEmbedding,
        5,
      );

      // Extract context text
      const contextTexts = similarChunks
        .map((chunk) => chunk.text || '')
        .filter((text) => text.length > 0);

      let context = contextTexts.join('\n\n');

      this.logger.log(`Found ${contextTexts.length} relevant context chunks`);
      this.logger.log(`Context preview: "${context.substring(0, 200)}..."`);

      // If no context found, try to get all chunks as fallback
      if (context.length === 0) {
        this.logger.warn('No similar chunks found, trying fallback approach');
        const allChunks = await this.milvusService.getAllChunks();
        const allTexts = allChunks
          .map((chunk) => chunk.text || '')
          .filter((text) => text.length > 0);
        context = allTexts.join('\n\n');
        this.logger.log(
          `Fallback: Using ${allTexts.length} total chunks as context`,
        );
      }

      // Generate response using local Llama model
      const response = await this.generateResponseWithLlama(query, context);

      return response;
    } catch (error) {
      this.logger.error('Error processing query:', error.message);
      throw new Error(`Failed to process query: ${error.message}`);
    }
  }

  private async generateResponseWithLlama(
    query: string,
    context: string,
  ): Promise<string> {
    try {
      // Debug: Log the context being used
      this.logger.log(`Context being used: "${context.substring(0, 200)}..."`);

      const prompt = `Answer the question directly using the provided context. Be concise and straightforward.

Context: ${context}

Question: ${query}

Answer directly:`;

      this.logger.log('Sending request to local Llama model...');
      this.logger.log(`Query: ${query}`);
      this.logger.log(`Context length: ${context.length} characters`);

      const response = await axios.post(
        `${this.baseUrl}/api/generate`,
        {
          model: this.localLlmModel,
          prompt: prompt,
          stream: false,
          options: {
            temperature: 0.1, // Very low temperature for consistent, direct responses
            num_predict: 100, // Shorter responses
            top_p: 0.9,
            stop: ['Question:', 'Context:', '\n\n'],
          },
        },
        {
          timeout: 120000,
          headers: {
            'Content-Type': 'application/json',
          },
        },
      );

      if (response.data?.response) {
        const answer = response.data.response.trim();
        this.logger.log('Successfully generated response with Llama model');
        return answer;
      } else {
        throw new Error('No response received from Llama model');
      }
    } catch (error) {
      this.logger.error('Error generating response with Llama:', error.message);
      throw new Error(`Failed to generate response: ${error.message}`);
    }
  }

  async testConnection(): Promise<boolean> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/tags`, {
        timeout: 5000,
      });

      this.logger.log('Successfully connected to Llama server');
      this.logger.log(`Available models: ${JSON.stringify(response.data)}`);
      return true;
    } catch (error) {
      this.logger.error(
        `Failed to connect to Llama server at ${this.baseUrl}:`,
        error.message,
      );
      return false;
    }
  }
}
