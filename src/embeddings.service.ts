import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios from 'axios';

@Injectable()
export class EmbeddingsService {
  private readonly logger = new Logger(EmbeddingsService.name);
  private readonly localLlmHost: string;
  private readonly localLlmPort: string;
  private readonly localLlmModel: string;
  private readonly baseUrl: string;

  constructor(private configService: ConfigService) {
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

  async generateEmbedding(
    text: string,
    retries: number = 3,
  ): Promise<number[]> {
    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        // Try to use dedicated embedding model first
        try {
          const embeddingResponse = await axios.post(
            `${this.baseUrl}/api/embeddings`,
            {
              model: 'nomic-embed-text', // Common embedding model
              prompt: text,
            },
            {
              timeout: 30000,
              headers: {
                'Content-Type': 'application/json',
              },
            },
          );

          if (embeddingResponse.data?.embedding) {
            this.logger.log(
              'Successfully generated embedding using dedicated model',
            );
            return embeddingResponse.data.embedding;
          }
        } catch (embeddingError) {
          this.logger.warn(
            `Dedicated embedding model not available: ${embeddingError.message}`,
          );
        }

        // Fallback: Use Llama model to generate text-based embedding
        try {
          this.logger.log(
            `Attempting to generate embedding using Llama model for text: "${text.substring(0, 50)}..."`,
          );

          const prompt = `You are a text embedding generator. Convert the following text into a numerical vector representation.

Text: "${text}"

Generate exactly 100 floating point numbers between -1.0 and 1.0, separated by commas. Each number should represent a semantic aspect of the text. Provide only the numbers, no other text.

Example format: 0.1, -0.3, 0.7, -0.2, 0.5, ...

Numbers:`;

          const response = await axios.post(
            `${this.baseUrl}/api/generate`,
            {
              model: this.localLlmModel,
              prompt: prompt,
              stream: false,
              options: {
                temperature: 0.1, // Low temperature for consistency
                max_tokens: 2000,
                top_p: 0.9,
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
              `Llama model response received: ${response.data.response.substring(0, 100)}...`,
            );

            try {
              // Parse the response to extract numbers
              const responseText = response.data.response;

              // Extract numbers using regex - look for decimal numbers
              const numberMatches = responseText.match(/-?\d*\.?\d+/g);

              if (numberMatches && numberMatches.length >= 50) {
                this.logger.log(
                  `Found ${numberMatches.length} numbers in Llama response`,
                );

                // Take the first 100 numbers and convert to floats
                let embedding = numberMatches.slice(0, 100).map((num) => {
                  const parsed = parseFloat(num);
                  return isNaN(parsed) ? 0 : parsed;
                });

                // Normalize values to [-1, 1] range
                embedding = embedding.map((val) =>
                  Math.max(-1, Math.min(1, val)),
                );

                // Expand to 1536 dimensions using pattern repetition and slight variations
                const fullEmbedding = [];
                for (let i = 0; i < 1536; i++) {
                  const baseIndex = i % embedding.length;
                  const variation = (i / embedding.length) * 0.1; // Small variation
                  fullEmbedding.push(embedding[baseIndex] + variation);
                }

                // Normalize the final vector
                const magnitude = Math.sqrt(
                  fullEmbedding.reduce((sum, val) => sum + val * val, 0),
                );
                if (magnitude > 0) {
                  const normalizedEmbedding = fullEmbedding.map(
                    (val) => val / magnitude,
                  );
                  this.logger.log(
                    'Successfully generated embedding using Llama model',
                  );
                  return normalizedEmbedding;
                }
              } else {
                this.logger.warn(
                  `Not enough numbers found in Llama response. Found: ${numberMatches ? numberMatches.length : 0}`,
                );
              }
            } catch (parseError) {
              this.logger.warn(
                `Failed to parse Llama embedding response: ${parseError.message}`,
              );
            }
          } else {
            this.logger.warn('No response received from Llama model');
          }
        } catch (llamaError) {
          this.logger.warn(
            `Llama model embedding error: ${llamaError.message}`,
          );
          if (llamaError.response) {
            this.logger.warn(
              `Llama error response: ${JSON.stringify(llamaError.response.data)}`,
            );
          }
        }

        // Final fallback to simple hash-based embedding
        this.logger.log('Using fallback simple embedding');
        return this.generateSimpleEmbedding(text);
      } catch (error) {
        this.logger.error(
          `Error generating embedding (attempt ${attempt}/${retries}):`,
          error.message,
        );

        if (attempt < retries) {
          const delay = Math.pow(2, attempt) * 1000;
          await new Promise((resolve) => setTimeout(resolve, delay));
          continue;
        }

        throw new Error(`Failed to generate embedding: ${error.message}`);
      }
    }
  }

  async generateEmbeddings(texts: string[]): Promise<number[][]> {
    const embeddings: number[][] = [];
    const batchSize = 5;

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const batchPromises = batch.map((text) => this.generateEmbedding(text));

      try {
        const batchEmbeddings = await Promise.all(batchPromises);
        embeddings.push(...batchEmbeddings);

        // Add delay between batches
        if (i + batchSize < texts.length) {
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
      } catch (error) {
        this.logger.error('Error generating batch embeddings:', error.message);
        throw error;
      }
    }

    return embeddings;
  }

  private generateSimpleEmbedding(text: string): number[] {
    // Simple hash-based embedding for development/testing
    // This creates a consistent 1536-dimensional vector based on text content
    const hash = this.simpleHash(text);
    const embedding = new Array(1536).fill(0);

    // Use hash to seed pseudo-random values
    let seed = hash;
    for (let i = 0; i < 1536; i++) {
      seed = (seed * 9301 + 49297) % 233280;
      embedding[i] = (seed / 233280) * 2 - 1; // Normalize to [-1, 1]
    }

    // Normalize the vector
    const magnitude = Math.sqrt(
      embedding.reduce((sum, val) => sum + val * val, 0),
    );
    return embedding.map((val) => val / magnitude);
  }

  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }
}
