import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios from 'axios';

@Injectable()
export class EmbeddingsService {
  private readonly logger = new Logger(EmbeddingsService.name);
  private readonly localLlmHost: string;
  private readonly localLlmPort: string;
  private readonly embeddingModel: string;
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
    // Use a proper embedding model instead of text generation model
    this.embeddingModel = this.configService.get<string>(
      'EMBEDDING_MODEL',
      'nomic-embed-text:latest',
    );
    this.baseUrl = `http://${this.localLlmHost}:${this.localLlmPort}`;
  }

  async generateEmbedding(
    text: string,
    retries: number = 3,
  ): Promise<number[]> {
    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        // Try using dedicated embedding model first
        try {
          this.logger.log(
            `Generating embedding using embedding model (attempt ${attempt}) for text: "${text.substring(0, 50)}..."`,
          );

          const response = await axios.post(
            `${this.baseUrl}/api/embeddings`,
            {
              model: this.embeddingModel,
              prompt: text,
            },
            {
              timeout: 30000,
              headers: {
                'Content-Type': 'application/json',
              },
            },
          );

          if (response.data?.embedding) {
            this.logger.log('Successfully generated embedding using dedicated model');
            let embedding = response.data.embedding;

            // Ensure we have exactly 384 dimensions
            if (embedding.length > 384) {
              embedding = embedding.slice(0, 384);
            } else if (embedding.length < 384) {
              // Pad with zeros if needed
              while (embedding.length < 384) {
                embedding.push(0);
              }
            }

            return embedding;
          }
        } catch (embeddingError) {
          this.logger.warn(
            `Dedicated embedding model not available: ${embeddingError.message}`,
          );
        }

        // Fallback: Use simple hash-based embedding
        this.logger.log('Using fallback simple embedding for consistent results');
        return this.generateSimpleEmbedding(text);
      } catch (error) {
        this.logger.error(
          `Error generating embedding (attempt ${attempt}/${retries}):`,
          error.message,
        );

        if (attempt < retries) {
          const delay = Math.pow(2, attempt) * 1000;
          this.logger.log(`Retrying in ${delay}ms...`);
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
    // This creates a consistent 384-dimensional vector based on text content
    const hash = this.simpleHash(text);
    const embedding = new Array(384).fill(0);

    // Use hash to seed pseudo-random values
    let seed = hash;
    for (let i = 0; i < 384; i++) {
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
