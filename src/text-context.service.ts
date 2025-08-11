import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { MilvusService, ChunkData } from './milvus.service';
import { EmbeddingsService } from './embeddings.service';
import { nanoid } from 'nanoid';
import * as fs from 'fs';
import * as path from 'path';

@Injectable()
export class TextContextService implements OnModuleInit {
  private readonly logger = new Logger(TextContextService.name);

  constructor(
    private milvusService: MilvusService,
    private embeddingsService: EmbeddingsService,
  ) {}

  async onModuleInit() {
    await this.loadAndProcessTextFile();
  }

  private async loadAndProcessTextFile(): Promise<void> {
    try {
      const filePath = path.join(process.cwd(), 'about_me.txt');
      const content = fs.readFileSync(filePath, 'utf-8');

      this.logger.log('Text file loaded successfully');

      const chunks = this.chunkText(content);
      await this.processAndStoreChunks(chunks);
    } catch (error) {
      this.logger.error('Error loading text file:', error.message);
    }
  }

  private chunkText(text: string, chunkSize: number = 500, overlap: number = 50): string[] {
    const sentences = text.split(/[.!?]+/).filter(sentence => sentence.trim().length > 0);
    const chunks: string[] = [];
    let currentChunk = '';

    for (const sentence of sentences) {
      const trimmedSentence = sentence.trim();
      if (currentChunk.length + trimmedSentence.length <= chunkSize) {
        currentChunk += (currentChunk ? '. ' : '') + trimmedSentence;
      } else {
        if (currentChunk) {
          chunks.push(currentChunk + '.');
        }

        // Handle overlap by including the last part of the previous chunk
        if (overlap > 0 && chunks.length > 0) {
          const lastChunk = chunks[chunks.length - 1];
          const overlapText = lastChunk.slice(-overlap);
          currentChunk = overlapText + ' ' + trimmedSentence;
        } else {
          currentChunk = trimmedSentence;
        }
      }
    }

    if (currentChunk) {
      chunks.push(currentChunk + '.');
    }

    return chunks;
  }

  private async processAndStoreChunks(chunks: string[]): Promise<void> {
    this.logger.log(`Processing ${chunks.length} chunks...`);

    const embeddings = await this.embeddingsService.generateEmbeddings(chunks);

    const chunkData: ChunkData[] = chunks.map((chunk, index) => ({
      id: nanoid(),
      text: chunk,
      embedding: embeddings[index],
    }));

    await this.milvusService.insertChunks(chunkData);
    this.logger.log('All chunks processed and stored in Milvus');
  }

  async reprocessTextFile(): Promise<void> {
    this.logger.log('Reprocessing text file...');
    await this.loadAndProcessTextFile();
  }
}
