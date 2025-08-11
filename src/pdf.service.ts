import { Injectable, Logger } from '@nestjs/common';
import { MilvusService, ChunkData } from './milvus.service';
import { EmbeddingsService } from './embeddings.service';
import * as pdfParse from 'pdf-parse';
import { randomBytes } from 'crypto';

export interface PdfDocument {
  id: string;
  filename: string;
  uploadDate: Date;
  textContent: string;
  chunkCount: number;
}

@Injectable()
export class PdfService {
  private readonly logger = new Logger(PdfService.name);
  private readonly documents: Map<string, PdfDocument> = new Map();

  constructor(
    private milvusService: MilvusService,
    private embeddingsService: EmbeddingsService,
  ) {}

  async processPdf(buffer: Buffer, filename: string): Promise<PdfDocument> {
    try {
      this.logger.log(`Processing PDF: ${filename}`);

      // Parse PDF content
      const pdfData = await pdfParse(buffer);
      const textContent = pdfData.text;

      if (!textContent || textContent.trim().length === 0) {
        throw new Error('PDF contains no readable text');
      }

      this.logger.log(`Extracted ${textContent.length} characters from PDF`);

      // Create document record
      const documentId = this.generateDocumentId();
      const document: PdfDocument = {
        id: documentId,
        filename,
        uploadDate: new Date(),
        textContent,
        chunkCount: 0,
      };

      // Chunk the text content
      const chunks = this.chunkText(textContent, documentId, filename);
      document.chunkCount = chunks.length;

      this.logger.log(`Created ${chunks.length} chunks from PDF`);

      // Process and store chunks
      await this.processAndStoreChunks(chunks);

      // Store document metadata
      this.documents.set(documentId, document);

      this.logger.log(
        `Successfully processed PDF: ${filename} (ID: ${documentId})`,
      );
      return document;
    } catch (error) {
      this.logger.error(`Error processing PDF ${filename}:`, error.message);
      throw new Error(`Failed to process PDF: ${error.message}`);
    }
  }

  private chunkText(
    text: string,
    documentId: string,
    filename: string,
    chunkSize: number = 500,
  ): ChunkData[] {
    const sentences = text.split(/[.!?]+/).filter((s) => s.trim().length > 0);
    const chunks: ChunkData[] = [];
    let currentChunk = '';
    let chunkIndex = 0;

    for (const sentence of sentences) {
      const trimmedSentence = sentence.trim();
      if (
        currentChunk.length + trimmedSentence.length > chunkSize &&
        currentChunk.length > 0
      ) {
        // Save current chunk
        chunks.push({
          id: `${documentId}_chunk_${chunkIndex}`,
          text: currentChunk.trim(),
          embedding: [], // Will be filled later
        });

        currentChunk = trimmedSentence;
        chunkIndex++;
      } else {
        currentChunk += (currentChunk.length > 0 ? '. ' : '') + trimmedSentence;
      }
    }

    // Add the last chunk if it's not empty
    if (currentChunk.trim().length > 0) {
      chunks.push({
        id: `${documentId}_chunk_${chunkIndex}`,
        text: currentChunk.trim(),
        embedding: [],
      });
    }

    // If no chunks were created, create one with the full text
    if (chunks.length === 0) {
      chunks.push({
        id: `${documentId}_chunk_0`,
        text: text.trim(),
        embedding: [],
      });
    }

    this.logger.log(`Created ${chunks.length} chunks from ${filename}`);
    return chunks;
  }

  private async processAndStoreChunks(chunks: ChunkData[]): Promise<void> {
    this.logger.log(`Processing ${chunks.length} chunks...`);

    // Generate embeddings for all chunks
    const texts = chunks.map((chunk) => chunk.text);
    const embeddings = await this.embeddingsService.generateEmbeddings(texts);

    // Assign embeddings to chunks
    for (let i = 0; i < chunks.length; i++) {
      chunks[i].embedding = embeddings[i];
    }

    // Store in Milvus
    await this.milvusService.insertChunks(chunks);

    this.logger.log('All chunks processed and stored in Milvus');
  }

  private generateDocumentId(): string {
    return `doc_${randomBytes(8).toString('hex')}_${Date.now()}`;
  }

  getAllDocuments(): PdfDocument[] {
    return Array.from(this.documents.values());
  }

  getDocument(id: string): PdfDocument | undefined {
    return this.documents.get(id);
  }

  async deleteDocument(id: string): Promise<boolean> {
    const document = this.documents.get(id);
    if (!document) {
      return false;
    }

    try {
      // Note: In a real implementation, you might want to delete specific chunks from Milvus
      // For now, we'll just remove from our in-memory storage
      this.documents.delete(id);
      this.logger.log(`Deleted document: ${document.filename} (ID: ${id})`);
      return true;
    } catch (error) {
      this.logger.error(`Error deleting document ${id}:`, error.message);
      return false;
    }
  }

  getDocumentStats(): { totalDocuments: number; totalChunks: number } {
    const documents = Array.from(this.documents.values());
    return {
      totalDocuments: documents.length,
      totalChunks: documents.reduce((sum, doc) => sum + doc.chunkCount, 0),
    };
  }
}
