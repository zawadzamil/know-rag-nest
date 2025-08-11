import {
    Controller,
    Post,
    UploadedFile,
    UseInterceptors,
    Body,
    Get,
    Param,
    Delete,
    BadRequestException,
    NotFoundException,
    Logger,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { PdfService, PdfDocument } from './pdf.service';
import { AiQueryService } from './ai-query.service';

@Controller('pdf')
export class PdfController {
    private readonly logger = new Logger(PdfController.name);

    constructor(
        private readonly pdfService: PdfService,
        private readonly aiQueryService: AiQueryService,
    ) {}

    @Post('upload')
    @UseInterceptors(FileInterceptor('file'))
    async uploadPdf(@UploadedFile() file: Express.Multer.File): Promise<{
        message: string;
        document: PdfDocument;
    }> {
        if (!file) {
            throw new BadRequestException('No file uploaded');
        }

        if (file.mimetype !== 'application/pdf') {
            throw new BadRequestException('Only PDF files are allowed');
        }

        this.logger.log(`Received PDF upload: ${file.originalname} (${file.size} bytes)`);

        try {
            const document = await this.pdfService.processPdf(file.buffer, file.originalname);

            return {
                message: 'PDF uploaded and processed successfully',
                document: {
                    id: document.id,
                    filename: document.filename,
                    uploadDate: document.uploadDate,
                    textContent: document.textContent.substring(0, 200) + '...', // Truncate for response
                    chunkCount: document.chunkCount,
                },
            };
        } catch (error) {
            this.logger.error('Error processing PDF:', error.message);
            throw new BadRequestException(`Failed to process PDF: ${error.message}`);
        }
    }

    @Post('query')
    async queryDocuments(@Body() queryDto: {
        question: string;
        pdfOnly?: boolean;
      }): Promise<{
        question: string;
        answer: string;
        timestamp: string;
        sources: string[];
      }> {
        if (!queryDto.question || queryDto.question.trim().length === 0) {
          throw new BadRequestException('Question is required');
        }

        this.logger.log(`Received query: ${queryDto.question}`);

        try {
          // Use PDF-specific query if requested
          const answer = queryDto.pdfOnly
            ? await this.queryPdfContentOnly(queryDto.question)
            : await this.aiQueryService.queryWithContext(queryDto.question);

          // Get document sources for reference
          const sources = this.pdfService.getAllDocuments().map(doc => doc.filename);

          return {
            question: queryDto.question,
            answer,
            timestamp: new Date().toISOString(),
            sources,
          };
        } catch (error) {
          this.logger.error('Error processing query:', error.message);
          throw new BadRequestException(`Failed to process query: ${error.message}`);
        }
      }

      private async queryPdfContentOnly(question: string): Promise<string> {
        const documents = this.pdfService.getAllDocuments();
        if (documents.length === 0) {
          return 'No PDF documents have been uploaded yet.';
        }

        // Create context from all PDF documents
        const pdfContext = documents
          .map(doc => `Document: ${doc.filename}\n${doc.textContent}`)
          .join('\n\n---\n\n');

        // Use a simple direct query without vector search for PDF-only content
        const prompt = `Answer the question using only the provided PDF documents. Be concise and direct.
  
        PDF Documents:
        ${pdfContext}
  
        Question: ${question}
  
        Answer:`;

        try {
          const response = await fetch(`http://192.168.68.121:11434/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model: 'llama3.1:latest',
              prompt,
              stream: false,
              options: {
                temperature: 0.1,
                num_predict: 200,
              },
            }),
          });

          const data = await response.json();
          return data.response?.trim() || 'Unable to generate response';
        } catch (error) {
          this.logger.error('Error querying PDF content:', error.message);
          throw new Error('Failed to query PDF content');
        }
      }

    @Get('documents')
    getAllDocuments(): {
        documents: PdfDocument[];
        stats: { totalDocuments: number; totalChunks: number };
    } {
        const documents = this.pdfService.getAllDocuments();
        const stats = this.pdfService.getDocumentStats();

        return {
            documents: documents.map((doc) => ({
                ...doc,
                textContent: doc.textContent.substring(0, 200) + '...', // Truncate for listing
            })),
            stats,
        };
    }

    @Get('documents/:id')
    getDocument(@Param('id') id: string): PdfDocument {
        const document = this.pdfService.getDocument(id);
        if (!document) {
            throw new NotFoundException(`Document with ID ${id} not found`);
        }
        return document;
    }

    @Delete('documents/:id')
    async deleteDocument(@Param('id') id: string): Promise<{
        message: string;
        deleted: boolean;
    }> {
        const deleted = await this.pdfService.deleteDocument(id);
        if (!deleted) {
            throw new NotFoundException(`Document with ID ${id} not found`);
        }

        return {
            message: 'Document deleted successfully',
            deleted: true,
        };
    }

    @Get('stats')
    getStats() {
        return this.pdfService.getDocumentStats();
    }
}
