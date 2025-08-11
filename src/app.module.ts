import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { MulterModule } from '@nestjs/platform-express';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { MilvusService } from './milvus.service';
import { EmbeddingsService } from './embeddings.service';
import { TextContextService } from './text-context.service';
import { AiQueryService } from './ai-query.service';
import { PdfController } from './pdf.controller';
import { PdfService } from './pdf.service';
import * as multer from 'multer';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: '.env',
    }),
    MulterModule.register({
      storage: multer.memoryStorage(), // Store files in memory for processing
      limits: {
        fileSize: 10 * 1024 * 1024, // 10MB limit
      },
    }),
  ],
  controllers: [AppController, PdfController],
  providers: [
    AppService,
    MilvusService,
    EmbeddingsService,
    TextContextService,
    AiQueryService,
    PdfService,
  ],
})
export class AppModule {}
