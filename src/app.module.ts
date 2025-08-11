import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { MilvusService } from './milvus.service';
import { EmbeddingsService } from './embeddings.service';
import { TextContextService } from './text-context.service';
import { AiQueryService } from './ai-query.service';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: '.env',
    }),
  ],
  controllers: [AppController],
  providers: [
    AppService,
    MilvusService,
    EmbeddingsService,
    TextContextService,
    AiQueryService,
  ],
})
export class AppModule {}
