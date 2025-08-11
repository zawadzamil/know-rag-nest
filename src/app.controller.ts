import { Controller, Get, Post, Body, Query } from '@nestjs/common';
import { AppService } from './app.service';
import { AiQueryService, QueryResponse } from './ai-query.service';
import { TextContextService } from './text-context.service';

export class QueryDto {
  question: string;
}

@Controller()
export class AppController {
  constructor(
    private readonly appService: AppService,
    private readonly aiQueryService: AiQueryService,
    private readonly textContextService: TextContextService,
  ) {}

  @Get()
  getHello(): string {
    return this.appService.getHello();
  }

  @Post('query')
  async askQuestion(@Body() queryDto: QueryDto): Promise<QueryResponse> {
    return await this.aiQueryService.processQuery(queryDto.question);
  }

  @Get('query')
  async askQuestionGet(@Query('q') question: string): Promise<QueryResponse> {
    if (!question) {
      throw new Error('Question parameter is required');
    }
    return await this.aiQueryService.processQuery(question);
  }

  @Post('reprocess')
  async reprocessContext(): Promise<{ message: string }> {
    await this.textContextService.reprocessTextFile();
    return {
      message: 'Text file reprocessed and embeddings updated successfully',
    };
  }

  @Get('health')
  getHealth(): { status: string; timestamp: string } {
    return {
      status: 'ok',
      timestamp: new Date().toISOString(),
    };
  }
}
