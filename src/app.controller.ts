import { Controller, Get, Post, Body, Query } from '@nestjs/common';
import { AppService } from './app.service';
import { AiQueryService } from './ai-query.service';
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

  @Get('health')
  async getHealth() {
    const llamaConnection = await this.aiQueryService.testConnection();
    return {
      status: 'ok',
      llamaConnected: llamaConnection,
      timestamp: new Date().toISOString(),
    };
  }

  @Post('query')
  async query(@Body('question') question: string) {
    if (!question) {
      return { error: 'Question is required' };
    }

    try {
      const answer = await this.aiQueryService.queryWithContext(question);
      return {
        question,
        answer,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      return {
        error: error.message,
        question,
        timestamp: new Date().toISOString(),
      };
    }
  }

  @Post('reprocess')
  async reprocessText() {
    try {
      await this.textContextService.reprocessTextFile();
      return {
        message: 'Text file reprocessed successfully',
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      return {
        error: error.message,
        timestamp: new Date().toISOString(),
      };
    }
  }

  @Get('test-llama')
  async testLlama(@Query('message') message: string = 'Hello, how are you?') {
    const isConnected = await this.aiQueryService.testConnection();
    return {
      connected: isConnected,
      testMessage: message,
      timestamp: new Date().toISOString(),
    };
  }
}
