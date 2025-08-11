import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { MilvusClient, DataType } from '@zilliz/milvus2-sdk-node';

export interface ChunkData {
  id: string;
  text: string;
  embedding: number[];
}

@Injectable()
export class MilvusService implements OnModuleInit {
  private readonly logger = new Logger(MilvusService.name);
  private client: MilvusClient;
  private collectionName: string;

  constructor(private configService: ConfigService) {
    this.collectionName = this.configService.get<string>('COLLECTION_NAME', 'context_chunks');
  }

  async onModuleInit() {
    await this.connect();
    await this.createCollection();
  }

  private async connect() {
    const host = this.configService.get<string>('MILVUS_HOST', 'localhost');
    const port = this.configService.get<string>('MILVUS_PORT', '19530');

    this.client = new MilvusClient({
      address: `${host}:${port}`,
    });

    this.logger.log(`Connected to Milvus at ${host}:${port}`);
  }

  private async createCollection() {
    const dimension = this.configService.get<number>('VECTOR_DIMENSION', 1536);

    // Check if collection exists
    const collections = await this.client.listCollections();
    const collectionExists = collections.data.some(
      (collection) => collection.name === this.collectionName,
    );

    if (!collectionExists) {
      // Create collection
      await this.client.createCollection({
        collection_name: this.collectionName,
        fields: [
          {
            name: 'id',
            data_type: DataType.VarChar,
            max_length: 100,
            is_primary_key: true,
          },
          {
            name: 'text',
            data_type: DataType.VarChar,
            max_length: 65535,
          },
          {
            name: 'embedding',
            data_type: DataType.FloatVector,
            dimension: dimension,
          },
        ],
      });

      // Create index for vector field
      await this.client.createIndex({
        collection_name: this.collectionName,
        field_name: 'embedding',
        index_type: 'IVF_FLAT',
        metric_type: 'IP',
        params: { nlist: 1024 },
      });

      this.logger.log(`Collection '${this.collectionName}' created successfully`);
    } else {
      this.logger.log(`Collection '${this.collectionName}' already exists`);
    }

    // Load collection
    await this.client.loadCollection({
      collection_name: this.collectionName,
    });
  }

  async insertChunks(chunks: ChunkData[]): Promise<void> {
    const data = chunks.map((chunk) => ({
      id: chunk.id,
      text: chunk.text,
      embedding: chunk.embedding,
    }));

    await this.client.insert({
      collection_name: this.collectionName,
      data: data,
    });

    this.logger.log(`Inserted ${chunks.length} chunks into Milvus`);
  }

  async searchSimilar(queryEmbedding: number[], topK: number = 5): Promise<any[]> {
    const searchParams = {
      collection_name: this.collectionName,
      vector: queryEmbedding,
      filter: '',
      params: { nprobe: 10 },
      limit: topK,
      offset: 0,
      metric_type: 'IP',
      output_fields: ['id', 'text'],
    };

    const result = await this.client.search(searchParams);
    return result.results;
  }

  async deleteCollection(): Promise<void> {
    await this.client.dropCollection({
      collection_name: this.collectionName,
    });
    this.logger.log(`Collection '${this.collectionName}' deleted`);
  }
}
