import { Document } from "@langchain/core/documents";
import { splitPdf } from "./fileLoader.ts";
import neo4j from "neo4j-driver";
import { CONFIG } from "./config.ts";
import { pipeline } from "@huggingface/transformers";
import { ChatOpenAI } from "@langchain/openai";
import { readFile } from "node:fs/promises"
import { ChatPromptTemplate } from "@langchain/core/prompts";

const embedder = await pipeline(
    "feature-extraction",
    CONFIG.embedding.modelName,
    CONFIG.embedding.pretrainedOptions,
);

const driver = neo4j.driver(
    CONFIG.neo4j.url,
    neo4j.auth.basic(
        CONFIG.neo4j.username,
        CONFIG.neo4j.password
    )
)

const prepareRAG = async () => {
    console.log('\n...preparing RAG...\n');

    const chunks = await splitPdf();
    // console.log(`\n${JSON.stringify(chunks[0], null, 2)}\n`);
    await storeChunksInNeo4j(chunks);

    console.log('\n...RAG prepared !...\n');
}

const storeChunksInNeo4j = async (chunks: Document[]) => {
    const session = driver.session();

    console.log(`\n...storing in Neo4j...\n`);
    try {
        for (const [i, chunk] of chunks.entries()) {

            const output = await embedder(chunk.pageContent, { pooling: "mean", normalize: true });

            const embedding = Array.from(output.data)

            await session.run(
                `CREATE (c:${CONFIG.neo4j.nodeLabel} {
                    index: $index, 
                    content: $content, 
                    source: $source, 
                    page: $page,
                    embedding: $embedding
                })`,
                {
                    index: neo4j.int(i),
                    content: chunk.pageContent,
                    source: chunk?.metadata?.source ?? '',
                    page: neo4j.int(chunk?.metadata?.loc?.pageNumber),
                    embedding,
                }
            );

            console.log(`Saved ${chunks.length} chunks to Neo4j`);
        }

    } finally {
        await session.close();
        await driver.close();
    }
}

export const askQuestions = async (questions: string[]) => {
    console.log('...starting to ask questions...\n');

    for (const question of questions) {

        console.log(`\n=> Question: ${question}\n`)

        // enrich with RAG

        // handover question to LLM
        const nplModel = new ChatOpenAI({
            modelName: CONFIG.openRouter.nlpModel,
            temperature: CONFIG.openRouter.temperature,
            maxRetries: CONFIG.openRouter.maxRetries,
            apiKey: CONFIG.openRouter.apiKey,
            configuration: {
                baseURL: CONFIG.openRouter.url,
                defaultHeaders: CONFIG.openRouter.defaultHeaders
            }
        });

        const { promptConfig } = CONFIG;

        const responseChain = ChatPromptTemplate.fromTemplate(CONFIG.templateText)
            .pipe(nplModel)

        const response = await responseChain.invoke({
            role: promptConfig.role,
            task: promptConfig.task,
            tone: promptConfig.tone,
            language: promptConfig.constraints.language,
            format: promptConfig.constraints.format,
            instructions: promptConfig.instructions.map((instruction: string, idx: number) =>
                `${idx + 1}. ${instruction}`
            ).join('\n'),
            question,
            context: ``
        })

        console.log(`== response: ${response.content}`)

    }

    console.log('\n...questions answered; closing application...\n');
};

await prepareRAG();


const questions = [
    'What is a tensor ?',
    // 'How tensorflow stores the data ?',
    // 'What can man do with tensorflow.js?',
    // 'What is the difference between regular tensorflow and tensorflow.js?'
]

await askQuestions(questions);
