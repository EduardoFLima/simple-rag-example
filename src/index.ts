import { PDFProcessor } from "./PDFProcessor.ts"
import { Document } from "@langchain/core/documents";
import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";
import { CONFIG } from "./config.ts";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";

const questions = [
    'What is a tensor ?',
    // 'How tensorflow stores the data ?',
    // 'What can man do with tensorflow.js?',
    // 'What is the difference between regular tensorflow and tensorflow.js?'
]

const clearDB = async (vectorStore: Neo4jVectorStore, nodeLabel: string) => {
    console.log("\n...Cleaning existing docs...");

    await vectorStore.query(
        `MATCH (n:\`${nodeLabel}\`) DETACH DELETE n`
    )

    console.log("\n...Docs successfully removed! ✅...");
}

const storeChunksInNeo4j = async (vectorStore: Neo4jVectorStore, chunks: Document[]) => {
    console.log(`\n...storing in Neo4j...`);

    for (const [i, chunk] of chunks.entries()) {
        await vectorStore.addDocuments([chunk])
    }

    console.log(`\n...Stored ${chunks.length} chunks to Neo4j! ✅...`);
}

const prepareRAG = async (vectorStore: Neo4jVectorStore) => {
    console.log('\n\n...preparing RAG...');

    await clearDB(vectorStore, CONFIG.neo4j.nodeLabel);

    const chunks = await new PDFProcessor(CONFIG.pdf.path, CONFIG.textSplitter).splitPdf();

    await storeChunksInNeo4j(vectorStore, chunks);

    console.log('\n...RAG prepared ! ✅...\n');
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

const run = async () => {
    const embeddings = new HuggingFaceTransformersEmbeddings({
        model: CONFIG.embedding.modelName,
        pretrainedOptions: CONFIG.embedding.pretrainedOptions as PretrainedOptions
    })

    let neo4jVectorStore = null;

    try {
        neo4jVectorStore = await Neo4jVectorStore.fromExistingGraph(
            embeddings,
            CONFIG.neo4j
        )
        await prepareRAG(neo4jVectorStore);
    }
    finally {
        neo4jVectorStore?.close();
    }

    // await askQuestions(questions);
}

await run();
