import { PDFProcessor } from "./PDFProcessor.ts"
import { CONFIG } from "./config.ts";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { VectorStore } from "./VectorStore.ts";
import { StringOutputParser } from "@langchain/core/output_parsers";

let neo4jVectorStore: VectorStore;

const questions = [
    "Como converter objetos JavaScript em tensores?",
    "O que é normalização de dados e por que é necessária?",
    "Como funciona uma rede neural no TensorFlow.js?",
    "O que significa treinar uma rede neural?",
    "o que é hot enconding e quando usar?"
]

const prepareRAG = async () => {
    console.log('\n\n=== preparing RAG ===');

    await neo4jVectorStore.clearDB();

    const chunks = await new PDFProcessor(CONFIG.pdf.path, CONFIG.textSplitter).splitPdf();

    await neo4jVectorStore.storeChunksInNeo4j(chunks);

    console.log('\n===RAG prepared ! ✅ ===\n\n');
}

export const askQuestions = async (questions: string[]) => {
    console.log('...starting to ask questions...\n');

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
        .pipe(new StringOutputParser())

    for (const question of questions) {

        console.log(`\n=> Question: ${question}`)

        // enrich with RAG
        const context: string = await neo4jVectorStore.similaritySearch(question, CONFIG.similarity.topK)

        // handover question to LLM
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
            context
        })

        console.log(`\n== response: ${response}`)
    }

    console.log('\n...questions answered; closing application...\n');
};

const run = async () => {

    try {
        neo4jVectorStore = await VectorStore.create(
            CONFIG.neo4j.nodeLabel,
            CONFIG.embedding.modelName,
            CONFIG.embedding.pretrainedOptions as PretrainedOptions,
            CONFIG.neo4j
        );

        await prepareRAG();

        await askQuestions(questions);
    }
    catch (error) {
        console.log(`\n\n !!! some error happened: !!!\n\n${error}`)
    }
    finally {
        neo4jVectorStore.close();
    }

}

await run();
