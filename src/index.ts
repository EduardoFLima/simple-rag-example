import { Document } from "@langchain/core/documents";
import { splitPdf } from "./fileLoader.ts";
import neo4j from "neo4j-driver";

const driver = neo4j.driver(
    'neo4j://localhost',
    neo4j.auth.basic('neo4j', 'password')
)

const prepareRAG = async () => {
    const chunks = await splitPdf();
    // console.log(`\n${JSON.stringify(chunks[0], null, 2)}\n`);
    await storeChunksInNeo4j(chunks);
}

const storeChunksInNeo4j = async (chunks: Document[]) => {
    const session = driver.session();

    console.log(`\n...storing in Neo4j...\n`);
    try {

        for (const [i, chunk] of chunks.entries()) {
            await session.run(
                `CREATE (c:Chunk {index: $index, content: $content, source: $source, page: $page})`,
                {
                    index: neo4j.int(i),
                    content: chunk.pageContent,
                    source: chunk?.metadata?.source ?? '',
                    page: neo4j.int(chunk?.metadata?.loc?.pageNumber)
                }
            );

            console.log(`Saved ${chunks.length} chunks to Neo4j`);
        }

    } finally {
        await session.close();
        await driver.close();
    }
}

export const run = async () => {
    console.log('...starting...\n');

    await prepareRAG();



    console.log('\n...ending...\n');
};

await run();
