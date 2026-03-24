import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector"
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers"

export class VectorStore {

    private store: Neo4jVectorStore
    private nodeLabel: string

    constructor(
        nodeLabel: string,
        store: Neo4jVectorStore
    ) {
        this.nodeLabel = nodeLabel
        this.store = store
    }

    static async create(
        nodeLabel: string,
        model: string,
        pretrainedOptions: PretrainedOptions,
        neo4jConfig: any,
    ) {
        const embeddings = new HuggingFaceTransformersEmbeddings({
            model,
            pretrainedOptions
        })
        
        const store = await Neo4jVectorStore.fromExistingGraph(embeddings, neo4jConfig);

        return new VectorStore(nodeLabel, store)
    }

    async close() {
        this.store.close();
    }


    async clearDB() {
        console.log("\n...Cleaning existing docs...");

        await this.store.query(
            `MATCH (n:\`${this.nodeLabel}\`) DETACH DELETE n`
        )

        console.log("\n...Docs successfully removed! ✅...");
    }

    async storeChunksInNeo4j(chunks: Document[]) {
        console.log(`\n...storing in Neo4j...`);

        for (const [i, chunk] of chunks.entries()) {
            await this.store.addDocuments([chunk])
        }

        console.log(`\n...Stored ${chunks.length} chunks to Neo4j! ✅...`);
    }

    async similaritySearch(question: string, topK: number): Promise<string> {

        return this.store.similaritySearchWithScore(question, topK)
            .then(documentsWithScore => documentsWithScore
                .map(documentWithScore => ({
                    document: documentWithScore[0],
                    score: documentWithScore[1]
                }))
                .filter(documentWithScore => documentWithScore.score > 0.5)
                .map(documentWithScore => documentWithScore.document.pageContent)
                .join("\n\n")
            );

    }



}