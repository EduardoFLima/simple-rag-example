import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { type TextSplitterConfig } from "./config.ts";

export class PDFProcessor {

    private pdfPath: string
    private textSplitterConfig: TextSplitterConfig

    constructor(pdfPath: string, textSplitterConfig: TextSplitterConfig) {
        this.pdfPath = pdfPath
        this.textSplitterConfig = textSplitterConfig
    }

    async splitPdf() {

        const pdfLoader = new PDFLoader(this.pdfPath);

        const docs = await pdfLoader.load();

        const textSplitter = new RecursiveCharacterTextSplitter(this.textSplitterConfig);

        const documents = await textSplitter.splitDocuments(docs);

        return documents.map(doc => ({
            ...doc,
            metadata: {
                source: doc.metadata.source,
            }
        }))
    };
}

