import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { CONFIG } from "./config.ts";
import { Document } from "@langchain/core/documents";


export const splitPdf = async (): Promise<Document[]> => {

    const pdfLoader = new PDFLoader(CONFIG.pdf.path);

    const docs = await pdfLoader.load();

    const textSplitter = new RecursiveCharacterTextSplitter(CONFIG.textSplitter);

    return await textSplitter.splitDocuments(docs);
};
