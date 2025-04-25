#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ErrorCode, ListToolsRequestSchema, McpError, } from '@modelcontextprotocol/sdk/types.js';
import stringSimilarity from 'string-similarity';
import natural from 'natural';
import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from '@google/generative-ai';
// Function to normalize text: lowercase, remove punctuation, normalize whitespace
function normalizeText(text) {
    let str = String(text).toLowerCase();
    // Keep letters, numbers, and CJK characters, remove other punctuation
    str = str.replace(/[^\p{L}\p{N}\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\s]/gu, '');
    str = str.replace(/\s+/g, ' ').trim(); // Normalize whitespace
    return str;
}
// Helper to safely get text from Gemini response
function getGeminiResponseText(response) {
    try {
        // Check for different possible response structures
        if (response?.response?.candidates?.[0]?.content?.parts?.[0]?.text) {
            return response.response.candidates[0].content.parts[0].text.trim();
        }
        if (response?.text) { // Simpler structure if generateContent returns directly
            return typeof response.text === 'function' ? response.text().trim() : String(response.text).trim();
        }
        // Add more checks if needed based on actual Gemini SDK behavior
    }
    catch (e) {
        console.error("Error parsing Gemini response:", e);
    }
    return null; // Return null if text cannot be extracted
}
class EntityResolutionServer {
    constructor() {
        this.server = new Server({
            name: 'entity-resolution-server',
            version: '0.2.1', // Incremented version for fix
        }, {
            capabilities: {
                tools: {},
            },
        });
        this.setupToolHandlers();
        this.server.onerror = (error) => console.error('[MCP Error]', error);
        process.on('SIGINT', async () => {
            await this.server.close();
            process.exit(0);
        });
    }
    // Calculates syntactic similarity scores and prepares data for LLM
    calculateSyntacticSimilarities(entity1, entity2) {
        let totalWeight = 0;
        let weightedDiceSimilarity = 0;
        let weightedLevenshteinSimilarity = 0;
        const fieldDetails = {};
        const keys1 = Object.keys(entity1);
        const keys2 = Object.keys(entity2);
        const commonKeys = keys1.filter(key => keys2.includes(key));
        if (commonKeys.length === 0) {
            return { dice: 0, levenshtein: 0, fieldDetails: {} };
        }
        for (const key of commonKeys) {
            const value1 = entity1[key];
            const value2 = entity2[key];
            // Normalize values before comparison
            const normalized1 = normalizeText(value1);
            const normalized2 = normalizeText(value2);
            // Calculate Dice similarity
            const diceSim = stringSimilarity.compareTwoStrings(normalized1, normalized2);
            // Calculate Levenshtein similarity (normalized to 0-1 range)
            const levenshteinDist = natural.LevenshteinDistance(normalized1, normalized2, { insertion_cost: 1, deletion_cost: 1, substitution_cost: 1 });
            const maxLength = Math.max(normalized1.length, normalized2.length);
            const levenshteinSim = maxLength === 0 ? 1 : 1 - (levenshteinDist / maxLength);
            // Apply field weights (currently uniform)
            const weight = 1.0; // TODO: Make weights configurable
            totalWeight += weight;
            weightedDiceSimilarity += diceSim * weight;
            weightedLevenshteinSimilarity += levenshteinSim * weight;
            fieldDetails[key] = {
                dice: diceSim,
                levenshtein: levenshteinSim,
                normalized1: normalized1,
                normalized2: normalized2,
                value1: value1, // Store original value
                value2: value2, // Store original value
            };
        }
        const overallDice = totalWeight > 0 ? weightedDiceSimilarity / totalWeight : 0;
        const overallLevenshtein = totalWeight > 0 ? weightedLevenshteinSimilarity / totalWeight : 0;
        return {
            dice: overallDice,
            levenshtein: overallLevenshtein,
            fieldDetails: fieldDetails
        };
    }
    setupToolHandlers() {
        this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
            tools: [
                {
                    name: 'compare_entities',
                    description: 'Compare two entities using syntactic and optional semantic (LLM) methods.',
                    inputSchema: {
                        type: 'object',
                        properties: {
                            entity1: {
                                type: 'object',
                                description: 'First entity information',
                                additionalProperties: true
                            },
                            entity2: {
                                type: 'object',
                                description: 'Second entity information',
                                additionalProperties: true
                            },
                            threshold: {
                                type: 'number',
                                description: 'Syntactic similarity threshold (0-1, based on Dice) to consider entities as matching',
                                minimum: 0,
                                maximum: 1,
                                default: 0.8
                            },
                            apiKey: {
                                type: 'string',
                                description: '(Optional) Google Generative AI API Key for semantic comparison'
                            },
                        },
                        // Make apiKey optional
                        required: ['entity1', 'entity2']
                    }
                }
            ]
        }));
        this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
            if (request.params.name !== 'compare_entities') {
                throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${request.params.name}`);
            }
            // --- Input Validation ---
            const args = request.params.arguments;
            const entity1 = args.entity1;
            const entity2 = args.entity2;
            const threshold = typeof args.threshold === 'number' ? args.threshold : 0.8;
            const apiKey = args.apiKey; // apiKey is optional
            if (!entity1 || typeof entity1 !== 'object' || !entity2 || typeof entity2 !== 'object') {
                throw new McpError(ErrorCode.InvalidParams, 'Invalid entity format. Both entity1 and entity2 must be objects.');
            }
            // --- Syntactic Comparison ---
            const syntacticSimilarities = this.calculateSyntacticSimilarities(entity1, entity2);
            const isMatchSyntactic = syntacticSimilarities.dice >= threshold;
            // --- Semantic (LLM) Comparison ---
            let llmFieldResults = {};
            let finalLlmAnalysis = null;
            let llmError = null;
            // --- Semantic (LLM) Comparison (only if apiKey is provided) ---
            if (apiKey) {
                try {
                    const genAI = new GoogleGenerativeAI(apiKey);
                    const safetySettings = [
                        { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                        { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
                        { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
                        { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                    ];
                    const comparisonModel = genAI.getGenerativeModel({ model: "gemini-1.5-flash-latest", safetySettings });
                    const fieldComparisonPromises = Object.entries(syntacticSimilarities.fieldDetails).map(async ([key, details]) => {
                        const prompt = `请判断这两个值在语义层面是否一致，仅返回 "true" 或 "false":\n值1: ${JSON.stringify(details.value1)}\n值2: ${JSON.stringify(details.value2)}`;
                        try {
                            const result = await comparisonModel.generateContent(prompt);
                            const textResult = getGeminiResponseText(result);
                            let llmSaysEqual = null;
                            if (textResult !== null) {
                                if (textResult.toLowerCase() === 'true') {
                                    llmSaysEqual = true;
                                }
                                else if (textResult.toLowerCase() === 'false') {
                                    llmSaysEqual = false;
                                }
                                else {
                                    llmSaysEqual = `Unexpected LLM response: ${textResult}`;
                                }
                            }
                            llmFieldResults[key] = { llmSaysEqual };
                        }
                        catch (error) {
                            console.error(`LLM error comparing field ${key}:`, error);
                            llmFieldResults[key] = { llmSaysEqual: null, error: error.message || 'Unknown LLM error' };
                        }
                    });
                    await Promise.all(fieldComparisonPromises);
                    // --- Final LLM Analysis ---
                    const combinedResultsForFinalAnalysis = {
                        syntactic: syntacticSimilarities,
                        semanticFieldChecks: llmFieldResults
                    };
                    const finalPrompt = `综合以下字段的语法和语义比较信息，判断这两个实体是否可能指向同一个真实世界的主体？请提供简要分析和最终判断 (可能匹配/不太可能匹配)。\n\n比较详情:\n${JSON.stringify(combinedResultsForFinalAnalysis, null, 2)}`;
                    try {
                        const finalResult = await comparisonModel.generateContent(finalPrompt);
                        finalLlmAnalysis = getGeminiResponseText(finalResult);
                    }
                    catch (error) {
                        console.error(`LLM error during final analysis:`, error);
                        finalLlmAnalysis = `Error during final analysis: ${error.message || 'Unknown LLM error'}`;
                    }
                }
                catch (error) {
                    console.error("Error during LLM processing:", error);
                    llmError = `LLM Initialization or Processing Error: ${error.message || 'Unknown error'}`;
                    // Ensure field results reflect the error if initialization failed
                    Object.keys(syntacticSimilarities.fieldDetails).forEach(key => {
                        if (!llmFieldResults[key]) {
                            llmFieldResults[key] = { llmSaysEqual: null, error: llmError ?? 'Skipped due to init error' };
                        }
                    });
                    finalLlmAnalysis = `Analysis skipped due to error: ${llmError}`;
                }
            }
            else {
                // --- Case where apiKey is NOT provided ---
                llmError = "API Key not provided. Skipping semantic analysis.";
                finalLlmAnalysis = "Semantic analysis skipped (no API key).";
                // Populate llmFieldResults to indicate skipping
                Object.keys(syntacticSimilarities.fieldDetails).forEach(key => {
                    llmFieldResults[key] = { llmSaysEqual: null, error: 'Skipped (no API key)' };
                });
            }
            // --- Format Final Response ---
            const responseJson = {
                overallSyntacticSimilarity: {
                    dice: syntacticSimilarities.dice,
                    levenshtein: syntacticSimilarities.levenshtein,
                },
                isMatchSyntactic: isMatchSyntactic,
                threshold: threshold,
                matchDetailsSyntactic: `Entities are ${isMatchSyntactic ? 'likely' : 'unlikely'} to be the same based ONLY on Dice similarity (${(syntacticSimilarities.dice * 100).toFixed(2)}%) and threshold ${threshold}.`,
                fieldDetails: Object.entries(syntacticSimilarities.fieldDetails).reduce((acc, [key, details]) => {
                    acc[key] = {
                        ...details, // dice, levenshtein, normalized1/2, value1/2
                        llmSemanticCheck: llmFieldResults[key] || { llmSaysEqual: null, error: 'Not processed' } // Add LLM result per field
                    };
                    return acc;
                }, {}),
                finalLlmAnalysis: finalLlmAnalysis, // Use the value determined above
                llmProcessingError: llmError // Include LLM error message if any
            };
            return {
                content: [
                    {
                        type: 'text',
                        text: JSON.stringify(responseJson, null, 2) // Pretty print JSON
                    }
                ]
            };
        });
    }
    async run() {
        const transport = new StdioServerTransport();
        await this.server.connect(transport);
        console.error('Entity Resolution MCP server running on stdio (v0.2.1)'); // Updated log
    }
}
const server = new EntityResolutionServer();
server.run().catch(console.error);
