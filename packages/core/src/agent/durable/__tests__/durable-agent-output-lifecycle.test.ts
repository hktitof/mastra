/**
 * DurableAgent output processor lifecycle guarantees (issue #22980).
 *
 * Three contracts the durable path must honor, mirroring the ordinary agent:
 * 1. A tool result goes through processToolResult exactly once (dedicated hook)
 *    and through processOutputStream exactly once (public stream) — not twice
 *    through the stream hook and never through the dedicated hook.
 * 2. A processOutputStep tripwire that requests a retry makes the durable agent
 *    call the model again (bounded by maxProcessorRetries), instead of ending
 *    the run.
 * 3. A terminal tripwire stays recorded, but the public durable stream still
 *    reaches its final finish chunk instead of terminating at the tripwire.
 */

import type { LanguageModelV2 } from '@ai-sdk/provider-v5';
import { MockLanguageModelV2, convertArrayToReadableStream } from '@internal/ai-sdk-v5/test';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { z } from 'zod';
import { EventEmitterPubSub } from '../../../events/event-emitter';
import { Mastra } from '../../../mastra';
import { InMemoryStore } from '../../../storage';
import { createTool } from '../../../tools';
import { Agent } from '../../agent';
import { createDurableAgent } from '../create-durable-agent';

function createToolCallingModel(toolName: string, toolArgs: Record<string, unknown>) {
  return new MockLanguageModelV2({
    doStream: async () => ({
      stream: convertArrayToReadableStream([
        { type: 'stream-start', warnings: [] },
        { type: 'response-metadata', id: 'resp-1', modelId: 'mock', timestamp: new Date(0) },
        {
          type: 'tool-call',
          id: 'tc-1',
          toolCallType: 'function',
          toolCallId: 'tc-1',
          toolName,
          args: JSON.stringify(toolArgs),
        },
        {
          type: 'finish',
          finishReason: 'tool-calls',
          usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
        },
      ]),
      rawCall: { rawPrompt: null, rawSettings: {} },
      warnings: [],
    }),
  });
}

function createCountingTextModel(callCount: { value: number }) {
  return new MockLanguageModelV2({
    doStream: async () => {
      callCount.value++;
      return {
        stream: convertArrayToReadableStream([
          { type: 'stream-start', warnings: [] },
          { type: 'response-metadata', id: 'resp-' + callCount.value, modelId: 'mock', timestamp: new Date(0) },
          { type: 'text-start', id: 'text-1' },
          { type: 'text-delta', id: 'text-1', delta: 'Hello.' },
          { type: 'text-end', id: 'text-1' },
          {
            type: 'finish',
            finishReason: 'stop',
            usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
          },
        ]),
        rawCall: { rawPrompt: null, rawSettings: {} },
        warnings: [],
      };
    },
  });
}

async function drain(stream: ReadableStream<any>) {
  const out: any[] = [];
  for await (const c of stream) out.push(c);
  return out;
}

describe('DurableAgent output processor lifecycle', () => {
  let pubsub: EventEmitterPubSub;

  beforeEach(() => {
    pubsub = new EventEmitterPubSub();
  });

  afterEach(async () => {
    await pubsub.close();
  });

  it('dispatches processToolResult once and processOutputStream once per tool result', async () => {
    const processToolResult = vi.fn(async () => {});
    const processOutputStream = vi.fn(async ({ part }: any) => part);

    const outputProcessor = {
      id: 'counting-processor',
      name: 'Counting Processor',
      processToolResult,
      processOutputStream,
    };

    const weatherTool = createTool({
      id: 'getWeather',
      description: 'Get weather',
      inputSchema: z.object({ city: z.string() }),
      outputSchema: z.object({ temp: z.number() }),
      execute: async () => ({ temp: 72 }),
    });

    const baseAgent = new Agent({
      id: 'lifecycle-agent',
      name: 'Lifecycle Agent',
      instructions: 'You are a helpful agent.',
      model: createToolCallingModel('getWeather', { city: 'NYC' }) as LanguageModelV2,
      tools: { getWeather: weatherTool },
      outputProcessors: [outputProcessor as any],
    });

    const durableAgent = createDurableAgent({ agent: baseAgent, pubsub });
    new Mastra({
      agents: { 'lifecycle-agent': durableAgent as any },
      logger: false,
      storage: new InMemoryStore(),
      pubsub,
    });

    const result = await durableAgent.stream('What is the weather in NYC?', {
      maxSteps: 3,
    });

    await drain(result.fullStream);

    const streamToolResultCalls = processOutputStream.mock.calls.filter(
      ([ctx]: any[]) => ctx?.part?.type === 'tool-result',
    );

    expect(processToolResult).toHaveBeenCalledTimes(1);
    expect(streamToolResultCalls.length).toBe(1);
  });

  it('retries the model call when processOutputStep requests a retry', async () => {
    const callCount = { value: 0 };
    let outputStepCalls = 0;

    const retryProcessor = {
      id: 'retry-processor',
      name: 'Retry Processor',
      processOutputStep: vi.fn(async ({ abort }: any) => {
        outputStepCalls++;
        if (outputStepCalls === 1) {
          abort('not good enough', { retry: true });
        }
      }),
    };

    const baseAgent = new Agent({
      id: 'retry-agent',
      name: 'Retry Agent',
      instructions: 'You are a helpful agent.',
      model: createCountingTextModel(callCount) as unknown as LanguageModelV2,
      outputProcessors: [retryProcessor as any],
    });

    const durableAgent = createDurableAgent({ agent: baseAgent, pubsub });
    new Mastra({
      agents: { 'retry-agent': durableAgent as any },
      logger: false,
      storage: new InMemoryStore(),
      pubsub,
    });

    const result = await durableAgent.stream('Say something.', {
      maxProcessorRetries: 1,
    });

    await drain(result.fullStream);

    expect(outputStepCalls).toBe(2);
    expect(callCount.value).toBe(2);
  });

  it('reaches finish after a terminal tripwire and keeps the tripwire recorded', async () => {
    const tripwireProcessor = {
      id: 'blocking-processor',
      name: 'Blocking Processor',
      processOutputStep: vi.fn(async ({ abort }: any) => {
        abort('blocked content', {});
      }),
    };

    const baseAgent = new Agent({
      id: 'tripwire-agent',
      name: 'Tripwire Agent',
      instructions: 'You are a helpful agent.',
      model: createCountingTextModel({ value: 0 }) as unknown as LanguageModelV2,
      outputProcessors: [tripwireProcessor as any],
    });

    const durableAgent = createDurableAgent({ agent: baseAgent, pubsub });
    new Mastra({
      agents: { 'tripwire-agent': durableAgent as any },
      logger: false,
      storage: new InMemoryStore(),
      pubsub,
    });

    const result = await durableAgent.stream('Say something.', {});

    const chunks = await drain(result.fullStream);

    const tripwireChunks = chunks.filter((c: any) => c.type === 'tripwire');
    expect(tripwireChunks.length).toBeGreaterThan(0);
    const last = chunks[chunks.length - 1];
    expect(last.type).toBe('finish');
  });
});
