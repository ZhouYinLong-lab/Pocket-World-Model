import * as ort from "onnxruntime-web";
import type { Action, Point, World } from "./sim";

export type OnnxWorldModel = {
  predictPosition: (world: World, action: Action) => Promise<Point | null>;
};

/** Load a checkpoint exported with pocketworld-export-onnx. */
export async function loadPocketWorldModel(modelUrl = "/pocketworld.onnx"): Promise<OnnxWorldModel | null> {
  try {
    const session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"] });
    return {
      predictPosition: async (world, action) => {
        const observation = rasterizeWorld(world);
        const result = await session.run({
          observation: new ort.Tensor("float32", observation, [1, 3, 64, 64]),
          action: new ort.Tensor("int64", BigInt64Array.from([BigInt(action)]), [1]),
        });
        const position = result.next_position as ort.Tensor | undefined;
        if (position) {
          const values = position.data as Float32Array;
          return { x: values[0] * 64, y: values[1] * 64 };
        }
        const output = result.next_observation as ort.Tensor;
        return findAgent(output.data as Float32Array);
      },
    };
  } catch {
    return null;
  }
}

function rasterizeWorld(world: World): Float32Array {
  const pixels = new Float32Array(3 * 64 * 64);
  for (let index = 0; index < pixels.length; index += 1) pixels[index] = 15 / 255;
  for (let y = 0; y < 64; y += 8) for (let x = 0; x < 64; x += 1) pixels[(0 * 64 + y) * 64 + x] = 17 / 255;
  for (let x = 0; x < 64; x += 8) for (let y = 0; y < 64; y += 1) pixels[(0 * 64 + y) * 64 + x] = 17 / 255;
  for (const wall of world.walls) fillRect(pixels, wall.x, wall.y, wall.w, wall.h, [75, 93, 113]);
  fillDisc(pixels, world.goal.x, world.goal.y, 4, [247, 190, 69]);
  fillDisc(pixels, world.position.x, world.position.y, 3, [93, 224, 183]);
  return pixels;
}

function fillRect(pixels: Float32Array, x: number, y: number, width: number, height: number, color: number[]): void {
  for (let row = Math.max(0, Math.floor(y)); row < Math.min(64, Math.ceil(y + height)); row += 1) {
    for (let column = Math.max(0, Math.floor(x)); column < Math.min(64, Math.ceil(x + width)); column += 1) {
      for (let channel = 0; channel < 3; channel += 1) pixels[(channel * 64 + row) * 64 + column] = color[channel] / 255;
    }
  }
}

function fillDisc(pixels: Float32Array, x: number, y: number, radius: number, color: number[]): void {
  for (let row = Math.max(0, Math.floor(y - radius)); row < Math.min(64, Math.ceil(y + radius)); row += 1) {
    for (let column = Math.max(0, Math.floor(x - radius)); column < Math.min(64, Math.ceil(x + radius)); column += 1) {
      if ((column - x) ** 2 + (row - y) ** 2 <= radius ** 2) for (let channel = 0; channel < 3; channel += 1) pixels[(channel * 64 + row) * 64 + column] = color[channel] / 255;
    }
  }
}

function findAgent(data: Float32Array): Point | null {
  let xTotal = 0;
  let yTotal = 0;
  let count = 0;
  for (let y = 0; y < 64; y += 1) for (let x = 0; x < 64; x += 1) {
    const red = data[y * 64 + x];
    const green = data[64 * 64 + y * 64 + x];
    const blue = data[2 * 64 * 64 + y * 64 + x];
    if (green > red + 0.08 && green > blue + 0.05) { xTotal += x; yTotal += y; count += 1; }
  }
  return count ? { x: xTotal / count, y: yTotal / count } : null;
}
