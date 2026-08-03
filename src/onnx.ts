import * as ort from "onnxruntime-web";

/** Optional browser runtime for a checkpoint exported with pocketworld-export-onnx. */
export async function loadPocketWorldModel(modelUrl = "/pocketworld.onnx"): Promise<ort.InferenceSession | null> {
  try {
    return await ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"] });
  } catch {
    return null;
  }
}

