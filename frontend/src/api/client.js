
import axios from "axios";
import { API_BASE } from "../config/api";

const client = axios.create({ baseURL: API_BASE, timeout: 120_000 });

export async function inferImage(endpoint, file, onProgress) {
  const form = new FormData();
  form.append("file", file);
  const res = await client.post(endpoint, form, {
    responseType: "blob",
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => onProgress && onProgress(Math.round((e.loaded / e.total) * 100)),
  });
  return { blob: res.data, headers: res.headers };
}

export async function inferImageJson(endpoint, file) {
  const form = new FormData();
  form.append("file", file);
  const res = await client.post(endpoint, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

export function buildStreamUrl(endpoint) {
  return endpoint;
}

export async function healthCheck() {
  const res = await client.get("/health");
  return res.data;
}
