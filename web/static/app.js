const textPromptEl = document.getElementById("textPrompt");
const runTextEl = document.getElementById("runText");
const startRecEl = document.getElementById("startRec");
const stopRecEl = document.getElementById("stopRec");
const runRecordedEl = document.getElementById("runRecorded");
const wavFileEl = document.getElementById("wavFile");
const runUploadEl = document.getElementById("runUpload");
const recordingStateEl = document.getElementById("recordingState");
const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const routingEl = document.getElementById("routing");
const callsEl = document.getElementById("calls");
const executionEl = document.getElementById("execution");

let mediaStream = null;
let audioContext = null;
let sourceNode = null;
let processorNode = null;
let chunks = [];
let sampleRate = 16000;
let recordedBlob = null;

for (const chip of document.querySelectorAll(".chip")) {
  chip.addEventListener("click", () => {
    textPromptEl.value = chip.dataset.text || "";
    textPromptEl.focus();
  });
}

function setBusy(flag, message) {
  runTextEl.disabled = flag;
  runUploadEl.disabled = flag;
  runRecordedEl.disabled = flag || !recordedBlob;
  startRecEl.disabled = flag || !!mediaStream;
  stopRecEl.disabled = flag || !mediaStream;
  statusEl.textContent = message;
}

function writeResult(data) {
  transcriptEl.textContent = data.transcript || data.input || "-";
  routingEl.textContent = [
    `source: ${data.source || "unknown"}`,
    `confidence: ${Number(data.confidence || 0).toFixed(3)}`,
    `time: ${Math.round(Number(data.total_time_ms || 0))} ms`,
    `calendar_id: ${data.calendar_id || "primary"}`,
    `timezone: ${data.timezone || "-"}`,
  ].join("\n");
  callsEl.textContent = JSON.stringify(data.function_calls || [], null, 2);
  executionEl.textContent = JSON.stringify(data.executions || [], null, 2);
}

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok || !data.ok) {
    throw new Error(data.error || "Request failed.");
  }
  return data;
}

async function postAudio(blob) {
  const formData = new FormData();
  formData.append("audio", blob, "voice.wav");
  const res = await fetch("/api/voice-act", { method: "POST", body: formData });
  const data = await res.json();
  if (!res.ok || !data.ok) {
    throw new Error(data.error || "Voice request failed.");
  }
  return data;
}

runTextEl.addEventListener("click", async () => {
  const message = textPromptEl.value.trim();
  if (!message) {
    statusEl.textContent = "Enter a text command first.";
    return;
  }
  setBusy(true, "Running text-to-action...");
  try {
    const data = await postJson("/api/text-act", { message });
    writeResult(data);
    setBusy(false, "Done.");
  } catch (err) {
    setBusy(false, `Error: ${err.message}`);
  }
});

startRecEl.addEventListener("click", async () => {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    sampleRate = audioContext.sampleRate;
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    processorNode = audioContext.createScriptProcessor(4096, 1, 1);
    chunks = [];
    recordedBlob = null;
    runRecordedEl.disabled = true;
    processorNode.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      chunks.push(new Float32Array(input));
    };
    sourceNode.connect(processorNode);
    processorNode.connect(audioContext.destination);
    recordingStateEl.textContent = "Recording...";
    startRecEl.disabled = true;
    stopRecEl.disabled = false;
  } catch (err) {
    recordingStateEl.textContent = `Mic error: ${err.message}`;
  }
});

stopRecEl.addEventListener("click", async () => {
  if (!mediaStream) {
    return;
  }

  processorNode.disconnect();
  sourceNode.disconnect();
  mediaStream.getTracks().forEach((t) => t.stop());
  mediaStream = null;

  if (audioContext) {
    await audioContext.close();
    audioContext = null;
  }

  const samples = mergeChunks(chunks);
  recordedBlob = encodeWav(samples, sampleRate);
  recordingStateEl.textContent = `Recorded ${(samples.length / sampleRate).toFixed(1)}s audio.`;
  startRecEl.disabled = false;
  stopRecEl.disabled = true;
  runRecordedEl.disabled = false;
});

runRecordedEl.addEventListener("click", async () => {
  if (!recordedBlob) {
    statusEl.textContent = "No recording available.";
    return;
  }
  setBusy(true, "Transcribing + executing...");
  try {
    const data = await postAudio(recordedBlob);
    writeResult(data);
    setBusy(false, "Done.");
  } catch (err) {
    setBusy(false, `Error: ${err.message}`);
  }
});

runUploadEl.addEventListener("click", async () => {
  const file = wavFileEl.files && wavFileEl.files[0];
  if (!file) {
    statusEl.textContent = "Choose a WAV file first.";
    return;
  }
  setBusy(true, "Uploading WAV + executing...");
  try {
    const data = await postAudio(file);
    writeResult(data);
    setBusy(false, "Done.");
  } catch (err) {
    setBusy(false, `Error: ${err.message}`);
  }
});

function mergeChunks(chunksList) {
  const total = chunksList.reduce((sum, c) => sum + c.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunksList) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function encodeWav(samples, sampleRateValue) {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRateValue, true);
  view.setUint32(28, sampleRateValue * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);

  floatTo16BitPCM(view, 44, samples);
  return new Blob([view], { type: "audio/wav" });
}

function floatTo16BitPCM(output, offset, input) {
  let currentOffset = offset;
  for (let i = 0; i < input.length; i += 1) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(currentOffset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    currentOffset += 2;
  }
}

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i += 1) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}
