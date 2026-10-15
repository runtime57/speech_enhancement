"use strict";

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "0:00";
  return `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;
}

function renderSample(sample, models) {
  const card = document.querySelector("#sample-template").content.firstElementChild.cloneNode(true);
  card.dataset.sample = sample.id;
  card.querySelector("h2").textContent = sample.label;
  card.querySelector(".sample-badge").textContent = `${sample.duration_seconds.toFixed(2)} s · 16 kHz`;
  const audio = card.querySelector("audio");
  const image = card.querySelector(".spectrogram");
  const play = card.querySelector(".play-btn");
  const progress = card.querySelector(".progress");
  const current = card.querySelector(".current-time");
  const duration = card.querySelector(".duration");
  const status = card.querySelector(".player-status");
  const variants = [{id: "noisy", label: "Noisy"}, {id: "clean", label: "Clean reference"}, ...models];
  const audioCache = new Map();
  let pending = null;
  let selection = 0;

  function updatePlayButton() {
    play.textContent = audio.paused ? "▶" : "❚❚";
    play.setAttribute("aria-label", audio.paused ? "Play audio" : "Pause audio");
  }

  async function startPlayback() {
    try {
      await audio.play();
      status.textContent = "";
    } catch (error) {
      if (error.name !== "AbortError") status.textContent = "Playback could not start. Press Play to try again.";
    }
    updatePlayButton();
  }

  async function selectVariant(variant) {
    const position = pending?.position ?? (audio.readyState >= 1 ? audio.currentTime : 0);
    const resume = pending?.resume ?? !audio.paused;
    const version = ++selection;
    audio.pause();
    pending = {position, resume, version};
    play.disabled = true;
    progress.disabled = true;
    const paths = sample.variants[variant.id];
    card.querySelectorAll(".variant-btn").forEach(button => {
      const selected = button.dataset.variant === variant.id;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    image.src = `${paths.image}?v=${paths.image_sha256.slice(0, 12)}`;
    image.alt = `${sample.label}: ${variant.label} spectrogram, 0–8 kHz, shared −80 to 0 dB scale`;
    const info = card.querySelector(".model-info");
    info.textContent = variant.parameters
      ? `${variant.label} · ${(variant.parameters / 1e6).toFixed(3)} M parameters · epoch ${variant.epoch} · `
      : (variant.id === "noisy" ? "Original noisy recording" : "Clean reference recording");
    if (variant.weights_url) {
      const link = document.createElement("a");
      link.href = `${variant.weights_url}?v=${variant.weights_sha256.slice(0, 12)}`;
      link.textContent = "Download weights";
      link.setAttribute("download", "");
      info.append(link);
    }
    status.textContent = "Loading audio…";
    updatePlayButton();
    try {
      // Short clips are cached as blobs so A/B seeking also works on static servers without range requests.
      if (!audioCache.has(paths.audio)) {
        audioCache.set(paths.audio, fetch(`${paths.audio}?v=${paths.audio_sha256.slice(0, 12)}`)
          .then(response => {
            if (!response.ok) throw new Error(`Audio HTTP ${response.status}`);
            return response.blob();
          }).then(blob => URL.createObjectURL(blob)).catch(error => {
            audioCache.delete(paths.audio);
            throw error;
          }));
      }
      const url = await audioCache.get(paths.audio);
      if (version !== selection) return;
      pending.url = url;
      audio.dataset.source = paths.audio;
      audio.src = url;
      audio.load();
    } catch (error) {
      if (version !== selection) return;
      pending = null;
      status.textContent = "Audio could not load. Select a variant to try again.";
    }
  }

  variants.forEach(variant => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "variant-btn";
    button.dataset.variant = variant.id;
    button.textContent = variant.label;
    button.addEventListener("click", () => selectVariant(variant));
    card.querySelector(".buttons").append(button);
  });
  audio.addEventListener("loadedmetadata", () => {
    duration.textContent = formatTime(audio.duration);
    if (pending?.version === selection && pending.url === audio.src) {
      const request = pending;
      pending = null;
      audio.currentTime = Math.min(request.position, Math.max(0, audio.duration - .01));
      play.disabled = false;
      progress.disabled = false;
      status.textContent = "";
      if (request.resume) startPlayback();
    }
  });
  audio.addEventListener("timeupdate", () => {
    current.textContent = formatTime(audio.currentTime);
    progress.value = audio.duration ? audio.currentTime / audio.duration * 100 : 0;
  });
  audio.addEventListener("play", () => {
    document.querySelectorAll("audio").forEach(other => { if (other !== audio) other.pause(); });
    updatePlayButton();
  });
  audio.addEventListener("pause", updatePlayButton);
  audio.addEventListener("ended", updatePlayButton);
  audio.addEventListener("error", () => { status.textContent = "Audio could not load. Select another variant or reload the page."; });
  play.addEventListener("click", () => audio.paused ? startPlayback() : audio.pause());
  progress.addEventListener("input", () => {
    if (Number.isFinite(audio.duration)) audio.currentTime = Number(progress.value) / 100 * audio.duration;
  });
  selectVariant(variants[0]);
  return card;
}

async function main() {
  const status = document.querySelector("#load-status");
  try {
    const response = await fetch("demo_manifest.json", {cache: "no-cache"});
    if (!response.ok) throw new Error(`Manifest HTTP ${response.status}`);
    const data = await response.json();
    data.samples.forEach(sample => document.querySelector(".samples").append(renderSample(sample, data.models)));
    status.hidden = true;
  } catch (error) {
    status.textContent = "Examples could not load. Reload the page to try again.";
    console.error(error);
  }
}

main();
