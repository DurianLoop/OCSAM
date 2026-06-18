r"""Canvas web demo for SAM-style click, box, and text-prompt segmentation.

Run from D:\\SAM\\code after activating the GPU environment:

    python demos\medical_sam_click_app.py --host 127.0.0.1 --port 7860

Open http://127.0.0.1:7860, choose an image, then use one of three prompts:
click, box, or text. Click and box are native SAM prompts. Text mode is an
offline-friendly SAM automatic-mask candidate selector that can be replaced by
Grounded-SAM/CLIP later if those weights are added.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib_gpu"))

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
SAM3_ROOT = WORKSPACE_ROOT / "sam3"
SAM3_RUNTIME_DEPS = SAM3_ROOT / ".deps_runtime"
SAM2_ROOT = WORKSPACE_ROOT / "sam2"
sys.path.insert(0, str(REPO_ROOT))
if SAM3_RUNTIME_DEPS.exists():
    sys.path.insert(0, str(SAM3_RUNTIME_DEPS))
if SAM3_ROOT.exists():
    sys.path.insert(0, str(SAM3_ROOT))
if SAM2_ROOT.exists():
    sys.path.insert(0, str(SAM2_ROOT))

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from PIL import Image

from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from workbench.registry import registry_payload
from workbench.samples import SampleEntry, find_image_samples, find_video_samples, sample_manifest_payload


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OCSAM Prompt Demo</title>
  <style>
    :root {
      color-scheme: light;
      font-family: "IBM Plex Sans", "Microsoft YaHei", "Segoe UI", sans-serif;
      background: #ffffff;
      color: #161616;
      --cds-background: #ffffff;
      --cds-layer-01: #f4f4f4;
      --cds-layer-02: #e0e0e0;
      --cds-layer-hover: #e8e8e8;
      --cds-text-primary: #161616;
      --cds-text-secondary: #525252;
      --cds-text-placeholder: #6f6f6f;
      --cds-border-subtle: #c6c6c6;
      --cds-border-light: #e0e0e0;
      --cds-blue-60: #0f62fe;
      --cds-blue-70: #0043ce;
      --cds-blue-10: #edf5ff;
      --cds-green-50: #24a148;
      --cds-red-60: #da1e28;
      --cds-yellow-30: #f1c21b;
      --stage: #161616;
      --mono: "IBM Plex Mono", "Cascadia Mono", "Consolas", monospace;
    }
    * { box-sizing: border-box; }
    body {
      background: var(--cds-background);
      margin: 0;
      height: 100vh;
      overflow: hidden;
    }
    .app-shell {
      display: grid;
      grid-template-columns: 72px minmax(0, 1fr);
      height: 100vh;
      overflow: hidden;
      width: 100vw;
    }
    .rail {
      background: #161616;
      color: #f4f4f4;
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: 100vh;
      padding: 16px 0;
    }
    .brand-mark {
      align-items: center;
      border: 1px solid #393939;
      display: flex;
      font-family: var(--mono);
      font-size: 15px;
      font-weight: 600;
      height: 40px;
      justify-content: center;
      letter-spacing: 0;
      margin: 0 auto;
      width: 40px;
    }
    .rail-stack {
      align-content: start;
      display: grid;
      gap: 8px;
      margin-top: 32px;
    }
    .rail-item {
      background: transparent;
      border: 0;
      border-left: 3px solid transparent;
      border-bottom: 0;
      color: #c6c6c6;
      font-size: 14px;
      justify-self: stretch;
      min-height: 48px;
      padding: 14px 0;
      text-align: center;
      width: 100%;
    }
    .rail-item:hover { background: #262626; color: #ffffff; }
    .rail-item.active {
      border-left-color: var(--cds-blue-60);
      color: #ffffff;
    }
    .workbench {
      display: grid;
      grid-template-rows: 96px minmax(0, 1fr);
      height: 100vh;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
      position: relative;
    }
    .topbar {
      align-items: stretch;
      border-bottom: 1px solid var(--cds-border-light);
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 560px);
      min-height: 96px;
      min-width: 0;
      overflow: hidden;
    }
    .title-block {
      align-content: center;
      display: grid;
      gap: 4px;
      padding: 16px 24px;
    }
    h1 {
      color: var(--cds-text-primary);
      font-size: 32px;
      font-weight: 300;
      line-height: 1.25;
      margin: 0;
      text-wrap: pretty;
    }
    .sub {
      color: var(--cds-text-secondary);
      font-size: 14px;
      letter-spacing: 0.16px;
      line-height: 1.35;
      margin: 0;
    }
    .top-metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      min-width: 0;
    }
    .metric {
      border-left: 1px solid var(--cds-border-light);
      display: grid;
      gap: 8px;
      padding: 16px;
    }
    .metric-label {
      color: var(--cds-text-secondary);
      font-family: var(--mono);
      font-size: 12px;
      letter-spacing: 0.32px;
      line-height: 1.33;
    }
    .metric-value {
      color: var(--cds-text-primary);
      font-size: 18px;
      font-weight: 400;
      line-height: 1.25;
    }
    .metric-value.state-good { color: var(--cds-green-50); }
    .metric-value.state-warn { color: #8a6a00; }
    .metric-value.state-busy { color: var(--cds-blue-60); }
    main.workspace {
      background: var(--cds-layer-01);
      display: grid;
      gap: 1px;
      grid-template-columns: clamp(220px, 18vw, 280px) minmax(0, 1fr) clamp(260px, 22vw, 320px);
      height: calc(100vh - 96px);
      min-height: 0;
      min-width: 0;
      overflow: hidden;
      padding: 0;
      width: 100%;
    }
    section { min-width: 0; }
    .panel {
      background: var(--cds-background);
      border: 0;
      border-radius: 0;
      box-shadow: none;
      padding: 0;
    }
    .controls {
      display: grid;
      gap: 0;
      grid-auto-rows: max-content;
      height: calc(100vh - 96px);
      min-height: 0;
      overflow: hidden;
    }
    .control-block {
      border-top: 1px solid var(--cds-border-light);
      display: grid;
      gap: 8px;
      padding: 12px;
    }
    .control-block:first-child { border-top: 0; }
    .control-title {
      align-items: center;
      display: flex;
      gap: 10px;
      justify-content: space-between;
    }
    label {
      color: var(--cds-text-secondary);
      display: block;
      font-size: 12px;
      font-weight: 400;
      letter-spacing: 0.32px;
      line-height: 1.33;
      margin: 0 0 4px;
    }
    .control-title label { margin: 0; }
    input[type="file"], input[type="text"], select, button {
      box-sizing: border-box;
      font: inherit;
      min-height: 36px;
      width: 100%;
    }
    input[type="text"], select, button {
      border: 0;
      border-bottom: 2px solid transparent;
      border-radius: 0;
      background: var(--cds-layer-01);
      color: var(--cds-text-primary);
      padding: 0 16px;
    }
    input[type="file"] {
      background: var(--cds-layer-01);
      border: 1px dashed var(--cds-border-subtle);
      border-radius: 0;
      color: var(--cds-text-secondary);
      font-size: 13px;
      padding: 8px 10px;
    }
    input.file-hidden {
      height: 1px;
      opacity: 0;
      overflow: hidden;
      padding: 0;
      position: absolute;
      width: 1px;
    }
    input[type="text"]:focus, select:focus, button:focus {
      border-bottom-color: var(--cds-blue-60);
      outline: 2px solid var(--cds-blue-60);
      outline-offset: -2px;
    }
    button {
      cursor: pointer;
      font-size: 14px;
      font-weight: 400;
      letter-spacing: 0.16px;
      transition: background 140ms ease, color 140ms ease, border-color 140ms ease;
    }
    button:hover { background: var(--cds-layer-hover); }
    button:disabled {
      color: #8d8d8d;
      cursor: not-allowed;
      background: var(--cds-layer-01);
    }
    body[data-busy="true"] button:disabled,
    body[data-busy="true"] input:disabled,
    body[data-busy="true"] select:disabled {
      cursor: wait;
    }
    .primary {
      background: var(--cds-blue-60);
      border-bottom-color: var(--cds-blue-60);
      color: #ffffff;
    }
    .primary:hover { background: #0353e9; }
    .secondary {
      background: #393939;
      color: #ffffff;
    }
    .secondary:hover { background: #4c4c4c; }
    .mini-action {
      font-size: 12px;
      min-height: 32px;
      padding: 4px 8px;
      width: auto;
    }
    .tabs {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 1px;
      background: var(--cds-border-light);
    }
    .tab { background: var(--cds-layer-01); }
    .tab[aria-pressed="true"] {
      background: var(--cds-blue-60);
      color: #ffffff;
    }
    .tab:disabled {
      border-style: dashed;
      opacity: 0.6;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .row.three {
      grid-template-columns: repeat(3, 1fr);
    }
    .row.four {
      grid-template-columns: repeat(4, 1fr);
    }
    .action-row {
      grid-template-columns: 1.2fr 1fr;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .chip {
      align-items: center;
      background: var(--cds-layer-01);
      border: 1px solid var(--cds-border-light);
      border-radius: 999px;
      color: var(--cds-text-primary);
      display: inline-flex;
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 400;
      letter-spacing: 0.16px;
      line-height: 1;
      min-height: 28px;
      padding: 6px 10px;
      white-space: nowrap;
    }
    .chip.good { background: #defbe6; border-color: #a7f0ba; color: #0e6027; }
    .chip.warn { background: #fcf4d6; border-color: #f1c21b; color: #684e00; }
    .chip.busy { background: var(--cds-blue-10); border-color: #78a9ff; color: var(--cds-blue-70); }
    .examples {
      background: var(--cds-layer-01);
      display: grid;
      gap: 1px;
      max-height: 168px;
      overflow: auto;
    }
    .example-btn {
      min-height: 32px;
      overflow: hidden;
      background: var(--cds-background);
      border-bottom: 0;
      text-align: left;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .example-btn[aria-pressed="true"] {
      background: var(--cds-blue-10);
      box-shadow: inset 3px 0 0 var(--cds-blue-60);
      color: var(--cds-blue-70);
    }
    .status {
      background: var(--cds-layer-01);
      border-left: 3px solid var(--cds-border-subtle);
      color: var(--cds-text-primary);
      font-size: 13px;
      letter-spacing: 0.16px;
      line-height: 1.45;
      min-height: 52px;
      max-height: 120px;
      overflow: auto;
      padding: 10px 11px;
      white-space: pre-wrap;
    }
    .status[data-kind="ok"] { border-left-color: var(--cds-green-50); background: #defbe6; }
    .status[data-kind="warn"] { border-left-color: var(--cds-yellow-30); background: #fcf4d6; }
    .status[data-kind="error"] { border-left-color: var(--cds-red-60); background: #fff1f1; }
    .status[data-kind="busy"] { border-left-color: var(--cds-blue-60); background: var(--cds-blue-10); }
    .stage {
      background: var(--cds-background);
      display: grid;
      grid-template-rows: 56px minmax(0, 1fr) 72px;
      gap: 1px;
      align-items: start;
      height: calc(100vh - 96px);
      min-height: 0;
      overflow: hidden;
    }
    .stage-toolbar {
      align-items: center;
      background: var(--cds-background);
      border-bottom: 1px solid var(--cds-border-light);
      display: flex;
      gap: 16px;
      justify-content: space-between;
      min-height: 56px;
      padding: 8px 16px;
    }
    .tool-tabs {
      display: flex;
      gap: 1px;
      background: var(--cds-border-light);
    }
    .tool-tabs .tab {
      min-width: 96px;
    }
    .stage-grid {
      display: grid;
      gap: 1px;
      grid-template-columns: 1fr 1fr;
      height: 100%;
      min-height: 0;
      overflow: hidden;
    }
    .run-strip {
      background: #262626;
      color: #f4f4f4;
      display: grid;
      gap: 1px;
      grid-template-columns: repeat(4, 1fr);
      min-height: 72px;
    }
    .run-cell {
      border-left: 1px solid #393939;
      display: grid;
      gap: 6px;
      padding: 12px 16px;
    }
    .run-cell:first-child { border-left: 0; }
    .run-label {
      color: #a8a8a8;
      font-family: var(--mono);
      font-size: 12px;
      letter-spacing: 0.32px;
    }
    .run-value {
      color: #f4f4f4;
      font-size: 14px;
      letter-spacing: 0.16px;
    }
    .viewport {
      align-items: center;
      background: var(--stage);
      display: flex;
      justify-content: center;
      min-height: 0;
      overflow: hidden;
      padding: 12px;
      position: relative;
    }
    canvas, .result-img {
      display: block;
      height: auto;
      max-width: 100%;
      max-height: 100%;
    }
    canvas {
      cursor: crosshair;
      margin: 0 auto;
    }
    .result-img {
      margin: 0 auto;
    }
    .compare-grid {
      align-content: center;
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      height: 100%;
      width: 100%;
    }
    .compare-card {
      background: #262626;
      border: 1px solid #393939;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-height: 0;
    }
    .compare-title {
      align-items: center;
      border-bottom: 1px solid #393939;
      color: #f4f4f4;
      display: flex;
      font-family: var(--mono);
      font-size: 12px;
      justify-content: space-between;
      letter-spacing: 0.32px;
      min-height: 32px;
      padding: 6px 8px;
    }
    .compare-card img {
      align-self: center;
      display: block;
      justify-self: center;
      max-height: 100%;
      max-width: 100%;
    }
    .compare-error {
      align-self: center;
      color: #ffb3b8;
      font-size: 12px;
      justify-self: center;
      line-height: 1.45;
      padding: 12px;
      text-align: center;
    }
    .caption {
      color: var(--cds-text-secondary);
      font-size: 13px;
      letter-spacing: 0.16px;
      margin: 0 0 8px;
    }
    .muted {
      color: var(--cds-text-secondary);
      font-size: 12px;
      letter-spacing: 0.32px;
      line-height: 1.45;
      margin: 6px 0 0;
    }
    .empty {
      border: 1px solid #393939;
      color: #c6c6c6;
      font-family: var(--mono);
      font-size: 13px;
      line-height: 1.5;
      max-width: 280px;
      padding: 24px;
      text-align: center;
    }
    .support-state {
      color: var(--cds-text-secondary);
      font-size: 12px;
      letter-spacing: 0.32px;
      line-height: 1.4;
      margin: 0;
    }
    .support-state.ready { color: var(--cds-green-50); font-weight: 600; }
    .upload-stack {
      display: grid;
      gap: 8px;
    }
    .upload-card {
      background: var(--cds-layer-01);
      border: 1px dashed var(--cds-border-subtle);
      color: var(--cds-text-primary);
      cursor: pointer;
      display: grid;
      gap: 4px;
      min-height: 64px;
      padding: 10px 12px;
    }
    .upload-card:hover { background: var(--cds-layer-hover); }
    .upload-title {
      color: var(--cds-text-primary);
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 0.16px;
    }
    .upload-meta {
      color: var(--cds-text-secondary);
      font-size: 12px;
      line-height: 1.4;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .upload-hint {
      color: var(--cds-text-placeholder);
      font-size: 12px;
      line-height: 1.4;
    }
    .split-title {
      align-items: center;
      background: var(--cds-background);
      border-bottom: 1px solid var(--cds-border-light);
      display: flex;
      justify-content: space-between;
      gap: 10px;
      min-height: 40px;
      padding: 8px 12px;
    }
    .split-title .caption { margin: 0; }
    .viewer-panel {
      background: var(--cds-background);
      display: grid;
      grid-template-rows: 40px 40px minmax(0, 1fr);
      height: 100%;
      min-width: 0;
      min-height: 0;
    }
    .media-band {
      align-items: center;
      background: #262626;
      border-bottom: 1px solid #393939;
      color: #f4f4f4;
      display: flex;
      gap: 16px;
      justify-content: space-between;
      min-height: 40px;
      padding: 0 12px;
    }
    .media-band span {
      color: #c6c6c6;
      font-family: var(--mono);
      font-size: 12px;
      letter-spacing: 0.32px;
    }
    .rail-drawer {
      background: var(--cds-background);
      border-left: 1px solid var(--cds-border-light);
      box-shadow: -18px 0 40px rgba(0, 0, 0, 0.16);
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      height: calc(100vh - 96px);
      max-width: 380px;
      min-width: 320px;
      overflow: hidden;
      position: absolute;
      right: 0;
      top: 96px;
      width: min(380px, 34vw);
      z-index: 20;
    }
    .rail-drawer[hidden] { display: none; }
    .drawer-head {
      align-items: center;
      border-bottom: 1px solid var(--cds-border-light);
      display: flex;
      justify-content: space-between;
      min-height: 56px;
      padding: 12px 16px;
    }
    .drawer-body {
      display: grid;
      gap: 0;
      overflow: auto;
    }
    .drawer-panel[hidden] { display: none; }
    .drawer-section {
      border-bottom: 1px solid var(--cds-border-light);
      display: grid;
      gap: 10px;
      padding: 14px 16px;
    }
    .kv-list {
      display: grid;
      gap: 1px;
      background: var(--cds-border-light);
    }
    .kv-row {
      background: var(--cds-background);
      display: grid;
      grid-template-columns: 112px minmax(0, 1fr);
      gap: 12px;
      min-height: 34px;
      padding: 8px 10px;
    }
    .kv-key {
      color: var(--cds-text-secondary);
      font-family: var(--mono);
      font-size: 12px;
      letter-spacing: 0.32px;
    }
    .kv-value {
      color: var(--cds-text-primary);
      font-size: 13px;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .drawer-note {
      color: var(--cds-text-secondary);
      font-size: 13px;
      line-height: 1.45;
      margin: 0;
      text-wrap: pretty;
    }
    .cap-list {
      display: grid;
      gap: 8px;
    }
    .cap-item {
      background: var(--cds-layer-01);
      display: grid;
      gap: 4px;
      padding: 10px;
    }
    .cap-name {
      color: var(--cds-text-primary);
      font-size: 13px;
      font-weight: 600;
    }
    .cap-text {
      color: var(--cds-text-secondary);
      font-size: 12px;
      line-height: 1.45;
      margin: 0;
    }
    .quick-stack {
      display: grid;
      gap: 1px;
    }
    .group-heading {
      color: var(--cds-text-primary);
      font-size: 16px;
      font-weight: 400;
      line-height: 1.35;
      margin: 0;
    }
    .group-meta {
      color: var(--cds-text-secondary);
      font-family: var(--mono);
      font-size: 12px;
      letter-spacing: 0.32px;
      margin: 0;
    }
    @media (prefers-reduced-motion: no-preference) {
      .panel, .chip, button, .status, .example-btn {
        transition: background 140ms ease, color 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
      }
    }
    @media (max-width: 1120px) {
      .app-shell { grid-template-columns: 1fr; }
      .rail { display: none; }
      .topbar { grid-template-columns: 1fr; }
      .top-metrics { min-width: 0; }
      main.workspace { grid-template-columns: 1fr; }
      .rail-drawer {
        height: calc(100vh - 96px);
        max-width: none;
        width: min(420px, 92vw);
      }
      .controls {
        max-height: none;
        position: static;
      }
      .stage-grid, .row { grid-template-columns: 1fr; }
      .viewport { min-height: 360px; }
      .run-strip { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 640px) {
      .top-metrics { grid-template-columns: repeat(2, 1fr); }
      h1 { font-size: 26px; }
      .title-block { padding: 16px; }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <aside class="rail" aria-label="Workbench navigation">
      <div class="brand-mark">OC</div>
      <div class="rail-stack">
        <button class="rail-item active" type="button" data-panel="lab">Lab</button>
        <button class="rail-item" type="button" data-panel="run">Run</button>
        <button class="rail-item" type="button" data-panel="set">Set</button>
      </div>
      <button class="rail-item" type="button" data-panel="set">SAM</button>
    </aside>
    <div class="workbench">
      <header class="topbar">
        <div class="title-block">
          <h1>OCSAM Prompt Workbench</h1>
          <p class="sub">SAM / MedSAM / SAM2 / Matcher / SAM3 unified prompt segmentation</p>
        </div>
        <div class="top-metrics" aria-label="Current run state">
          <div class="metric">
            <span class="metric-label">MEDIA</span>
            <span id="mediaChip" class="metric-value">Image</span>
          </div>
          <div class="metric">
            <span class="metric-label">MODEL</span>
            <span id="modelChip" class="metric-value">SAM ViT-B</span>
          </div>
          <div class="metric">
            <span class="metric-label">PROMPT</span>
            <span id="promptChip" class="metric-value">Click</span>
          </div>
          <div class="metric">
            <span class="metric-label">MATCHER</span>
            <span id="supportChip" class="metric-value">No support</span>
          </div>
        </div>
      </header>
      <main class="workspace">
        <section class="panel controls library-panel">
          <div class="control-block">
            <p class="group-heading">Media Library</p>
            <p class="group-meta">Target image / SAM2 clip</p>
          </div>
          <div class="control-block">
            <label for="file">Upload Image</label>
            <input id="file" type="file" accept="image/*">
          </div>
          <div class="control-block">
            <label>Local Examples</label>
            <div id="examples" class="examples"></div>
          </div>
          <div class="control-block">
            <label>SAM2 Video Examples</label>
            <div id="videos" class="examples"></div>
          </div>
          <div class="control-block">
            <label>Status</label>
            <div id="status" class="status">Waiting for an image.</div>
          </div>
        </section>
        <section class="stage">
          <div class="stage-toolbar">
            <div>
              <p class="group-heading">Segmentation Stage</p>
              <p class="group-meta">Prompt on the left, inspect mask on the right</p>
            </div>
            <div class="tool-tabs">
              <button class="tab" data-mode="click" aria-pressed="true">Click</button>
              <button class="tab" data-mode="box" aria-pressed="false">Box</button>
              <button class="tab" data-mode="text" aria-pressed="false">Text</button>
            </div>
          </div>
          <div class="stage-grid">
            <div class="viewer-panel">
              <div class="split-title">
                <p class="caption" id="inputCaption">Input Image</p>
                <p class="caption" id="canvasSize">No image</p>
              </div>
              <div class="media-band"><strong>Prompt Canvas</strong><span>Input frame</span></div>
              <div class="viewport"><canvas id="canvas"></canvas></div>
            </div>
            <div class="viewer-panel">
              <div class="split-title">
                <p class="caption">Segmentation Result</p>
                <p class="caption" id="resultCaption">Idle</p>
              </div>
              <div class="media-band"><strong>Mask Overlay</strong><span>Model output</span></div>
              <div class="viewport">
                <div id="resultEmpty" class="empty">No mask overlay</div>
                <div id="compareGrid" class="compare-grid" hidden></div>
                <img id="result" class="result-img" alt="" hidden>
              </div>
            </div>
          </div>
          <div class="run-strip">
            <div class="run-cell"><span class="run-label">CANVAS</span><span class="run-value" id="runCanvasState">No image</span></div>
            <div class="run-cell"><span class="run-label">PROMPTS</span><span class="run-value" id="runPromptState">0 active</span></div>
            <div class="run-cell"><span class="run-label">OUTPUT</span><span class="run-value" id="runOutputState">Idle</span></div>
            <div class="run-cell"><span class="run-label">MODE</span><span class="run-value" id="runModeState">Standard</span></div>
          </div>
        </section>
        <section class="panel controls inspector-panel">
          <div class="control-block">
            <p class="group-heading">Run Inspector</p>
            <p class="group-meta">Model, prompt, one-shot reference</p>
          </div>
          <div class="control-block">
            <label for="backend">Model</label>
            <select id="backend">
              <option value="sam">SAM ViT-B</option>
              <option value="sam2">SAM2.1 Tiny</option>
              <option value="sam3">SAM3</option>
              <option value="medsam">MedSAM ViT-B</option>
            </select>
          </div>
          <div class="control-block">
            <div class="control-title">
              <label>Prompt Controls</label>
              <button id="resetMode" class="mini-action secondary" type="button">Reset</button>
            </div>
            <div id="clickPanel">
              <label>Click Prompt</label>
              <div class="row">
                <button id="posClick" aria-pressed="true">Foreground</button>
                <button id="negClick" aria-pressed="false">Background</button>
              </div>
              <p class="muted">Foreground / background points</p>
            </div>
            <div id="textPanel" hidden>
              <label for="textPrompt">Text Concept</label>
              <input id="textPrompt" type="text" placeholder="e.g. child, truck, nuclei, lesion">
              <button id="runText" class="primary">Segment Text</button>
            </div>
          </div>
          <div class="control-block">
            <label for="mode">Mask Selection</label>
            <select id="mode">
              <option value="standard">Standard</option>
              <option value="conservative">Conservative</option>
            </select>
          </div>
          <div class="control-block">
            <div class="control-title">
              <label>Matcher One-Shot</label>
              <p id="supportState" class="support-state">No reference</p>
            </div>
            <div class="upload-stack">
              <label class="upload-card" for="matcherSupportFile">
                <span class="upload-title">Reference image</span>
                <span id="matcherSupportFileName" class="upload-meta">No image selected</span>
                <span class="upload-hint">Support example that contains the object or structure.</span>
              </label>
              <input id="matcherSupportFile" class="file-hidden" type="file" accept="image/*">
              <label class="upload-card" for="matcherSupportMaskFile">
                <span class="upload-title">Reference mask optional</span>
                <span id="matcherSupportMaskFileName" class="upload-meta">No mask selected</span>
                <span class="upload-hint">If empty, the reference box below becomes a rectangular mask.</span>
              </label>
              <input id="matcherSupportMaskFile" class="file-hidden" type="file" accept="image/*">
            </div>
            <input id="matcherSupportBox" type="text" placeholder="reference box: x0,y0,x1,y1">
            <div class="row">
              <select id="matcherVersion">
                <option value="1">v1 - multiple instances</option>
                <option value="2">v2 - whole object</option>
                <option value="3">v3 - object part</option>
              </select>
              <button id="runMatcher" class="primary">Run Matcher</button>
            </div>
            <p class="muted">v1 merges multiple matched instances; v2 favors a single whole object; v3 keeps part-level matches.</p>
          </div>
          <div class="control-block">
            <div class="row action-row">
              <button id="runCurrent" class="primary">Run</button>
              <button id="runCompare" class="secondary">Compare</button>
            </div>
          </div>
          <div class="control-block">
            <div class="row three">
              <button id="clearPrompts" class="secondary">Clear</button>
              <button id="clearResult" class="secondary">Reset Output</button>
              <button id="downloadResult" class="secondary">Download</button>
            </div>
          </div>
        </section>
      </main>
      <aside id="railDrawer" class="rail-drawer" hidden aria-label="Workbench drawer">
        <div class="drawer-head">
          <p class="group-heading" id="drawerTitle">Run</p>
          <button id="closeDrawer" class="mini-action secondary" type="button">Close</button>
        </div>
        <div class="drawer-body">
          <div id="runPanel" class="drawer-panel">
            <div class="drawer-section">
              <p class="group-heading">Current Run</p>
              <p class="drawer-note">Video prompts are staged on the first frame. Press Run to propagate through the clip.</p>
              <div class="kv-list">
                <div class="kv-row"><span class="kv-key">Media</span><span class="kv-value" id="drawerMedia">Image</span></div>
                <div class="kv-row"><span class="kv-key">Model</span><span class="kv-value" id="drawerModel">SAM ViT-B</span></div>
                <div class="kv-row"><span class="kv-key">Prompt</span><span class="kv-value" id="drawerPrompt">Click</span></div>
                <div class="kv-row"><span class="kv-key">Canvas</span><span class="kv-value" id="drawerCanvas">No image</span></div>
                <div class="kv-row"><span class="kv-key">Output</span><span class="kv-value" id="drawerOutput">Idle</span></div>
              </div>
              <div class="row action-row">
                <button id="drawerRunCurrent" class="primary" type="button">Run Current</button>
                <button id="drawerRunCompare" class="secondary" type="button">Compare</button>
              </div>
            </div>
            <div class="drawer-section">
              <p class="group-heading">Prompt Contract</p>
              <p class="drawer-note" id="drawerPromptHelp">Click or draw a box on the input canvas, then inspect the overlay on the right.</p>
            </div>
          </div>
          <div id="setPanel" class="drawer-panel" hidden>
            <div class="drawer-section">
              <p class="group-heading">Model Names</p>
              <p class="drawer-note"><strong>SAM ViT-B</strong> means the original SAM with a Vision Transformer Base image encoder. It is the lighter local SAM checkpoint used here for faster interaction.</p>
              <p class="drawer-note"><strong>SAM2.1 Tiny</strong> means the SAM2.1 Hiera Tiny checkpoint. Tiny is the backbone size, not a smaller output mask; it is chosen here because video propagation is much lighter to demo locally.</p>
            </div>
            <div class="drawer-section">
              <p class="group-heading">Prompt Support</p>
              <div class="cap-list">
                <div class="cap-item"><span class="cap-name">SAM ViT-B</span><p class="cap-text">Click and box are native SAM prompts. Text is implemented by selecting automatic-mask candidates in this workbench.</p></div>
                <div class="cap-item"><span class="cap-name">SAM2.1 Tiny</span><p class="cap-text">Click and box on images, plus first-frame click or box prompts for video propagation. Text is not native here.</p></div>
                <div class="cap-item"><span class="cap-name">SAM3</span><p class="cap-text">Text is the main SAM3 path in this demo; click and box are routed through the unified prompt contract when available.</p></div>
                <div class="cap-item"><span class="cap-name">MedSAM / Matcher</span><p class="cap-text">MedSAM is box-first for medical images. Matcher uses a support image plus mask or box for one-shot segmentation.</p></div>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </div>
  </div>
  <script>
    const fileInput = document.getElementById("file");
    const backendInput = document.getElementById("backend");
    const modeInput = document.getElementById("mode");
    const examples = document.getElementById("examples");
    const videos = document.getElementById("videos");
    const statusBox = document.getElementById("status");
    const canvas = document.getElementById("canvas");
    const result = document.getElementById("result");
    const resultEmpty = document.getElementById("resultEmpty");
    const compareGrid = document.getElementById("compareGrid");
    const inputCaption = document.getElementById("inputCaption");
    const canvasSize = document.getElementById("canvasSize");
    const resultCaption = document.getElementById("resultCaption");
    const mediaChip = document.getElementById("mediaChip");
    const modelChip = document.getElementById("modelChip");
    const promptChip = document.getElementById("promptChip");
    const supportChip = document.getElementById("supportChip");
    const runCanvasState = document.getElementById("runCanvasState");
    const runPromptState = document.getElementById("runPromptState");
    const runOutputState = document.getElementById("runOutputState");
    const runModeState = document.getElementById("runModeState");
    const supportState = document.getElementById("supportState");
    const textPrompt = document.getElementById("textPrompt");
    const runText = document.getElementById("runText");
    const matcherSupportFile = document.getElementById("matcherSupportFile");
    const matcherSupportMaskFile = document.getElementById("matcherSupportMaskFile");
    const matcherSupportFileName = document.getElementById("matcherSupportFileName");
    const matcherSupportMaskFileName = document.getElementById("matcherSupportMaskFileName");
    const matcherSupportBox = document.getElementById("matcherSupportBox");
    const matcherVersion = document.getElementById("matcherVersion");
    const runMatcher = document.getElementById("runMatcher");
    const clearPrompts = document.getElementById("clearPrompts");
    const clearResult = document.getElementById("clearResult");
    const runCurrent = document.getElementById("runCurrent");
    const runCompare = document.getElementById("runCompare");
    const downloadResult = document.getElementById("downloadResult");
    const resetMode = document.getElementById("resetMode");
    const clickPanel = document.getElementById("clickPanel");
    const textPanel = document.getElementById("textPanel");
    const posClick = document.getElementById("posClick");
    const negClick = document.getElementById("negClick");
    const railDrawer = document.getElementById("railDrawer");
    const drawerTitle = document.getElementById("drawerTitle");
    const runPanel = document.getElementById("runPanel");
    const setPanel = document.getElementById("setPanel");
    const closeDrawer = document.getElementById("closeDrawer");
    const drawerRunCurrent = document.getElementById("drawerRunCurrent");
    const drawerRunCompare = document.getElementById("drawerRunCompare");
    const drawerMedia = document.getElementById("drawerMedia");
    const drawerModel = document.getElementById("drawerModel");
    const drawerPrompt = document.getElementById("drawerPrompt");
    const drawerCanvas = document.getElementById("drawerCanvas");
    const drawerOutput = document.getElementById("drawerOutput");
    const drawerPromptHelp = document.getElementById("drawerPromptHelp");
    const ctx = canvas.getContext("2d");
    const sourceCanvas = document.createElement("canvas");
    const sourceCtx = sourceCanvas.getContext("2d");
    const modelLabels = {
      sam: "SAM ViT-B",
      sam2: "SAM2.1 Tiny",
      sam3: "SAM3",
      medsam: "MedSAM ViT-B"
    };
    const promptLabels = { click: "Click", box: "Box", text: "Text" };
    const modelCaps = {
      sam: { click: true, box: true, text: true },
      sam2: { click: true, box: true, text: false },
      sam3: { click: true, box: true, text: true },
      medsam: { click: false, box: true, text: false }
    };

    let loaded = false;
    let mediaKind = "image";
    let currentVideoId = null;
    let backend = "sam";
    let interaction = "click";
    let activePointLabel = 1;
    let baseImage = null;
    let points = [];
    let currentBox = null;
    let dragStart = null;
    let isDragging = false;
    let matcherSupportImage = "";
    let matcherSupportMask = "";
    let selectedExampleUrl = "";
    let selectedVideoId = null;

    function setStatus(text, kind = "info") {
      statusBox.textContent = text;
      statusBox.dataset.kind = kind;
    }

    function setBusy(isBusy) {
      document.body.dataset.busy = String(isBusy);
      resultCaption.textContent = isBusy ? "Running" : (result.hidden ? "Idle" : "Ready");
      runOutputState.textContent = isBusy ? "Running" : (result.hidden ? "Idle" : "Ready");
      updateCapabilityUI();
    }

    function isPromptAllowed(mode) {
      if (mediaKind === "video") return mode === "click" || mode === "box";
      return Boolean(modelCaps[backend]?.[mode]);
    }

    function firstAllowedPrompt() {
      return ["click", "box", "text"].find((mode) => isPromptAllowed(mode)) || "click";
    }

    function updateCapabilityUI() {
      const active = document.body.dataset.busy === "true";
      document.querySelectorAll("button, input, select").forEach((el) => {
        el.disabled = active && el.id !== "file";
      });
      mediaChip.textContent = mediaKind === "video" ? "Video" : "Image";
      mediaChip.className = mediaKind === "video" ? "metric-value state-good" : "metric-value";
      modelChip.textContent = modelLabels[backend] || backend;
      modelChip.className = backend === "sam3" ? "metric-value state-good" : "metric-value";
      promptChip.textContent = promptLabels[interaction] || interaction;
      promptChip.className = active ? "metric-value state-busy" : "metric-value";
      supportChip.textContent = matcherSupportImage ? "Support ready" : "No support";
      supportChip.className = matcherSupportImage ? "metric-value state-good" : "metric-value state-warn";
      supportState.textContent = matcherSupportImage
        ? (matcherSupportMask ? "Reference mask ready" : "Reference box mode")
        : "No reference";
      supportState.className = matcherSupportImage ? "support-state ready" : "support-state";
      document.querySelectorAll(".tab").forEach((button) => {
        const allowed = isPromptAllowed(button.dataset.mode);
        button.disabled = active || !allowed;
        button.title = allowed ? "" : "Not available for the current model/media.";
        button.setAttribute("aria-pressed", String(button.dataset.mode === interaction));
      });
      backendInput.disabled = active || mediaKind === "video";
      textPrompt.disabled = active || !isPromptAllowed("text");
      posClick.disabled = active || interaction !== "click";
      negClick.disabled = active || interaction !== "click";
      runText.disabled = active || !isPromptAllowed("text");
      modeInput.disabled = active || mediaKind === "video";
      matcherSupportFile.disabled = active;
      matcherSupportMaskFile.disabled = active;
      matcherSupportBox.disabled = active;
      matcherVersion.disabled = active;
      runMatcher.disabled = active || mediaKind !== "image";
      resetMode.disabled = active;
      clearPrompts.disabled = active;
      clearResult.disabled = active;
      runCurrent.disabled = active || !loaded;
      drawerRunCurrent.disabled = active || !loaded;
      runCompare.disabled = active || !loaded || mediaKind !== "image";
      drawerRunCompare.disabled = active || !loaded || mediaKind !== "image";
      downloadResult.disabled = active || result.hidden;
      runModeState.textContent = modeInput.value === "conservative" ? "Conservative" : "Standard";
      updatePromptSummary();
      syncDrawerState();
    }

    function updatePromptSummary() {
      if (!loaded) {
        runCanvasState.textContent = "No image";
        runPromptState.textContent = "0 active";
        syncDrawerState();
        return;
      }
      runCanvasState.textContent = `${canvas.width} x ${canvas.height}`;
      if (interaction === "click") {
        runPromptState.textContent = `${points.length} point${points.length === 1 ? "" : "s"}`;
      } else if (interaction === "box") {
        runPromptState.textContent = currentBox ? "1 box" : "0 boxes";
      } else {
        const text = textPrompt.value.trim();
        runPromptState.textContent = text ? "1 concept" : "0 concepts";
      }
      syncDrawerState();
    }

    function syncDrawerState() {
      if (!drawerMedia) return;
      drawerMedia.textContent = mediaKind === "video" ? "Video" : "Image";
      drawerModel.textContent = modelLabels[backend] || backend;
      drawerPrompt.textContent = `${promptLabels[interaction] || interaction}: ${runPromptState.textContent}`;
      drawerCanvas.textContent = runCanvasState.textContent;
      drawerOutput.textContent = runOutputState.textContent;
      if (mediaKind === "video") {
        drawerPromptHelp.textContent = "For video, place foreground/background points or draw one box on the first frame, then press Run to propagate masks through the sampled frames.";
      } else if (interaction === "text") {
        drawerPromptHelp.textContent = "Enter a text concept and press Run. SAM3 uses text directly; SAM uses automatic-mask candidate selection in this workbench.";
      } else if (interaction === "box") {
        drawerPromptHelp.textContent = "Draw a box around the target region. MedSAM is box-first, so this is the recommended medical prompt path.";
      } else {
        drawerPromptHelp.textContent = "Left click adds foreground points. Right click adds background points. Images run immediately; videos wait for Run.";
      }
    }

    function showResult(src, caption = "Ready") {
      result.src = src;
      result.hidden = false;
      compareGrid.hidden = true;
      resultEmpty.hidden = true;
      resultCaption.textContent = caption;
      runOutputState.textContent = caption;
      applyMediaDisplayScale();
    }

    function clearResultView() {
      result.removeAttribute("src");
      result.hidden = true;
      compareGrid.hidden = true;
      compareGrid.innerHTML = "";
      resultEmpty.hidden = false;
      resultCaption.textContent = "Idle";
      runOutputState.textContent = "Idle";
    }

    function showCompareResults(items) {
      result.hidden = true;
      resultEmpty.hidden = true;
      compareGrid.hidden = false;
      compareGrid.innerHTML = "";
      items.forEach((item) => {
        const card = document.createElement("div");
        card.className = "compare-card";
        const title = document.createElement("div");
        title.className = "compare-title";
        title.textContent = item.model.toUpperCase();
        card.appendChild(title);
        if (item.ok && item.overlay) {
          const img = document.createElement("img");
          img.src = item.overlay;
          img.alt = item.model;
          card.appendChild(img);
        } else {
          const err = document.createElement("div");
          err.className = "compare-error";
          err.textContent = item.error || "Not available";
          card.appendChild(err);
        }
        compareGrid.appendChild(card);
      });
      resultCaption.textContent = "Compare ready";
      runOutputState.textContent = "Compare ready";
    }

    function markSelected(container, predicate) {
      container.querySelectorAll("button").forEach((button) => {
        button.setAttribute("aria-pressed", String(predicate(button)));
      });
    }

    function displaySizeFor(viewport, width, height) {
      if (!viewport || !width || !height) return null;
      const availableWidth = Math.max(120, viewport.clientWidth - 24);
      const availableHeight = Math.max(120, viewport.clientHeight - 24);
      const scale = Math.min(availableWidth / width, availableHeight / height);
      return {
        width: Math.max(1, Math.round(width * scale)),
        height: Math.max(1, Math.round(height * scale))
      };
    }

    function applyMediaDisplayScale() {
      if (!canvas.width || !canvas.height) return;
      const inputSize = displaySizeFor(canvas.parentElement, canvas.width, canvas.height);
      if (inputSize) {
        canvas.style.width = `${inputSize.width}px`;
        canvas.style.height = `${inputSize.height}px`;
      }
      const resultSize = displaySizeFor(result.parentElement, canvas.width, canvas.height);
      if (resultSize) {
        result.style.width = `${resultSize.width}px`;
        result.style.height = `${resultSize.height}px`;
      }
    }

    function drawBaseImage() {
      if (!baseImage) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(baseImage, 0, 0, canvas.width, canvas.height);
    }

    function drawPrompts() {
      drawBaseImage();
      ctx.save();
      points.forEach((pt) => {
        ctx.strokeStyle = pt.label ? "#ffd84d" : "#ffffff";
        ctx.fillStyle = pt.label ? "#25c46a" : "#ff5050";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3.5, 0, Math.PI * 2);
        ctx.fill();
      });
      if (currentBox) {
        ctx.strokeStyle = "#4db4ff";
        ctx.lineWidth = 3;
        ctx.strokeRect(
          currentBox.x0,
          currentBox.y0,
          currentBox.x1 - currentBox.x0,
          currentBox.y1 - currentBox.y0
        );
      }
      ctx.restore();
      updatePromptSummary();
    }

    function setInteraction(nextMode, options = {}) {
      const previousMode = interaction;
      if (!isPromptAllowed(nextMode)) nextMode = firstAllowedPrompt();
      interaction = nextMode;
      clickPanel.hidden = nextMode !== "click";
      textPanel.hidden = nextMode !== "text";
      canvas.style.cursor = nextMode === "text" ? "default" : "crosshair";
      if (options.preserve && previousMode === nextMode) {
        drawPrompts();
      } else {
        clearPromptState();
      }
      const hint = mediaKind === "video"
        ? "SAM2 video mode is ready for a first-frame prompt."
        : backend === "medsam"
        ? "MedSAM is ready for a box prompt."
        : backend === "sam3"
        ? `SAM3 ${promptLabels[nextMode]} mode is ready.`
        : {
            click: "Click mode is ready.",
            box: "Box mode is ready.",
            text: "Text mode is ready."
          }[nextMode];
      updateCapabilityUI();
      setStatus(loaded ? hint : "Waiting for an image.", loaded ? "ok" : "info");
    }

    function setBackend(nextBackend) {
      backend = nextBackend;
      if (backend === "sam3") {
        textPrompt.placeholder = "e.g. child, truck, cell nucleus, lesion";
        modeInput.value = "standard";
        if (loaded) {
          setStatus("SAM3 ready. The first request may take 5-15 seconds while the model initializes.", "ok");
        }
      } else if (backend === "sam2") {
        textPrompt.placeholder = "SAM2 image text is not native in this demo; use click or box.";
        if (interaction === "text") setInteraction("click");
      } else if (backend === "medsam") {
        textPrompt.placeholder = "MedSAM uses box prompts in this workbench.";
        setInteraction("box", { preserve: interaction === "box" });
      } else {
        textPrompt.placeholder = "e.g. largest object, left lesion, yellow optic disc, nuclei";
      }
      setInteraction(interaction, { preserve: true });
    }

    function setActivePoint(label) {
      activePointLabel = label;
      posClick.setAttribute("aria-pressed", String(label === 1));
      negClick.setAttribute("aria-pressed", String(label === 0));
    }

    function clearPromptState() {
      points = [];
      currentBox = null;
      dragStart = null;
      isDragging = false;
      drawPrompts();
    }

    function clearAll() {
      clearPromptState();
      clearResultView();
    }

    function imageToPayload() {
      return sourceCanvas.toDataURL("image/png");
    }

    function canvasXY(event) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: Math.round((event.clientX - rect.left) * canvas.width / rect.width),
        y: Math.round((event.clientY - rect.top) * canvas.height / rect.height)
      };
    }

    function loadImageFromUrl(url) {
      const img = new Image();
      img.onload = () => {
        mediaKind = "image";
        currentVideoId = null;
        selectedVideoId = null;
        selectedExampleUrl = url;
        inputCaption.textContent = "Input Image";
        const maxSide = 1024;
        const scale = Math.min(1, maxSide / Math.max(img.width, img.height));
        canvas.width = Math.round(img.width * scale);
        canvas.height = Math.round(img.height * scale);
        canvasSize.textContent = `${canvas.width} x ${canvas.height}`;
        runCanvasState.textContent = `${canvas.width} x ${canvas.height}`;
        sourceCanvas.width = canvas.width;
        sourceCanvas.height = canvas.height;
        baseImage = img;
        sourceCtx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
        sourceCtx.drawImage(img, 0, 0, sourceCanvas.width, sourceCanvas.height);
        loaded = true;
        clearAll();
        markSelected(examples, (button) => button.dataset.url === selectedExampleUrl);
        markSelected(videos, () => false);
        setInteraction(interaction);
        requestAnimationFrame(applyMediaDisplayScale);
        setStatus(`Image loaded: ${canvas.width} x ${canvas.height}`, "ok");
      };
      img.onerror = () => setStatus("Failed to load image.", "error");
      img.src = url;
    }

    async function loadVideoExample(item) {
      const response = await fetch(`/video_preview/${item.id}`);
      const payload = await response.json();
      if (!response.ok) {
        setStatus(payload.detail || "Failed to load video preview.", "error");
        return;
      }
      const img = new Image();
      img.onload = () => {
        mediaKind = "video";
        currentVideoId = item.id;
        selectedVideoId = item.id;
        selectedExampleUrl = "";
        backendInput.value = "sam2";
        backend = "sam2";
        inputCaption.textContent = `Video First Frame: ${item.name}`;
        const maxSide = 1024;
        const scale = Math.min(1, maxSide / Math.max(img.width, img.height));
        canvas.width = Math.round(img.width * scale);
        canvas.height = Math.round(img.height * scale);
        canvasSize.textContent = `${canvas.width} x ${canvas.height}`;
        runCanvasState.textContent = `${canvas.width} x ${canvas.height}`;
        sourceCanvas.width = canvas.width;
        sourceCanvas.height = canvas.height;
        baseImage = img;
        sourceCtx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
        sourceCtx.drawImage(img, 0, 0, sourceCanvas.width, sourceCanvas.height);
        loaded = true;
        clearAll();
        if (interaction === "text") interaction = "click";
        setInteraction(interaction);
        markSelected(examples, () => false);
        markSelected(videos, (button) => Number(button.dataset.videoId) === selectedVideoId);
        requestAnimationFrame(applyMediaDisplayScale);
        setStatus(`Video loaded: ${item.name}`, "ok");
      };
      img.onerror = () => setStatus("Failed to load video preview.", "error");
      img.src = payload.preview;
    }

    async function runSegmentation(payload, statusText) {
      if (!loaded) {
        setStatus("Please upload or choose an image first.", "warn");
        return;
      }
      setBusy(true);
      setStatus(statusText, "busy");
      try {
        const isVideo = mediaKind === "video";
        const requestPayload = isVideo
          ? {
              video_id: currentVideoId,
              prompt_type: payload.prompt_type,
              points: payload.points || [],
              box: payload.box || null,
              max_frames: 36
            }
          : payload;
        const response = await fetch(isVideo ? "/video_segment" : "/segment", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestPayload)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Segmentation failed");
        showResult(isVideo ? data.animation : data.overlay, isVideo ? "GIF ready" : "Mask ready");
        setStatus(data.info, "ok");
      } catch (error) {
        setStatus(`Segmentation failed: ${error.message}`, "error");
      } finally {
        setBusy(false);
      }
    }

    function currentImagePromptPayload() {
      if (interaction === "click") {
        if (!points.length) throw new Error("Add at least one point before comparing.");
        return { prompt_type: "click", points };
      }
      if (interaction === "box") {
        if (!currentBox) throw new Error("Draw a box before comparing.");
        return { prompt_type: "box", box: [currentBox.x0, currentBox.y0, currentBox.x1, currentBox.y1] };
      }
      const text = textPrompt.value.trim();
      if (!text) throw new Error("Enter a text prompt before comparing.");
      return { prompt_type: "text", text };
    }

    async function runModelCompare() {
      if (!loaded || mediaKind !== "image") {
        setStatus("Compare mode needs a loaded image.", "warn");
        return;
      }
      let promptPayload;
      try {
        promptPayload = currentImagePromptPayload();
      } catch (error) {
        setStatus(error.message, "warn");
        return;
      }
      setBusy(true);
      setStatus("Running compatible models with the same prompt ...", "busy");
      try {
        const response = await fetch("/compare_segment", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: imageToPayload(),
            models: ["sam", "sam2", "sam3", "medsam"],
            ...promptPayload,
            mode: modeInput.value
          })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Compare failed");
        showCompareResults(data.results || []);
        setStatus("Compare finished. Incompatible models are shown as error cards.", "ok");
      } catch (error) {
        setStatus(`Compare failed: ${error.message}`, "error");
      } finally {
        setBusy(false);
      }
    }

    async function runClick() {
      if (!points.length) return;
      drawPrompts();
      await runSegmentation({
        image: imageToPayload(),
        backend,
        prompt_type: "click",
        points,
        mode: modeInput.value
      }, `Segmenting ${points.length} click prompt(s) ...`);
    }

    async function runBox() {
      if (!currentBox) return;
      drawPrompts();
      await runSegmentation({
        image: imageToPayload(),
        backend,
        prompt_type: "box",
        box: [currentBox.x0, currentBox.y0, currentBox.x1, currentBox.y1],
        mode: modeInput.value
      }, `Segmenting box [${currentBox.x0}, ${currentBox.y0}, ${currentBox.x1}, ${currentBox.y1}] ...`);
    }

    async function runTextPrompt() {
      if (mediaKind === "video") {
        setStatus("SAM2 video mode supports first-frame click or box prompts in this demo.", "warn");
        return;
      }
      if (backend === "sam2") {
        setStatus("SAM2 image text prompts are not native in this demo. Use click or box for SAM2.", "warn");
        return;
      }
      if (backend === "medsam") {
        setStatus("MedSAM uses box prompts in this workbench.", "warn");
        return;
      }
      const text = textPrompt.value.trim();
      if (!text) {
        setStatus("Please enter a text prompt.", "warn");
        return;
      }
      clearPromptState();
      await runSegmentation({
        image: imageToPayload(),
        backend,
        prompt_type: "text",
        text,
        mode: modeInput.value
      }, backend === "sam3"
        ? `Loading SAM3 and segmenting text prompt: ${text} ...`
        : `Segmenting text prompt: ${text} ...`);
    }

    async function runCurrentPrompt() {
      if (!loaded) {
        setStatus("Please upload or choose media first.", "warn");
        return;
      }
      if (interaction === "click") {
        if (!points.length) {
          setStatus(mediaKind === "video" ? "Add at least one first-frame point, then press Run." : "Add at least one point.", "warn");
          return;
        }
        await runClick();
        return;
      }
      if (interaction === "box") {
        if (!currentBox) {
          setStatus(mediaKind === "video" ? "Draw a first-frame box, then press Run." : "Draw a box first.", "warn");
          return;
        }
        await runBox();
        return;
      }
      await runTextPrompt();
    }

    function openRailPanel(panelName) {
      document.querySelectorAll(".rail-item[data-panel]").forEach((button) => {
        button.classList.toggle("active", button.dataset.panel === panelName);
      });
      if (panelName === "lab") {
        railDrawer.hidden = true;
        return;
      }
      railDrawer.hidden = false;
      runPanel.hidden = panelName !== "run";
      setPanel.hidden = panelName !== "set";
      drawerTitle.textContent = panelName === "run" ? "Run" : "Set";
      syncDrawerState();
    }

    function parseBoxText(text) {
      const values = text.split(",").map((v) => Number(v.trim()));
      if (values.length !== 4 || values.some((v) => !Number.isFinite(v))) {
        throw new Error("Reference box must be x0,y0,x1,y1");
      }
      if (values[2] <= values[0] || values[3] <= values[1]) {
        throw new Error("Reference box needs x1>x0 and y1>y0");
      }
      return values;
    }

    async function runMatcherPrompt() {
      if (!loaded || mediaKind !== "image") {
        setStatus("Matcher needs a target image loaded in the input canvas.", "warn");
        return;
      }
      if (!matcherSupportImage) {
        setStatus("Please upload a Matcher reference image first.", "warn");
        return;
      }
      let supportBox;
      try {
        supportBox = matcherSupportMask ? [0, 0, 1, 1] : parseBoxText(matcherSupportBox.value);
      } catch (error) {
        setStatus(error.message, "warn");
        return;
      }
      setBusy(true);
      setStatus("Running Matcher one-shot segmentation. This can take a while because Matcher loads SAM-H and DINOv2.", "busy");
      try {
        const response = await fetch("/matcher_segment", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: imageToPayload(),
            support_image: matcherSupportImage,
            support_mask: matcherSupportMask,
            support_box: supportBox,
            version: Number(matcherVersion.value)
          })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Matcher failed");
        showResult(data.overlay, "Matcher ready");
        setStatus(data.info, "ok");
      } catch (error) {
        setStatus(`Matcher failed: ${error.message}`, "error");
      } finally {
        setBusy(false);
      }
    }

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => loadImageFromUrl(reader.result);
      reader.readAsDataURL(file);
    });

    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("mousedown", (event) => {
      if (!loaded || interaction === "text") return;
      const { x, y } = canvasXY(event);
      if (interaction === "click") {
        const label = event.button === 2 ? 0 : activePointLabel;
        points.push({ x, y, label });
        currentBox = null;
        drawPrompts();
        if (mediaKind === "video") {
          setStatus(`${points.length} first-frame point${points.length === 1 ? "" : "s"} ready. Press Run to propagate video masks.`, "ok");
          return;
        }
        runClick();
        return;
      }
      if (interaction === "box") {
        points = [];
        dragStart = { x, y };
        currentBox = { x0: x, y0: y, x1: x, y1: y };
        isDragging = true;
        drawPrompts();
      }
    });
    canvas.addEventListener("mousemove", (event) => {
      if (!isDragging || interaction !== "box" || !dragStart) return;
      const { x, y } = canvasXY(event);
      currentBox = {
        x0: Math.min(dragStart.x, x),
        y0: Math.min(dragStart.y, y),
        x1: Math.max(dragStart.x, x),
        y1: Math.max(dragStart.y, y)
      };
      drawPrompts();
    });
    canvas.addEventListener("mouseup", () => {
      if (!isDragging || interaction !== "box") return;
      isDragging = false;
      const valid = currentBox && Math.abs(currentBox.x1 - currentBox.x0) >= 4 && Math.abs(currentBox.y1 - currentBox.y0) >= 4;
      if (valid) {
        drawPrompts();
        if (mediaKind === "video") {
          setStatus("First-frame box ready. Press Run to propagate video masks.", "ok");
          return;
        }
        runBox();
      }
    });

    document.querySelectorAll(".tab").forEach((button) => {
      button.addEventListener("click", () => setInteraction(button.dataset.mode));
    });
    document.querySelectorAll(".rail-item[data-panel]").forEach((button) => {
      button.addEventListener("click", () => openRailPanel(button.dataset.panel));
    });
    closeDrawer.addEventListener("click", () => openRailPanel("lab"));
    backendInput.addEventListener("change", () => setBackend(backendInput.value));
    modeInput.addEventListener("change", updateCapabilityUI);
    resetMode.addEventListener("click", () => setInteraction(firstAllowedPrompt()));
    posClick.addEventListener("click", () => setActivePoint(1));
    negClick.addEventListener("click", () => setActivePoint(0));
    runCurrent.addEventListener("click", runCurrentPrompt);
    drawerRunCurrent.addEventListener("click", runCurrentPrompt);
    runText.addEventListener("click", runTextPrompt);
    textPrompt.addEventListener("keydown", (event) => {
      if (event.key === "Enter") runTextPrompt();
    });
    textPrompt.addEventListener("input", updatePromptSummary);
    matcherSupportFile.addEventListener("change", () => {
      const file = matcherSupportFile.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        matcherSupportImage = reader.result;
        matcherSupportFileName.textContent = file.name;
        updateCapabilityUI();
        setStatus("Matcher reference image loaded. Add a reference mask or enter its reference box.", "ok");
      };
      reader.readAsDataURL(file);
    });
    matcherSupportMaskFile.addEventListener("change", () => {
      const file = matcherSupportMaskFile.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        matcherSupportMask = reader.result;
        matcherSupportMaskFileName.textContent = file.name;
        updateCapabilityUI();
        setStatus("Matcher reference mask loaded.", "ok");
      };
      reader.readAsDataURL(file);
    });
    runMatcher.addEventListener("click", runMatcherPrompt);
    runCompare.addEventListener("click", runModelCompare);
    drawerRunCompare.addEventListener("click", runModelCompare);
    clearPrompts.addEventListener("click", () => {
      clearPromptState();
      setStatus(loaded ? "Prompts cleared." : "Waiting for an image.", loaded ? "ok" : "info");
    });
    clearResult.addEventListener("click", () => {
      clearResultView();
      setStatus(loaded ? "Result cleared." : "Waiting for an image.", loaded ? "ok" : "info");
      updateCapabilityUI();
    });
    downloadResult.addEventListener("click", () => {
      if (result.hidden || !result.src) {
        setStatus("No result to download yet.", "warn");
        return;
      }
      const link = document.createElement("a");
      link.href = result.src;
      link.download = mediaKind === "video" ? "ocsam_sam2_result.gif" : `ocsam_${backend}_${interaction}_overlay.png`;
      document.body.appendChild(link);
      link.click();
      link.remove();
    });
    window.addEventListener("resize", applyMediaDisplayScale);

    async function loadExamples() {
      const response = await fetch("/examples");
      const payload = await response.json();
      examples.innerHTML = "";
      payload.examples.forEach((item) => {
        const button = document.createElement("button");
        button.className = "example-btn";
        button.dataset.url = item.url;
        button.setAttribute("aria-pressed", "false");
        button.textContent = `${item.dataset || "Sample"} · ${item.difficulty || "mixed"} · ${item.name}`;
        button.addEventListener("click", () => loadImageFromUrl(item.url));
        examples.appendChild(button);
      });
      markSelected(examples, (button) => button.dataset.url === selectedExampleUrl);
    }

    async function loadVideos() {
      const response = await fetch("/videos");
      const payload = await response.json();
      videos.innerHTML = "";
      payload.videos.forEach((item) => {
        const button = document.createElement("button");
        button.className = "example-btn";
        button.dataset.videoId = String(item.id);
        button.setAttribute("aria-pressed", "false");
        button.textContent = `${item.dataset || "Video"} · ${item.difficulty || "mixed"} · ${item.name} (${item.kind})`;
        button.addEventListener("click", () => loadVideoExample(item));
        videos.appendChild(button);
      });
      markSelected(videos, (button) => Number(button.dataset.videoId) === selectedVideoId);
    }

    updateCapabilityUI();
    clearResultView();
    loadExamples();
    loadVideos();
  </script>
</body>
</html>
"""


class PointPrompt(BaseModel):
    x: int
    y: int
    label: int = 1


class SegmentRequest(BaseModel):
    image: str
    backend: Literal["sam", "sam2", "sam3", "medsam"] = "sam"
    prompt_type: Literal["click", "box", "text"] = "click"
    points: list[PointPrompt] = []
    box: list[float] | None = None
    text: str = ""
    mode: str = "standard"


class MatcherRequest(BaseModel):
    image: str
    support_image: str
    support_mask: str = ""
    support_box: list[float]
    version: int = 1


class VideoSegmentRequest(BaseModel):
    video_id: int
    prompt_type: Literal["click", "box"] = "click"
    points: list[PointPrompt] = []
    box: list[float] | None = None
    max_frames: int = 36


class CompareRequest(BaseModel):
    image: str
    models: list[Literal["sam", "sam2", "sam3", "medsam"]] = ["sam", "sam2", "sam3", "medsam"]
    prompt_type: Literal["click", "box", "text"] = "click"
    points: list[PointPrompt] = []
    box: list[float] | None = None
    text: str = ""
    mode: str = "standard"


@dataclass
class SegmentResult:
    overlay: np.ndarray
    info: str
    masks: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    metadata: dict[str, Any]

    def prediction_payload(self) -> dict[str, Any]:
        masks_bool = np.asarray(self.masks).astype(bool)
        if masks_bool.ndim == 2:
            masks_bool = masks_bool[None, ...]
        boxes = np.asarray(self.boxes, dtype=np.float32)
        scores = np.asarray(self.scores, dtype=np.float32)
        return {
            "masks": [
                {
                    "encoding": "png",
                    "data": encode_mask_png(mask),
                    "shape": [int(mask.shape[0]), int(mask.shape[1])],
                    "area": int(mask.sum()),
                }
                for mask in masks_bool
            ],
            "boxes_xyxy": boxes.astype(float).round(2).tolist() if boxes.size else [],
            "scores": [float(x) for x in scores.tolist()] if scores.size else [],
            "metadata": self.metadata,
        }


@dataclass
class TextIntent:
    prefer_large: bool = False
    prefer_small: bool = False
    prefer_center: bool = False
    prefer_left: bool = False
    prefer_right: bool = False
    prefer_top: bool = False
    prefer_bottom: bool = False
    prefer_dark: bool = False
    prefer_bright: bool = False
    target_color: np.ndarray | None = None
    medical: str | None = None


class PromptSamServer:
    def __init__(
        self,
        checkpoint: Path,
        model_type: str,
        device: str,
        amg_points_per_side: int,
        sam3_checkpoint: Path | None = None,
        sam3_threshold: float = 0.5,
        sam2_checkpoint: Path | None = None,
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        medsam_checkpoint: Path | None = None,
        matcher_enabled: bool = True,
    ) -> None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        self.device = torch.device(device if device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device=self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)
        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=amg_points_per_side,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.88,
            crop_n_layers=0,
            min_mask_region_area=64,
        )
        self.cached_hash: str | None = None
        self.cached_auto_hash: str | None = None
        self.cached_auto_masks: list[dict[str, Any]] = []
        self.sam3_checkpoint = sam3_checkpoint
        self.sam3_threshold = sam3_threshold
        self.sam3_segmenter: Sam3TextSegmenter | None = None
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.sam2_image_segmenter: Sam2ImageSegmenter | None = None
        self.sam2_video_segmenter: Sam2VideoSegmenter | None = None
        self.medsam_checkpoint = medsam_checkpoint
        self.medsam_segmenter: MedSamBoxSegmenter | None = None
        self.matcher_enabled = matcher_enabled
        self.matcher_segmenter: MatcherOneShotSegmenter | None = None

    def warmup_sam3(self) -> None:
        if self.sam3_checkpoint is None:
            return
        if not self.sam3_checkpoint.exists():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {self.sam3_checkpoint}")
        if self.sam3_segmenter is None:
            self.sam3_segmenter = Sam3TextSegmenter(
                checkpoint=self.sam3_checkpoint,
                device=str(self.device),
                threshold=self.sam3_threshold,
            )

    def segment(self, req: SegmentRequest, image: np.ndarray) -> SegmentResult:
        if req.backend == "medsam":
            return self.segment_medsam(req, image)

        if req.backend == "sam2":
            return self.segment_sam2_image(req, image)

        if req.backend == "sam3":
            return self.segment_sam3(req, image)

        if req.prompt_type == "click":
            mask, prompt_info, marker = self.segment_click(image, req.points, req.mode)
        elif req.prompt_type == "box":
            mask, prompt_info, marker = self.segment_box(image, req.box)
        elif req.prompt_type == "text":
            mask, prompt_info, marker = self.segment_text(image, req.text, req.mode)
        else:
            raise ValueError(f"unsupported prompt type: {req.prompt_type}")

        overlay = make_overlay(image, mask, marker)
        area_pct = 100.0 * float(mask.sum()) / float(mask.size)
        info = f"{prompt_info} | Mask area: {area_pct:.2f}% | Device: {self.device}"
        box = masks_to_xyxy(mask[None, ...])
        score = parse_score_from_info(prompt_info)
        return SegmentResult(
            overlay=overlay,
            info=info,
            masks=mask[None, ...],
            boxes=box,
            scores=np.asarray([score], dtype=np.float32),
            metadata={
                "backend": "sam",
                "prompt_type": req.prompt_type,
                "mode": req.mode,
                "image_shape": list(image.shape),
                "native_text": False,
            },
        )

    def warmup_sam2_image(self) -> None:
        if self.sam2_checkpoint is None:
            return
        if not self.sam2_checkpoint.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.sam2_checkpoint}")
        if self.sam2_image_segmenter is None:
            self.sam2_image_segmenter = Sam2ImageSegmenter(
                checkpoint=self.sam2_checkpoint,
                config=self.sam2_config,
                device=str(self.device),
            )

    def warmup_sam2_video(self) -> None:
        if self.sam2_checkpoint is None:
            return
        if not self.sam2_checkpoint.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.sam2_checkpoint}")
        if self.sam2_video_segmenter is None:
            self.sam2_video_segmenter = Sam2VideoSegmenter(
                checkpoint=self.sam2_checkpoint,
                config=self.sam2_config,
                device=str(self.device),
            )

    def warmup_medsam(self) -> None:
        if self.medsam_checkpoint is None:
            return
        if not self.medsam_checkpoint.exists():
            raise FileNotFoundError(
                f"MedSAM checkpoint not found: {self.medsam_checkpoint}. "
                "Put medsam_vit_b.pth at D:\\SAM\\assets\\checkpoints\\medsam_vit_b.pth."
            )
        if self.medsam_segmenter is None:
            self.medsam_segmenter = MedSamBoxSegmenter(
                checkpoint=self.medsam_checkpoint,
                device=str(self.device),
            )

    def warmup_matcher(self) -> None:
        if not self.matcher_enabled:
            return
        if self.matcher_segmenter is None:
            self.matcher_segmenter = MatcherOneShotSegmenter(device=str(self.device))

    def segment_sam2_image(self, req: SegmentRequest, image: np.ndarray) -> SegmentResult:
        self.warmup_sam2_image()
        if self.sam2_image_segmenter is None:
            raise ValueError("SAM2 checkpoint path was not configured")
        masks, scores = self.sam2_image_segmenter.predict(
            image=image,
            prompt_type=req.prompt_type,
            points=req.points,
            box=req.box,
            mode=req.mode,
        )
        best = choose_sam_mask(masks, scores, prefer_small=(req.mode == "conservative"))
        mask = masks[best].astype(bool)
        overlay = make_overlay(image, mask, None)
        area_pct = 100.0 * float(mask.sum()) / float(mask.size)
        info = (
            f"SAM2 image {req.prompt_type} prompt | "
            f"SAM2 score: {float(scores[best]):.3f} | Mask area: {area_pct:.2f}% | Device: {self.device}"
        )
        return SegmentResult(
            overlay=overlay,
            info=info,
            masks=mask[None, ...],
            boxes=masks_to_xyxy(mask[None, ...]),
            scores=np.asarray([float(scores[best])], dtype=np.float32),
            metadata={
                "backend": "sam2",
                "prompt_type": req.prompt_type,
                "mode": req.mode,
                "image_shape": list(image.shape),
            },
        )

    def segment_medsam(self, req: SegmentRequest, image: np.ndarray) -> SegmentResult:
        if req.prompt_type != "box":
            raise ValueError("MedSAM in this workbench uses box prompts. Switch interaction to Box.")
        self.warmup_medsam()
        if self.medsam_segmenter is None:
            raise ValueError("MedSAM checkpoint path was not configured")
        mask = self.medsam_segmenter.predict(image, req.box)
        overlay = make_overlay(image, mask.astype(bool), None)
        area_pct = 100.0 * float(mask.sum()) / float(mask.size)
        info = f"MedSAM box prompt | Mask area: {area_pct:.2f}% | Device: {self.device}"
        mask_bool = mask.astype(bool)
        return SegmentResult(
            overlay=overlay,
            info=info,
            masks=mask_bool[None, ...],
            boxes=masks_to_xyxy(mask_bool[None, ...]),
            scores=np.asarray([1.0], dtype=np.float32),
            metadata={
                "backend": "medsam",
                "prompt_type": "box",
                "mode": req.mode,
                "image_shape": list(image.shape),
            },
        )

    def segment_sam3(self, req: SegmentRequest, image: np.ndarray) -> SegmentResult:
        if req.prompt_type == "text":
            return self.segment_sam3_text(image, req.text)
        self.warmup_sam3()
        if self.sam3_segmenter is None:
            raise ValueError("SAM3 checkpoint path was not configured")
        masks, scores = self.sam3_segmenter.predict_interactive(
            image=image,
            prompt_type=req.prompt_type,
            points=req.points,
            box=req.box,
            mode=req.mode,
        )
        if len(masks) == 0:
            raise ValueError("SAM3 returned no mask for this interactive prompt")
        best = choose_sam_mask(masks, scores, prefer_small=(req.mode == "conservative"))
        mask = masks[best].astype(bool)
        overlay = make_overlay(image, mask, None)
        area_pct = 100.0 * float(mask.sum()) / float(mask.size)
        info = (
            f"SAM3 interactive {req.prompt_type} prompt | "
            f"SAM3 score: {float(scores[best]):.3f} | Mask area: {area_pct:.2f}% | Device: {self.device}"
        )
        return SegmentResult(
            overlay=overlay,
            info=info,
            masks=mask[None, ...],
            boxes=masks_to_xyxy(mask[None, ...]),
            scores=np.asarray([float(scores[best])], dtype=np.float32),
            metadata={
                "backend": "sam3",
                "prompt_type": req.prompt_type,
                "mode": req.mode,
                "image_shape": list(image.shape),
                "native_text": False,
            },
        )

    def segment_sam2_video(self, req: VideoSegmentRequest, video_path: Path) -> tuple[list[np.ndarray], str]:
        self.warmup_sam2_video()
        if self.sam2_video_segmenter is None:
            raise ValueError("SAM2 checkpoint path was not configured")
        return self.sam2_video_segmenter.segment_video(
            video_path=video_path,
            prompt_type=req.prompt_type,
            points=req.points,
            box=req.box,
            max_frames=req.max_frames,
        )

    def segment_matcher(self, req: MatcherRequest) -> SegmentResult:
        self.warmup_matcher()
        if self.matcher_segmenter is None:
            raise ValueError("Matcher backend was not configured")
        target = decode_data_url(req.image)
        support = decode_data_url(req.support_image)
        support_mask = decode_data_url(req.support_mask)[:, :, 0] > 0 if req.support_mask else None
        mask = self.matcher_segmenter.predict(
            support_image=support,
            support_mask=support_mask,
            support_box=req.support_box,
            target_image=target,
            version=req.version,
        )
        overlay = make_overlay(target, mask.astype(bool), None)
        area_pct = 100.0 * float(mask.sum()) / float(mask.size)
        info = f"Matcher one-shot segmentation | Version: {req.version} | Mask area: {area_pct:.2f}% | Device: {self.device}"
        mask_bool = mask.astype(bool)
        return SegmentResult(
            overlay=overlay,
            info=info,
            masks=mask_bool[None, ...],
            boxes=masks_to_xyxy(mask_bool[None, ...]),
            scores=np.asarray([1.0], dtype=np.float32),
            metadata={
                "backend": "matcher",
                "prompt_type": "reference",
                "support_box": req.support_box,
                "support_mask": bool(req.support_mask),
                "version": req.version,
                "image_shape": list(target.shape),
            },
        )

    def segment_sam3_text(self, image: np.ndarray, text: str) -> SegmentResult:
        text = text.strip()
        if not text:
            raise ValueError("SAM3 text mode requires a prompt")
        if self.sam3_checkpoint is None:
            raise ValueError("SAM3 checkpoint path was not configured")
        self.warmup_sam3()
        masks, boxes, scores = self.sam3_segmenter.predict(image, text)
        overlay = make_multi_overlay(image, masks, boxes, scores)
        total_area = int(masks.any(axis=0).sum()) if len(masks) else 0
        area_pct = 100.0 * float(total_area) / float(image.shape[0] * image.shape[1])
        if len(scores):
            score_txt = ", ".join(f"{float(s):.2f}" for s in scores[:8])
        else:
            score_txt = "none"
        info = (
            f'SAM3 text prompt: "{text}" | Instances: {len(masks)} | '
            f"Scores: {score_txt} | Union area: {area_pct:.2f}% | Device: {self.device}"
        )
        return SegmentResult(
            overlay=overlay,
            info=info,
            masks=masks.astype(bool),
            boxes=boxes,
            scores=scores,
            metadata={
                "backend": "sam3",
                "prompt_type": "text",
                "text": text,
                "image_shape": list(image.shape),
                "native_text": True,
            },
        )

    def ensure_predictor_image(self, image: np.ndarray) -> None:
        digest = hashlib.sha1(image.tobytes()).hexdigest()
        if digest != self.cached_hash:
            self.predictor.set_image(image)
            self.cached_hash = digest

    def segment_click(
        self,
        image: np.ndarray,
        points: list[PointPrompt],
        mode: str,
    ) -> tuple[np.ndarray, str, tuple[int, int] | list[int] | None]:
        if not points:
            raise ValueError("click mode requires at least one point")
        self.ensure_predictor_image(image)
        h, w = image.shape[:2]
        coords = []
        labels = []
        for point in points:
            x = max(0, min(w - 1, int(point.x)))
            y = max(0, min(h - 1, int(point.y)))
            coords.append([x, y])
            labels.append(1 if int(point.label) else 0)

        if mode == "conservative" and len(coords) == 1 and labels[0] == 1:
            x, y = coords[0]
            offset = max(12, min(h, w) // 40)
            coords.extend([[max(0, x - offset), y], [min(w - 1, x + offset), y], [x, max(0, y - offset)]])
            labels.extend([0, 0, 0])

        masks, scores, _ = self.predictor.predict(
            point_coords=np.asarray(coords, dtype=np.float32),
            point_labels=np.asarray(labels, dtype=np.int32),
            multimask_output=True,
        )
        best = choose_sam_mask(masks, scores, prefer_small=(mode == "conservative" and len(points) == 1))
        marker = tuple(coords[-1])
        fg = sum(1 for label in labels if label == 1)
        bg = sum(1 for label in labels if label == 0)
        info = f"Click prompt: {fg} foreground, {bg} background | SAM score: {float(scores[best]):.3f}"
        return masks[best].astype(bool), info, marker

    def segment_box(
        self,
        image: np.ndarray,
        box: list[float] | None,
    ) -> tuple[np.ndarray, str, tuple[int, int] | list[int] | None]:
        if box is None or len(box) != 4:
            raise ValueError("box mode requires [x0, y0, x1, y1]")
        self.ensure_predictor_image(image)
        h, w = image.shape[:2]
        x0, y0, x1, y1 = box
        clean_box = np.asarray(
            [
                max(0, min(w - 1, min(x0, x1))),
                max(0, min(h - 1, min(y0, y1))),
                max(0, min(w - 1, max(x0, x1))),
                max(0, min(h - 1, max(y0, y1))),
            ],
            dtype=np.float32,
        )
        if clean_box[2] - clean_box[0] < 4 or clean_box[3] - clean_box[1] < 4:
            raise ValueError("box is too small")
        masks, scores, _ = self.predictor.predict(box=clean_box, multimask_output=True)
        best = int(np.argmax(scores))
        info = f"Box prompt: {clean_box.astype(int).tolist()} | SAM score: {float(scores[best]):.3f}"
        return masks[best].astype(bool), info, clean_box.astype(int).tolist()

    def segment_text(
        self,
        image: np.ndarray,
        text: str,
        mode: str,
    ) -> tuple[np.ndarray, str, tuple[int, int] | list[int] | None]:
        text = text.strip()
        if not text:
            raise ValueError("text mode requires a prompt")

        masks = self.auto_masks(image)
        if not masks:
            raise ValueError("SAM automatic mask generator returned no candidates")

        intent = parse_text_intent(text)
        best_idx, score, reasons = rank_text_masks(image, masks, intent, conservative=(mode == "conservative"))
        item = masks[best_idx]
        mask = item["segmentation"].astype(bool)
        marker = box_center(item["bbox"])
        sam_score = float(item.get("predicted_iou", 0.0))
        reason = ", ".join(reasons) if reasons else "generic objectness"
        info = f'Text prompt: "{text}" | Candidate {best_idx + 1}/{len(masks)} | SAM score: {sam_score:.3f} | Text score: {score:.3f} | Match: {reason}'
        return mask, info, marker

    def auto_masks(self, image: np.ndarray) -> list[dict[str, Any]]:
        digest = hashlib.sha1(image.tobytes()).hexdigest()
        if digest != self.cached_auto_hash:
            self.cached_auto_masks = self.mask_generator.generate(image)
            self.cached_auto_hash = digest
        return self.cached_auto_masks


class Sam3TextSegmenter:
    def __init__(self, checkpoint: Path, device: str, threshold: float) -> None:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.device = "cuda" if device.startswith("cuda") else device
        self.threshold = threshold
        self.autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.startswith("cuda")
            else contextlib.nullcontext()
        )
        self.model = build_sam3_image_model(
            device=self.device,
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
        )
        self.interactive = getattr(self.model, "inst_interactive_predictor", None)
        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=threshold,
        )

    def predict(self, image: np.ndarray, text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
        with self.autocast_context:
            state = self.processor.set_image(pil_image)
            state = self.processor.set_text_prompt(prompt=text, state=state)
        masks_t = state["masks"].detach().cpu()
        boxes_t = state["boxes"].detach().float().cpu()
        scores_t = state["scores"].detach().float().cpu()
        if masks_t.numel() == 0:
            return (
                np.zeros((0, image.shape[0], image.shape[1]), dtype=bool),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        masks = masks_t[:, 0].numpy().astype(bool)
        boxes = boxes_t.numpy().astype(np.float32)
        scores = scores_t.numpy().astype(np.float32)
        return masks, boxes, scores

    def predict_interactive(
        self,
        image: np.ndarray,
        prompt_type: str,
        points: list[PointPrompt],
        box: list[float] | None,
        mode: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        with self.autocast_context, torch.inference_mode():
            state = self.processor.set_image(Image.fromarray(image.astype(np.uint8), mode="RGB"))
            if "language_features" not in state["backbone_out"]:
                state["backbone_out"].update(self.model.backbone.forward_text(["visual"], device=self.device))
            if "geometric_prompt" not in state:
                state["geometric_prompt"] = self.model._get_dummy_prompt()

            h, w = image.shape[:2]
            if prompt_type == "box":
                _, _, clean_box = prepare_prompt_arrays(image, prompt_type, points, box)
                x0, y0, x1, y1 = clean_box.tolist()
                cx = ((x0 + x1) / 2.0) / float(w)
                cy = ((y0 + y1) / 2.0) / float(h)
                bw = max((x1 - x0) / float(w), 1e-6)
                bh = max((y1 - y0) / float(h), 1e-6)
                state = self.processor.add_geometric_prompt([cx, cy, bw, bh], True, state)
            elif prompt_type == "click":
                point_coords, point_labels, _ = prepare_prompt_arrays(image, prompt_type, points, box)
                norm_points = point_coords.copy()
                norm_points[:, 0] /= float(w)
                norm_points[:, 1] /= float(h)
                points_t = torch.tensor(norm_points, device=self.device, dtype=torch.float32).view(-1, 1, 2)
                labels_t = torch.tensor(point_labels, device=self.device, dtype=torch.long).view(-1, 1)
                state["geometric_prompt"].append_points(points_t, labels_t)
                state = self.processor._forward_grounding(state)
            else:
                raise ValueError(f"{prompt_type} is not supported by SAM3 interactive mode")

        masks_t = state["masks"].detach().cpu()
        scores_t = state["scores"].detach().float().cpu()
        if masks_t.numel() == 0:
            return (
                np.zeros((0, image.shape[0], image.shape[1]), dtype=bool),
                np.zeros((0,), dtype=np.float32),
            )
        masks = masks_t[:, 0].numpy().astype(bool)
        scores = scores_t.numpy().astype(np.float32)
        return masks, scores


class Sam2ImageSegmenter:
    def __init__(self, checkpoint: Path, config: str, device: str) -> None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = "cuda" if device.startswith("cuda") else device
        self.autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.startswith("cuda")
            else contextlib.nullcontext()
        )
        model = build_sam2(config, str(checkpoint), device=self.device)
        self.predictor = SAM2ImagePredictor(model)

    def predict(
        self,
        image: np.ndarray,
        prompt_type: str,
        points: list[PointPrompt],
        box: list[float] | None,
        mode: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        point_coords, point_labels, clean_box = prepare_prompt_arrays(image, prompt_type, points, box)
        point_count = 0 if point_labels is None else len(point_labels)
        multimask = not (prompt_type == "box" or point_count > 1)
        with torch.inference_mode(), self.autocast_context:
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=clean_box,
                multimask_output=multimask,
            )
        return masks.astype(bool), scores.astype(np.float32)


class Sam2VideoSegmenter:
    def __init__(self, checkpoint: Path, config: str, device: str) -> None:
        from sam2.build_sam import build_sam2_video_predictor

        self.device = "cuda" if device.startswith("cuda") else device
        self.autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.startswith("cuda")
            else contextlib.nullcontext()
        )
        self.predictor = build_sam2_video_predictor(config, str(checkpoint), device=self.device)

    def segment_video(
        self,
        video_path: Path,
        prompt_type: str,
        points: list[PointPrompt],
        box: list[float] | None,
        max_frames: int,
    ) -> tuple[list[np.ndarray], str]:
        first_frame = load_video_preview_frame(video_path)
        point_coords, point_labels, clean_box = prepare_prompt_arrays(first_frame, prompt_type, points, box)
        max_frames = max(1, min(int(max_frames), 72))
        overlays: list[np.ndarray] = []
        with torch.inference_mode(), self.autocast_context:
            state = self.predictor.init_state(str(video_path), offload_video_to_cpu=True)
            self.predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=1,
                points=point_coords,
                labels=point_labels,
                box=clean_box,
            )
            source_frames = load_video_frames_for_overlay(video_path, max_frames=max_frames)
            for frame_idx, object_ids, masks_t in self.predictor.propagate_in_video(state):
                if frame_idx >= len(source_frames) or len(overlays) >= max_frames:
                    break
                masks = (masks_t.detach().float().cpu().numpy() > 0.0)
                if masks.ndim == 4:
                    masks = masks[:, 0]
                boxes = masks_to_xyxy(masks)
                scores = np.ones((len(masks),), dtype=np.float32)
                overlays.append(make_multi_overlay(source_frames[frame_idx], masks, boxes, scores))
        info = (
            f"SAM2 video {prompt_type} prompt | Frames rendered: {len(overlays)} | "
            f"Source: {video_path.name} | Device: {self.device}"
        )
        return overlays, info


class MedSamBoxSegmenter:
    def __init__(self, checkpoint: Path, device: str) -> None:
        self.device = "cuda" if device.startswith("cuda") else device
        self.model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
        self.model.to(device=self.device)
        self.model.eval()

    def predict(self, image: np.ndarray, box: list[float] | None) -> np.ndarray:
        _, _, clean_box = prepare_prompt_arrays(image, "box", [], box)
        h, w = image.shape[:2]
        img_1024 = resize_rgb(image, (1024, 1024)).astype(np.float32)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), 1e-8, None)
        img_tensor = torch.tensor(img_1024, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0)
        box_1024 = clean_box[None, :] / np.array([w, h, w, h], dtype=np.float32) * 1024.0
        with torch.no_grad():
            image_embedding = self.model.image_encoder(img_tensor)
            box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=self.device)[:, None, :]
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            low_res_logits, _ = self.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            pred = torch.sigmoid(low_res_logits)
            pred = torch.nn.functional.interpolate(
                pred,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
        return (pred.squeeze().detach().cpu().numpy() > 0.5)


class MatcherOneShotSegmenter:
    def __init__(self, device: str) -> None:
        matcher_root = WORKSPACE_ROOT / "Matcher"
        if not matcher_root.exists():
            raise FileNotFoundError("Matcher repo not found at D:\\SAM\\Matcher")
        models_root = matcher_root / "models"
        required = [
            models_root / "sam_vit_h_4b8939.pth",
            models_root / "dinov2_vitl14_pretrain.pth",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Matcher missing required weights: " + "; ".join(missing))
        self.device = "cuda:0" if device.startswith("cuda") else "cpu"
        self.matcher_root = matcher_root
        self.worker = REPO_ROOT / "demos" / "matcher_oneshot_worker.py"

    def predict(
        self,
        support_image: np.ndarray,
        support_mask: np.ndarray | None,
        support_box: list[float],
        target_image: np.ndarray,
        version: int,
    ) -> np.ndarray:
        _, _, clean_box = prepare_prompt_arrays(support_image, "box", [], support_box)
        version = max(1, min(int(version), 3))
        with tempfile.TemporaryDirectory(prefix="matcher_oneshot_") as tmp:
            tmp_path = Path(tmp)
            support_path = tmp_path / "support.png"
            support_mask_path = tmp_path / "support_mask.png"
            target_path = tmp_path / "target.png"
            mask_path = tmp_path / "mask.npy"
            Image.fromarray(support_image.astype(np.uint8)).save(support_path)
            mask_arg = ""
            if support_mask is not None:
                Image.fromarray(support_mask.astype(np.uint8) * 255, mode="L").save(support_mask_path)
                mask_arg = str(support_mask_path)
            Image.fromarray(target_image.astype(np.uint8)).save(target_path)
            cmd = [
                sys.executable,
                "-B",
                str(self.worker),
                "--matcher-root",
                str(self.matcher_root),
                "--support",
                str(support_path),
                "--target",
                str(target_path),
                "--box",
                ",".join(str(float(v)) for v in clean_box.tolist()),
                "--mask",
                mask_arg,
                "--version",
                str(version),
                "--out",
                str(mask_path),
            ]
            env = os.environ.copy()
            env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            completed = subprocess.run(
                cmd,
                cwd=str(self.matcher_root),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
            )
            if completed.returncode != 0:
                raise RuntimeError((completed.stderr or completed.stdout).strip()[-1200:])
            return np.load(mask_path).astype(bool)


def choose_sam_mask(masks: np.ndarray, scores: np.ndarray, prefer_small: bool) -> int:
    if not prefer_small:
        return int(np.argmax(scores))
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float32)
    area_rank = 1.0 - areas / max(float(areas.max()), 1.0)
    combined = np.asarray(scores, dtype=np.float32) + 0.08 * area_rank
    return int(np.argmax(combined))


def prepare_prompt_arrays(
    image: np.ndarray,
    prompt_type: str,
    points: list[PointPrompt],
    box: list[float] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    h, w = image.shape[:2]
    clean_points: list[list[float]] = []
    clean_labels: list[int] = []
    clean_box: np.ndarray | None = None

    if prompt_type == "click":
        if not points:
            raise ValueError("click mode requires at least one point")
        for point in points:
            clean_points.append([
                float(max(0, min(w - 1, int(point.x)))),
                float(max(0, min(h - 1, int(point.y)))),
            ])
            clean_labels.append(1 if int(point.label) else 0)
    elif prompt_type == "box":
        if box is None or len(box) != 4:
            raise ValueError("box mode requires [x0, y0, x1, y1]")
        x0, y0, x1, y1 = box
        clean_box = np.asarray(
            [
                max(0, min(w - 1, min(x0, x1))),
                max(0, min(h - 1, min(y0, y1))),
                max(0, min(w - 1, max(x0, x1))),
                max(0, min(h - 1, max(y0, y1))),
            ],
            dtype=np.float32,
        )
        if clean_box[2] - clean_box[0] < 4 or clean_box[3] - clean_box[1] < 4:
            raise ValueError("box is too small")
    else:
        raise ValueError(f"{prompt_type} is not supported by this backend")

    point_coords = np.asarray(clean_points, dtype=np.float32) if clean_points else None
    point_labels = np.asarray(clean_labels, dtype=np.int32) if clean_labels else None
    return point_coords, point_labels, clean_box


def masks_to_xyxy(masks: np.ndarray) -> np.ndarray:
    boxes: list[list[float]] = []
    for mask in masks.astype(bool):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            boxes.append([0, 0, 0, 0])
        else:
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
    return np.asarray(boxes, dtype=np.float32)


def resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(image.astype(np.uint8)).resize(size, Image.BICUBIC))


def load_video_preview_frame(video_path: Path) -> np.ndarray:
    if video_path.is_dir():
        frames = sorted(video_path.glob("*.jpg"))
        if not frames:
            raise ValueError(f"video frame folder has no jpg frames: {video_path}")
        return np.asarray(Image.open(frames[0]).convert("RGB"))
    import decord

    vr = decord.VideoReader(str(video_path))
    if len(vr) == 0:
        raise ValueError(f"video has no frames: {video_path}")
    return decord_frame_to_numpy(vr[0])


def load_video_frames_for_overlay(video_path: Path, max_frames: int) -> list[np.ndarray]:
    if video_path.is_dir():
        frames = sorted(video_path.glob("*.jpg"))[:max_frames]
        return [np.asarray(Image.open(frame).convert("RGB")) for frame in frames]
    import decord

    vr = decord.VideoReader(str(video_path))
    return [decord_frame_to_numpy(vr[i]) for i in range(min(len(vr), max_frames))]


def decord_frame_to_numpy(frame: Any) -> np.ndarray:
    if hasattr(frame, "asnumpy"):
        return frame.asnumpy()
    if torch.is_tensor(frame):
        return frame.detach().cpu().numpy()
    return np.asarray(frame)


def parse_text_intent(text: str) -> TextIntent:
    t = text.lower()
    intent = TextIntent()
    if has_any(t, "largest", "biggest", "large", "big", "最大", "较大", "大的", "大目标"):
        intent.prefer_large = True
    if has_any(t, "smallest", "small", "tiny", "最小", "较小", "小的", "小目标"):
        intent.prefer_small = True
    if has_any(t, "center", "middle", "central", "中央", "中心", "中间"):
        intent.prefer_center = True
    if has_any(t, "left", "左"):
        intent.prefer_left = True
    if has_any(t, "right", "右"):
        intent.prefer_right = True
    if has_any(t, "top", "upper", "上方", "上部", "顶部"):
        intent.prefer_top = True
    if has_any(t, "bottom", "lower", "下方", "下部", "底部"):
        intent.prefer_bottom = True
    if has_any(t, "dark", "black", "darker", "暗", "黑", "深色"):
        intent.prefer_dark = True
    if has_any(t, "bright", "white", "light", "亮", "白", "浅色", "高亮"):
        intent.prefer_bright = True

    color_table = {
        ("red", "红"): np.array([210, 55, 45], dtype=np.float32),
        ("orange", "橙"): np.array([225, 130, 35], dtype=np.float32),
        ("yellow", "黄"): np.array([220, 200, 60], dtype=np.float32),
        ("green", "绿"): np.array([60, 170, 80], dtype=np.float32),
        ("blue", "蓝"): np.array([55, 120, 210], dtype=np.float32),
        ("purple", "violet", "紫"): np.array([135, 80, 170], dtype=np.float32),
    }
    for keys, rgb in color_table.items():
        if has_any(t, *keys):
            intent.target_color = rgb
            break

    if has_any(t, "nucleus", "nuclei", "cell", "细胞核", "细胞", "核"):
        intent.medical = "nuclei"
        intent.prefer_small = True
        intent.prefer_dark = True
    elif has_any(t, "lesion", "病灶", "出血", "hemorrhage", "microaneurysm", "ma", "红病灶"):
        intent.medical = "red-lesion"
        intent.prefer_small = True
        intent.target_color = np.array([170, 45, 40], dtype=np.float32)
    elif has_any(t, "exudate", "渗出", "hard exudate", "yellow lesion", "黄斑"):
        intent.medical = "yellow-lesion"
        intent.prefer_small = True
        intent.target_color = np.array([220, 190, 70], dtype=np.float32)
        intent.prefer_bright = True
    elif has_any(t, "optic disc", "disc", "视盘", "optic cup", "cup", "杯盘"):
        intent.medical = "optic-disc"
        intent.prefer_bright = True
        intent.target_color = np.array([218, 182, 95], dtype=np.float32)

    return intent


def has_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def rank_text_masks(
    image: np.ndarray,
    masks: list[dict[str, Any]],
    intent: TextIntent,
    conservative: bool,
) -> tuple[int, float, list[str]]:
    h, w = image.shape[:2]
    image_f = image.astype(np.float32)
    image_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    diag = float(np.hypot(w, h))
    areas = np.asarray([float(item["area"]) for item in masks], dtype=np.float32)
    max_area = max(float(areas.max()), 1.0)
    min_reasonable_area = max(16.0, image.size / 3 / 8000.0)

    best_idx = 0
    best_score = -1e9
    best_reasons: list[str] = []
    for idx, item in enumerate(masks):
        mask = item["segmentation"].astype(bool)
        area = float(item["area"])
        if area < min_reasonable_area:
            continue
        bbox = item["bbox"]
        cx, cy = box_center(bbox)
        center = np.array([cx, cy], dtype=np.float32)
        center_score = 1.0 - min(1.0, float(np.linalg.norm(center - image_center) / max(diag / 2.0, 1.0)))
        objectness = float(item.get("predicted_iou", 0.0)) + 0.25 * float(item.get("stability_score", 0.0))
        area_norm = area / max_area
        score = 0.35 * objectness
        reasons: list[str] = []

        if intent.prefer_large:
            score += 0.8 * area_norm
            reasons.append("large")
        if intent.prefer_small or conservative:
            small_score = 1.0 - min(1.0, area_norm * 5.0)
            score += 0.65 * small_score
            reasons.append("small")
        if intent.prefer_center or not reasons:
            score += 0.35 * center_score
            if intent.prefer_center:
                reasons.append("center")
        if intent.prefer_left:
            score += 0.45 * (1.0 - cx / max(w - 1, 1))
            reasons.append("left")
        if intent.prefer_right:
            score += 0.45 * (cx / max(w - 1, 1))
            reasons.append("right")
        if intent.prefer_top:
            score += 0.45 * (1.0 - cy / max(h - 1, 1))
            reasons.append("top")
        if intent.prefer_bottom:
            score += 0.45 * (cy / max(h - 1, 1))
            reasons.append("bottom")

        pixels = image_f[mask]
        if pixels.size:
            mean_rgb = pixels.mean(axis=0)
            brightness = float(mean_rgb.mean() / 255.0)
            saturation = float((mean_rgb.max() - mean_rgb.min()) / 255.0)
            if intent.prefer_dark:
                score += 0.45 * (1.0 - brightness)
                reasons.append("dark")
            if intent.prefer_bright:
                score += 0.45 * brightness
                reasons.append("bright")
            if intent.target_color is not None:
                dist = float(np.linalg.norm(mean_rgb - intent.target_color) / np.linalg.norm(np.array([255, 255, 255], dtype=np.float32)))
                score += 0.8 * (1.0 - min(1.0, dist)) + 0.15 * saturation
                reasons.append("color")
            if intent.medical == "nuclei":
                score += 0.2 * (1.0 - brightness)
            elif intent.medical in {"red-lesion", "yellow-lesion"}:
                score += 0.15 * saturation

        if score > best_score:
            best_score = score
            best_idx = idx
            best_reasons = reasons

    return best_idx, float(best_score), best_reasons


def box_center(bbox: list[float] | tuple[float, float, float, float]) -> tuple[int, int]:
    x, y, bw, bh = bbox
    return int(round(x + bw / 2.0)), int(round(y + bh / 2.0))


def decode_data_url(data_url: str) -> np.ndarray:
    marker = "base64,"
    if marker not in data_url:
        raise ValueError("image must be a base64 data URL")
    raw = base64.b64decode(data_url.split(marker, 1)[1])
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(image)


def encode_png(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_mask_png(mask: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_score_from_info(info: str) -> float:
    match = re.search(r"(?:SAM score|Text score):\s*([0-9.]+)", info)
    if not match:
        return 1.0
    return float(match.group(1))


def encode_gif(frames: list[np.ndarray], duration_ms: int = 90) -> str:
    if not frames:
        raise ValueError("no video frames were rendered")
    pil_frames = [
        Image.fromarray(frame.astype(np.uint8)).convert("P", palette=Image.Palette.ADAPTIVE)
        for frame in frames
    ]
    buffer = io.BytesIO()
    pil_frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return "data:image/gif;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def make_overlay(image: np.ndarray, mask: np.ndarray, marker: tuple[int, int] | list[int] | None) -> np.ndarray:
    out = image.copy().astype(np.float32) * 0.62
    tint = np.array([31, 143, 214], dtype=np.float32)
    original = image.astype(np.float32)
    out[mask] = 0.38 * original[mask] + 0.62 * tint

    boundary = mask.copy()
    boundary[1:, :] &= mask[:-1, :]
    boundary[:-1, :] &= mask[1:, :]
    boundary[:, 1:] &= mask[:, :-1]
    boundary[:, :-1] &= mask[:, 1:]
    boundary = mask & ~boundary
    out[boundary] = np.array([255, 216, 77], dtype=np.float32)

    if marker is not None and len(marker) == 2:
        x, y = int(marker[0]), int(marker[1])
        h, w = mask.shape
        yy, xx = np.ogrid[:h, :w]
        ring = (xx - x) ** 2 + (yy - y) ** 2
        out[(ring <= 54) & (ring >= 20)] = np.array([255, 216, 77], dtype=np.float32)
        out[ring < 20] = np.array([255, 80, 80], dtype=np.float32)
    elif marker is not None and len(marker) == 4:
        x0, y0, x1, y1 = [int(v) for v in marker]
        out[max(0, y0 - 1): min(out.shape[0], y0 + 2), max(0, x0): min(out.shape[1], x1 + 1)] = np.array([77, 180, 255], dtype=np.float32)
        out[max(0, y1 - 1): min(out.shape[0], y1 + 2), max(0, x0): min(out.shape[1], x1 + 1)] = np.array([77, 180, 255], dtype=np.float32)
        out[max(0, y0): min(out.shape[0], y1 + 1), max(0, x0 - 1): min(out.shape[1], x0 + 2)] = np.array([77, 180, 255], dtype=np.float32)
        out[max(0, y0): min(out.shape[0], y1 + 1), max(0, x1 - 1): min(out.shape[1], x1 + 2)] = np.array([77, 180, 255], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_multi_overlay(
    image: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    out = image.copy().astype(np.float32)
    if len(masks) == 0:
        return out.astype(np.uint8)

    palette = np.asarray(
        [
            [31, 143, 214],
            [46, 160, 67],
            [219, 92, 54],
            [139, 92, 246],
            [236, 180, 39],
            [20, 184, 166],
            [236, 72, 153],
            [99, 102, 241],
        ],
        dtype=np.float32,
    )
    for idx, mask in enumerate(masks.astype(bool)):
        color = palette[idx % len(palette)]
        out[mask] = 0.45 * out[mask] + 0.55 * color

        boundary = mask.copy()
        boundary[1:, :] &= mask[:-1, :]
        boundary[:-1, :] &= mask[1:, :]
        boundary[:, 1:] &= mask[:, :-1]
        boundary[:, :-1] &= mask[:, 1:]
        boundary = mask & ~boundary
        out[boundary] = np.array([255, 245, 120], dtype=np.float32)

        if idx < len(boxes):
            x0, y0, x1, y1 = [int(round(v)) for v in boxes[idx]]
            x0 = max(0, min(image.shape[1] - 1, x0))
            x1 = max(0, min(image.shape[1] - 1, x1))
            y0 = max(0, min(image.shape[0] - 1, y0))
            y1 = max(0, min(image.shape[0] - 1, y1))
            line_color = np.array([77, 180, 255], dtype=np.float32)
            out[max(0, y0 - 1): min(out.shape[0], y0 + 2), x0: x1 + 1] = line_color
            out[max(0, y1 - 1): min(out.shape[0], y1 + 2), x0: x1 + 1] = line_color
            out[y0: y1 + 1, max(0, x0 - 1): min(out.shape[1], x0 + 2)] = line_color
            out[y0: y1 + 1, max(0, x1 - 1): min(out.shape[1], x1 + 2)] = line_color

    return np.clip(out, 0, 255).astype(np.uint8)


def find_examples() -> list[Path]:
    return [sample.path for sample in find_image_samples(REPO_ROOT, WORKSPACE_ROOT)]


def find_video_examples() -> list[Path]:
    return [sample.path for sample in find_video_samples(WORKSPACE_ROOT)]


def prediction_payload(
    *,
    overlay_key: str,
    overlay_value: str,
    info: str,
    prediction: dict[str, Any] | None = None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        overlay_key: overlay_value,
        "info": info,
        "prediction": prediction or {"masks": [], "boxes_xyxy": [], "scores": [], "metadata": metadata},
    }


def create_app(server: PromptSamServer) -> FastAPI:
    app = FastAPI(title="OCSAM Prompt Demo")
    image_samples = find_image_samples(REPO_ROOT, WORKSPACE_ROOT)
    video_samples = find_video_samples(WORKSPACE_ROOT)
    examples = [sample.path for sample in image_samples]
    video_examples = [sample.path for sample in video_samples]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.get("/models")
    def list_models() -> dict[str, Any]:
        return registry_payload(WORKSPACE_ROOT)

    @app.get("/samples")
    def list_samples() -> dict[str, Any]:
        return sample_manifest_payload(image_samples, video_samples)

    @app.get("/examples")
    def list_examples() -> dict[str, Any]:
        return {
            "examples": [
                sample.to_dict(url=f"/example/{i}")
                for i, sample in enumerate(image_samples)
            ]
        }

    @app.get("/example/{example_id}")
    def get_example(example_id: int) -> FileResponse:
        if example_id < 0 or example_id >= len(examples):
            raise HTTPException(status_code=404, detail="example not found")
        return FileResponse(examples[example_id])

    @app.get("/videos")
    def list_videos() -> dict[str, Any]:
        return {
            "videos": [
                {
                    "dataset": video_samples[i].dataset,
                    "difficulty": video_samples[i].difficulty,
                    "id": i,
                    "name": p.name,
                    "kind": "frames" if p.is_dir() else "mp4",
                    "scene": video_samples[i].scene,
                    "url": f"/video/{i}" if p.is_file() else "",
                }
                for i, p in enumerate(video_examples)
            ]
        }

    @app.get("/video/{video_id}")
    def get_video(video_id: int) -> FileResponse:
        if video_id < 0 or video_id >= len(video_examples) or not video_examples[video_id].is_file():
            raise HTTPException(status_code=404, detail="video not found")
        return FileResponse(video_examples[video_id], media_type="video/mp4")

    @app.get("/video_preview/{video_id}")
    def get_video_preview(video_id: int) -> dict[str, str]:
        if video_id < 0 or video_id >= len(video_examples):
            raise HTTPException(status_code=404, detail="video not found")
        try:
            frame = load_video_preview_frame(video_examples[video_id])
            return {"preview": encode_png(frame)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/segment")
    def segment(req: SegmentRequest) -> dict[str, Any]:
        try:
            image = decode_data_url(req.image)
            result = server.segment(req, image)
            return prediction_payload(
                overlay_key="overlay",
                overlay_value=encode_png(result.overlay),
                info=result.info,
                prediction=result.prediction_payload(),
                metadata={
                    "backend": req.backend,
                    "prompt_type": req.prompt_type,
                    "mode": req.mode,
                    "image_shape": list(image.shape),
                    "contract": "Prediction",
                },
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/compare_segment")
    def compare_segment(req: CompareRequest) -> dict[str, Any]:
        image = decode_data_url(req.image)
        results: list[dict[str, Any]] = []
        for model_name in req.models:
            model_req = SegmentRequest(
                image=req.image,
                backend=model_name,
                prompt_type=req.prompt_type,
                points=req.points,
                box=req.box,
                text=req.text,
                mode=req.mode,
            )
            try:
                result = server.segment(model_req, image)
                results.append(
                    {
                        "model": model_name,
                        "ok": True,
                        "overlay": encode_png(result.overlay),
                        "info": result.info,
                        "prediction": result.prediction_payload(),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "model": model_name,
                        "ok": False,
                        "error": str(exc),
                        "prediction": {"masks": [], "boxes_xyxy": [], "scores": [], "metadata": {"backend": model_name}},
                    }
                )
        return {"results": results}

    @app.post("/video_segment")
    def video_segment(req: VideoSegmentRequest) -> dict[str, Any]:
        try:
            if req.video_id < 0 or req.video_id >= len(video_examples):
                raise ValueError("video not found")
            frames, info = server.segment_sam2_video(req, video_examples[req.video_id])
            return prediction_payload(
                overlay_key="animation",
                overlay_value=encode_gif(frames),
                info=info,
                metadata={
                    "backend": "sam2",
                    "media_type": "video",
                    "prompt_type": req.prompt_type,
                    "video_id": req.video_id,
                    "rendered_frames": len(frames),
                    "contract": "VideoPrediction",
                },
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/matcher_segment")
    def matcher_segment(req: MatcherRequest) -> dict[str, Any]:
        try:
            result = server.segment_matcher(req)
            return prediction_payload(
                overlay_key="overlay",
                overlay_value=encode_png(result.overlay),
                info=result.info,
                prediction=result.prediction_payload(),
                metadata={
                    "backend": "matcher",
                    "prompt_type": "reference",
                    "support_box": req.support_box,
                    "version": req.version,
                    "contract": "Prediction",
                },
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "sam_vit_b.pth"))
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--amg-points-per-side", type=int, default=24)
    parser.add_argument("--sam3-checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "sam3_modelscope" / "sam3.pt"))
    parser.add_argument("--sam3-threshold", type=float, default=0.5)
    parser.add_argument("--preload-sam3", action="store_true")
    parser.add_argument("--sam2-checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "sam2.1_hiera_tiny.pt"))
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--preload-sam2", action="store_true")
    parser.add_argument("--medsam-checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "medsam_vit_b.pth"))
    parser.add_argument("--preload-medsam", action="store_true")
    parser.add_argument("--disable-matcher", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = PromptSamServer(
        checkpoint=Path(args.checkpoint),
        model_type=args.model_type,
        device=args.device,
        amg_points_per_side=args.amg_points_per_side,
        sam3_checkpoint=Path(args.sam3_checkpoint),
        sam3_threshold=args.sam3_threshold,
        sam2_checkpoint=Path(args.sam2_checkpoint),
        sam2_config=args.sam2_config,
        medsam_checkpoint=Path(args.medsam_checkpoint),
        matcher_enabled=not args.disable_matcher,
    )
    if args.preload_sam2:
        print("Preloading SAM2 image backend...", flush=True)
        server.warmup_sam2_image()
        print("SAM2 image backend ready.", flush=True)
    if args.preload_sam3:
        print("Preloading SAM3 text backend...", flush=True)
        server.warmup_sam3()
        print("SAM3 text backend ready.", flush=True)
    if args.preload_medsam:
        print("Preloading MedSAM backend...", flush=True)
        server.warmup_medsam()
        print("MedSAM backend ready.", flush=True)
    app = create_app(server)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
