# SAM Workbench Web UI Research And Redesign Plan

## 1. Goal

Build the web page as a research workbench for promptable medical-image segmentation, not as a landing page and not as a simple demo form.

The page should support the first-stage reproduction target:

- Run SAM, MedSAM, SAM2, Matcher, and SAM3 behind one visual interface.
- Keep image, video, and one-shot reference workflows visible without mixing their controls.
- Let the user know which prompt types are valid before they click.
- Make the input canvas, output mask, model state, and current prompt state obvious at a glance.

## 2. External UI References

### 2.1 Meta Segment Anything / SAM

Source:

- https://segment-anything.com/
- https://arxiv.org/abs/2304.02643

Relevant facts:

- SAM is built around promptable segmentation.
- Point, box, and mask-style prompting are first-class mental models.
- The interface should put the prompt canvas at the center, not bury it below controls.

Borrow:

- Prompt-first interaction model.
- A large central image canvas.
- Clear separation between prompt input and mask result.

Do not borrow:

- Consumer-demo simplicity. Our use case compares multiple research models and medical examples, so it needs more state and provenance.

### 2.2 Meta SAM2

Source:

- https://github.com/facebookresearch/sam2

Relevant facts:

- SAM2 extends promptable segmentation from images to videos.
- The repository describes adding prompts on a frame, then propagating masklets through the video.
- The UI therefore needs a visible media mode and a place for frame/video-specific controls.

Borrow:

- Image/video split in the mental model.
- First-frame prompt then propagated result.
- A video strip/timeline area, even if the current implementation only uses bundled preview examples.

Do not borrow:

- A separate app for video. In this phase, video should live in the same workbench.

### 2.3 Meta SAM3

Source:

- https://github.com/facebookresearch/sam3

Relevant facts:

- SAM3 is concept-oriented and supports open-vocabulary text concepts, visual prompts, and image/video tasks.
- It detects, segments, and tracks concept instances rather than only returning a single SAM-style mask.
- Text prompt should therefore feel like a first-class concept query, not a tiny extra input.

Borrow:

- Concept prompt area with text as a primary mode.
- Result state that can later expand to multiple instances and scores.

Do not borrow:

- Treating SAM3 as only a text box. Our local integration must expose click and box where supported, but with honest capability states.

### 2.4 CVAT

Source:

- https://www.cvat.ai/

Relevant facts:

- CVAT is a production annotation workspace for image, video, and 3D data.
- Its product messaging emphasizes one workspace for raw visual data, annotation tools, automated annotation, masks, tracking, quality control, and progress.
- This suggests the right structure is not a beautiful card page, but a dense workspace with tool rails, media browser, central canvas, and inspector panels.

Borrow:

- Left-side data/media context.
- Central annotation canvas.
- Tool/inspector controls separated from the media list.
- Status, progress, and quality hints.

Do not borrow:

- Full annotation-project complexity. We only need a reproduction workbench, not task assignment, roles, review workflow, or export formats yet.

### 2.5 MONAI Label / Medical Imaging Workflows

Source:

- https://arxiv.org/abs/2203.12362

Relevant facts:

- MONAI Label supports AI-assisted interactive labeling for 3D medical images and integrates with medical frontends such as 3D Slicer and OHIF.
- Medical image segmentation workflows are case-first and viewer-first: the image is the primary artifact, and model controls assist it.

Borrow:

- Viewer-first hierarchy.
- Explicit model/application state.
- Compact, clinical/research tone rather than marketing styling.

Do not borrow:

- Full 3D layout, multi-plane viewers, or DICOM series controls in phase 1.

## 3. Layout Decision

The page should use a four-region workbench:

1. App rail
   - Narrow left rail for product identity and high-level workspace location.
   - Should not contain real navigation yet; it acts as a stable visual anchor.

2. Top status matrix
   - Current media type.
   - Active model.
   - Active prompt type.
   - Matcher support state.
   - This prevents the user from losing context while switching models.

3. Main workspace
   - Left: media library.
   - Center: prompt canvas and mask result.
   - Right: model and prompt inspector.

4. Bottom/run strip inside the center region
   - Current canvas size.
   - Result state.
   - Prompt count or mode.
   - Later this can become frame timeline, run history, and benchmark queue.

## 4. Panel Responsibilities

### 4.1 Media Library Panel

Purpose:

- Choose input image or SAM2 video example.
- Upload custom image.
- Show status/log area.

Controls:

- Image upload.
- Local Examples.
- SAM2 Video Examples.
- Status.

Reason:

- Users think "what am I segmenting?" before "what model am I using?"
- Examples should not compete visually with model settings.

### 4.2 Canvas Stage

Purpose:

- Keep image interaction central.
- Show result beside input for comparison.

Controls and states:

- Input canvas.
- Result overlay.
- Canvas size.
- Result state.
- Empty state.
- Prompt marks drawn on input only, clean source image kept separately.

Reason:

- This protects the previous bug fix: backend receives a clean source canvas, while user sees prompt marks on the visible canvas.

### 4.3 Inspector Panel

Purpose:

- Configure model and prompt.
- Show model capability constraints.
- Run text and Matcher workflows.

Controls:

- Model select.
- Prompt type tabs.
- Click foreground/background toggle.
- Text concept prompt.
- Mask selection mode.
- Matcher reference image, support box, and version.
- Clear prompt/result actions.

Reason:

- Model settings are secondary to the image, but they must stay visible while prompting.

## 5. Interaction Rules

### 5.1 Model Capability Rules

SAM:

- Click: enabled.
- Box: enabled.
- Text: enabled as heuristic text selection in this demo.

SAM2:

- Image click: enabled.
- Image box: enabled.
- Image text: disabled in this demo.
- Video click/box: enabled as first-frame prompts.

MedSAM:

- Box: enabled.
- Click/text: disabled.

SAM3:

- Text: enabled as native concept prompt.
- Click/box: enabled where local integration supports visual prompts.

Matcher:

- Does not replace the active model selector.
- Lives as one-shot reference workflow: reference image + reference box + current target image.

### 5.2 Disable Before Error

The interface should disable impossible modes before the user clicks them.

Examples:

- MedSAM locks to box.
- SAM2 video hides/locks text.
- Matcher run button is unavailable for video mode.

### 5.3 Honest Loading

The page should say "Running" and lock controls during inference.

Do not:

- Let users stack multiple expensive inference calls accidentally.
- Show model controls as active while the backend is still processing.

## 6. Visual System

Chosen direction:

- Technical research workstation.
- Inspired by IBM Carbon / CVAT-style annotation workspace.
- Dense, flat, grid-based, not decorative.

Tokens:

- Background: white and neutral gray layers.
- Canvas: near-black image inspection stage.
- Accent: one blue for active actions.
- Success/warning/error: green/yellow/red for status only.
- Radius: mostly 0px; no rounded-card marketing style.
- Shadows: avoid card shadows; use borders and surface layers.

Typography:

- Prefer IBM Plex Sans/Mono if available.
- Fall back to Microsoft YaHei / Segoe UI / system sans.
- Use mono only for small metadata and run-state labels.

## 7. Current Phase Scope

In scope now:

- Reorganize the existing single FastAPI page.
- Keep all existing backend endpoints.
- Keep all existing element IDs needed by JavaScript.
- Add a more complete workbench structure.
- Add responsive behavior.
- Add clearer empty, busy, disabled, and selected states.

Out of scope now:

- True DICOM/3D volume viewer.
- Multi-mask instance table.
- Editable mask brush.
- Benchmark table.
- User/project management.
- Export formats.

## 8. Implementation Checklist

- [ ] Keep `/`, `/examples`, `/videos`, `/segment`, `/video_segment`, and `/matcher_segment` unchanged.
- [ ] Preserve `canvas`, `result`, `backend`, `mode`, prompt tabs, Matcher inputs, examples, and videos IDs.
- [ ] Move upload/examples/status into Media Library.
- [ ] Move model/prompt/Matcher controls into Inspector.
- [ ] Keep the prompt canvas and result overlay as the visual center.
- [ ] Add top status matrix using the same runtime state values.
- [ ] Verify Python syntax.
- [ ] Verify embedded JavaScript syntax.
- [ ] Verify served HTML contains the new workbench structure.

## 9. Later Extensions

Next useful UI extensions after this phase:

- Result instance list for SAM3 concept matches.
- Video timeline with frame index and masklet preview.
- Prompt history and rerun buttons.
- Benchmark queue panel.
- Medical case metadata panel.
- Side-by-side model comparison mode.
