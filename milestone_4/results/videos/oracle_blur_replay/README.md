# Oracle and blur: saved-plan video replay

The final presentation videos are in **`sidebar_prior/`**:

- [Oracle](sidebar_prior/cx0.25_cy0.25_layoutB__gamma_0.05__oracle.mp4)
- [Blur, JS = 0.15](sidebar_prior/cx0.25_cy0.25_layoutB__gamma_0.05__diffuse_blur_js_0.15.mp4)
- [Preview](sidebar_prior/preview.png)
- [Parameter sources and artifact hashes](sidebar_prior/render_manifest.json)
- [Video validation](sidebar_prior/video_validation.json)

Both videos show scene `cx0.25_cy0.25_layoutB`, gamma 0.05. Each plays the
15-second experiment over 20 seconds (301 frames, 1440 × 900, H.264).
The main map always shows the saved true prior. The actual planner prior is
displayed in the right information column, above the detection-probability
curve, with an independently labelled colour scale for each condition.

Both videos use the same existing target draw: seed 104, episode 0,
position (0.1375, 0.4625). Saved detection times are 0.55 s for oracle and
2.65 s for blur. Coverage continues through the experiment budget after this
illustrative target is found.

`render_saved.py` reads saved inputs, trajectories and `first_seen_grid`.
It imports the original camera execution and visibility helpers and checks
frozen input hashes. It never imports or invokes the planner. Experimental
parameters come from the existing source and frozen manifest; rendering
duration and layout are presentation choices.

To render into a new output directory, use a Python environment with NumPy,
Matplotlib, Pillow and an OpenCV build supporting H.264:

```sh
python -B milestone_4/results/videos/oracle_blur_replay/render_saved.py \
  --scene cx0.25_cy0.25_layoutB --gamma 0.05 \
  --priors oracle diffuse_blur_js_0.15 --seconds 20 \
  --output-subdir new_render
```

Original videos without the prior inset remain in this directory. The
intermediate map-overlay version with a shared inset scale is archived in
`prior_inset/`. Use `sidebar_prior/` for the final version.
