# Seat-Recommendation Strategy Benchmark — Results

| cell | n | hit_rate | hits | wrong_seat | miss | false_none | correct_reject | mean_s | mean_calls |
|---|---|---|---|---|---|---|---|---|---|
| s0_qwen | 36 | 89% | 31 | 1 | 3 | 0 | 1 | 5.9 | 1.0 |
| s2_qwen | 36 | 74% | 26 | 3 | 6 | 0 | 1 | 7.7 | 2.0 |
| s1_qwen | 36 | 74% | 26 | 0 | 8 | 1 | 1 | 6.2 | 1.0 |
| s1_gemini | 36 | 67% | 24 | 2 | 7 | 3 | 0 | 22.7 | 1.0 |
| s3_gemini | 36 | 58% | 21 | 5 | 10 | 0 | 0 | 10.4 | 1.0 |
| s2_gemini | 36 | 56% | 20 | 2 | 11 | 3 | 0 | 26.2 | 1.9 |
| s3_qwen | 36 | 44% | 16 | 1 | 16 | 3 | 0 | 1.6 | 1.0 |
| s0_gemini | 36 | 39% | 14 | 2 | 17 | 3 | 0 | 15.5 | 1.0 |

Contact sheets per cell under `sheets/`. Green box = empty GT cushion, red = occupied, cyan = predicted box, magenta dot = predicted point.

## Findings (2026-06-02)

**Headline: the fix is a provider swap, not a prompt redesign.** Qwen3-VL with the *existing pointing prompt* (`s0_qwen`, 89%) is the clear winner — ~2.3× the current production config, which is Gemini pointing (`s0_gemini`, 39%, the worst cell in the grid).

**Per-hypothesis:**
- The original hypothesis "bbox/select beats pointing" holds **only for Gemini**: `s1_gemini` 67% and `s2_gemini` 56% both beat `s0_gemini` 39%. So if we stay on Gemini, single-call bbox+select (S1) is the best Gemini option.
- It is **false for Qwen**: pointing (89%) beats bbox (`s1/s2_qwen` 74%) and set-of-mark (`s3_qwen` 44%). Qwen's native pointing is already excellent; structuring the output only loses information.
- **Provider dominates strategy.** The best Gemini cell (S1, 67%) still loses to plain Qwen pointing (89%).

**Set-of-mark (S3) underperformed on both providers** (gemini 58%, qwen 44%). `som_source` was `yolo_world` for all 36 scenes (no S1 fallback), so candidate *detection* was not the bottleneck — forcing a numbered pick over coarse YOLO-World boxes discarded the fine spatial reasoning the direct prompts use, and YOLO boxes sometimes spanned occupied seats (note S3's higher `wrong_seat`: gemini 5).

**Latency / cost:** Qwen is also far cheaper and faster — `s0_qwen` 5.9 s/scene, 1 call; vs `s0_gemini` 15.5 s and the two-call `s2_gemini` 26.2 s/scene. Gemini's enforced reasoning budget is the main latency source.

**Failure mode (from contact sheets):** on clean, well-separated 3-chair scenes both providers hit. Gemini pointing collapses on the sofa/stool/cluttered scenes (000–011), landing the point on a person, the floor, or the gap between cushions — exactly the production failure that motivated this study. Qwen holds up on those.

**Recommended input to the production-rewrite spec:** retarget the seat-recommend VLM to **Qwen3-VL via DashScope with the current pointing prompt** (smallest possible change to `_seat_vlm.py` — swap provider/model, keep the prompt + snap-to-horizontal + depth pipeline). Keep Gemini as a fallback; if a Gemini-only deployment is ever needed, use S1 bbox+select there, not pointing.

**Caveats:** 36 scenes, one GT annotation set (generous point-in-box cushion boxes), 2D-localization only (logs carried no depth). `wrong_seat`/`false_none` counts are small. Single sample per scene at temperature 0.2 — no multi-sample variance measured.
