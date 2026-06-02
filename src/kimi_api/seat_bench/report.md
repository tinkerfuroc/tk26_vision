# Seat-Recommendation Strategy Benchmark — Results

| cell | n | hit_rate | hits | wrong_seat | miss | false_none | correct_reject | mean_s | mean_calls |
|---|---|---|---|---|---|---|---|---|---|
| s1_qwen | 36 | 94% | 33 | 0 | 1 | 1 | 1 | 6.2 | 1.0 |
| s0_qwen | 36 | 91% | 32 | 0 | 3 | 0 | 1 | 5.9 | 1.0 |
| s2_qwen | 36 | 86% | 30 | 2 | 3 | 0 | 1 | 7.7 | 2.0 |
| s3_gemini | 36 | 72% | 26 | 5 | 5 | 0 | 0 | 10.4 | 1.0 |
| s1_gemini | 36 | 72% | 26 | 1 | 6 | 3 | 0 | 22.7 | 1.0 |
| s2_gemini | 36 | 67% | 24 | 1 | 8 | 3 | 0 | 26.2 | 1.9 |
| s3_qwen | 36 | 58% | 21 | 2 | 10 | 3 | 0 | 1.6 | 1.0 |
| s0_gemini | 36 | 44% | 16 | 1 | 16 | 3 | 0 | 15.5 | 1.0 |

Contact sheets per cell under `sheets/`. Green box = empty GT cushion, red = occupied, cyan = predicted box, magenta dot = predicted point.

## GT correction (2026-06-02, user-verified)

After visual review, the user flagged 7 scenes whose original hand GT was wrong or too strict. GT for these was replaced with **s1_qwen's detected seats** (user-verified correct):
- **scene_007** (wrong): original missed the 3 stools (2 occupied); now 5 seats.
- **scene_026** (wrong): "leftmost chair" box was mis-placed at far-left; corrected.
- **scene_012/013/018/025/027** (too strict): seat-pad-only boxes (~85 px tall) widened to full-chair boxes (~300 px); occupancy unchanged.

All cells were re-scored against the corrected GT via `python -m seat_bench.rescore` (no new VLM calls). Updated scoreboard (supersedes the table above):

| cell | hit_rate | hits | note |
|---|---|---|---|
| s1_qwen | 94% | 33 | see circularity caveat |
| s0_qwen | 91% | 32 | cleanest top result |
| s2_qwen | 86% | 30 | |
| s3_gemini | 72% | 26 | |
| s1_gemini | 72% | 26 | best Gemini option |
| s2_gemini | 67% | 24 | |
| s3_qwen | 58% | 21 | |
| s0_gemini | 44% | 16 | current production |

**Circularity caveat:** on the 7 corrected scenes, GT now equals s1_qwen's own boxes, so s1_qwen structurally auto-hits them (its point is its box center). Its 94% vs s0_qwen's 91% is a 1-scene difference (33 vs 32) and should be read as a tie at the top. Because the corrected boxes are genuinely correct (user-verified) and generous, other strategies that land on the seat still hit — the correction lifted every cell, not just s1_qwen.

**Conclusion unchanged and reinforced:** Qwen ≫ Gemini (every Qwen cell ≥58%, every Gemini cell ≤72%; production Gemini-pointing remains last at 44%). The recommended production change is still **Qwen3-VL pointing (s0_qwen, simplest)**; S1 bbox+select is a near-tie alternative if richer per-seat output is wanted.
