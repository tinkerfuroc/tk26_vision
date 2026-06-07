# Copyright 2026 Tinker
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gallery thumb retention stays in lockstep with views; version counts changes."""
import numpy as np

from vision_track.core.reid_gallery import ReIDGallery


def _v(i, dim=8):
    a = np.zeros(dim, dtype=np.float32)
    a[i] = 1.0
    return a


def test_thumbs_lockstep_and_version():
    g = ReIDGallery(enabled=True, size=3, novelty_max=0.99)
    assert g.version == 0 and g.thumbs == []
    assert g.maybe_add(_v(0), thumb="t0")          # anchor
    assert g.maybe_add(_v(1), thumb="t1")
    assert g.maybe_add(_v(2), thumb="t2")
    assert g.version == 3
    assert g.thumbs == ["t0", "t1", "t2"] and len(g) == 3
    # 4th admit evicts a non-anchor view; thumbs must follow the same index
    assert g.maybe_add(_v(3), thumb="t3")
    assert g.version == 4
    assert len(g) == 3 and len(g.thumbs) == 3
    assert g.thumbs[0] == "t0"                      # anchor thumb pinned
    assert "t3" in g.thumbs                         # newcomer survived the evict


def test_rejected_add_changes_nothing():
    g = ReIDGallery(enabled=True, size=3, novelty_max=0.99)
    g.maybe_add(_v(0), thumb="t0")
    v_before = g.version
    assert not g.maybe_add(_v(0), thumb="dup")      # novelty reject (cos=1.0)
    assert g.version == v_before and g.thumbs == ["t0"]
    assert not g.maybe_add(None, thumb="bad")       # invalid feature
    assert g.version == v_before


def test_thumbless_add_and_clear():
    g = ReIDGallery(enabled=True, size=3, novelty_max=0.99)
    g.maybe_add(_v(0))                              # thumb defaults to None
    assert g.thumbs == [None]
    v = g.version
    g.clear()
    assert len(g) == 0 and g.thumbs == [] and g.version == v + 1
    g.clear()                                       # clearing empty: no bump
    assert g.version == v + 1
