//! Split-screen swipe curtain state, HiDPI-exact physical pixel snapping, and
//! half-plane scissor intersection helpers for [`super::GlobeRenderer`].

/// Normalized split-screen swipe curtain parameters passed outside
/// [`super::DrawSnapshot`] so dragging the curtain never rebuilds the globe
/// quadtree or overlay cull lists.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwipeRenderState {
    /// Normalized horizontal curtain position in `[0.0, 1.0]` across the
    /// viewport width (`0.0` = far left, `1.0` = far right).
    pub split_t: f32,
    /// Optional overlay layer ID clipped strictly to the left half-plane.
    pub left_layer: Option<u64>,
    /// Optional overlay layer ID clipped strictly to the right half-plane.
    pub right_layer: Option<u64>,
}

/// Which half-plane an overlay layer occupies under an active swipe curtain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwipeClip {
    /// Unclipped across the full viewport.
    Full,
    /// Clipped to `[0, 0, split_x, height]`.
    Left,
    /// Clipped to `[split_x, 0, width - split_x, height]`.
    Right,
}

/// Snap a normalized curtain position `split_t ∈ [0.0, 1.0]` to an exact
/// physical framebuffer X coordinate in `0..=width_px`.
///
/// Both `GlobePaintCallback::prepare()` and the `egui::Painter` divider use
/// this helper so the rendered scissor edge and UI divider never drift by 1 px
/// under fractional HiDPI scaling.
pub fn snap_split_x(split_t: f32, width_px: u32) -> u32 {
    let t = if split_t.is_finite() {
        f64::from(split_t.clamp(0.0, 1.0))
    } else {
        0.5
    };
    (t * f64::from(width_px))
        .round()
        .clamp(0.0, f64::from(width_px)) as u32
}

/// Physical-pixel resolved swipe state used both for `LastRenderKey` dirty
/// checking and Pass-2 scissor enforcement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedSwipe {
    pub split_x: u32,
    pub left_layer: Option<u64>,
    pub right_layer: Option<u64>,
}

impl ResolvedSwipe {
    /// Resolve normalized [`SwipeRenderState`] against exact physical viewport
    /// width `width_px`.
    pub fn from_state(state: SwipeRenderState, width_px: u32) -> Self {
        Self {
            split_x: snap_split_x(state.split_t, width_px),
            left_layer: state.left_layer,
            right_layer: state.right_layer,
        }
    }

    /// Determine which half-plane `layer_id` renders into.
    ///
    /// If `left_layer == right_layer`, returns [`SwipeClip::Full`] so comparing
    /// a layer against itself never splits or drops it.
    pub fn clip_for_layer(&self, layer_id: u64) -> SwipeClip {
        if self.left_layer == self.right_layer {
            return SwipeClip::Full;
        }
        if matches!(self.left_layer, Some(id) if id == layer_id) {
            SwipeClip::Left
        } else if matches!(self.right_layer, Some(id) if id == layer_id) {
            SwipeClip::Right
        } else {
            SwipeClip::Full
        }
    }

    /// Compute the physical scissor rectangle `[x, y, w, h]` for `layer_id`,
    /// returning `None` when the half-plane has zero width or height (so `wgpu`
    /// never receives a zero-sized scissor rect).
    pub fn scissor_for_layer(&self, layer_id: u64, width: u32, height: u32) -> Option<[u32; 4]> {
        if width == 0 || height == 0 {
            return None;
        }
        match self.clip_for_layer(layer_id) {
            SwipeClip::Full => Some([0, 0, width, height]),
            SwipeClip::Left => {
                let w = self.split_x.min(width);
                (w > 0).then_some([0, 0, w, height])
            }
            SwipeClip::Right => {
                let x = self.split_x.min(width);
                let w = width.saturating_sub(x);
                (w > 0).then_some([x, 0, w, height])
            }
        }
    }
}

/// Intersect two `[x, y, width, height]` scissor rectangles, returning `None`
/// whenever the intersection is empty (zero width or zero height).
pub fn intersect_scissor_rects(a: [u32; 4], b: [u32; 4]) -> Option<[u32; 4]> {
    let x0 = a[0].max(b[0]);
    let y0 = a[1].max(b[1]);
    let x1 = a[0].saturating_add(a[2]).min(b[0].saturating_add(b[2]));
    let y1 = a[1].saturating_add(a[3]).min(b[1].saturating_add(b[3]));
    let w = x1.saturating_sub(x0);
    let h = y1.saturating_sub(y0);
    (w > 0 && h > 0).then_some([x0, y0, w, h])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap_split_x_handles_bounds_and_non_finite_values() {
        assert_eq!(snap_split_x(0.0, 1000), 0);
        assert_eq!(snap_split_x(1.0, 1000), 1000);
        assert_eq!(snap_split_x(-0.5, 1000), 0);
        assert_eq!(snap_split_x(1.5, 1000), 1000);
        assert_eq!(snap_split_x(0.5, 1001), 501);
        assert_eq!(snap_split_x(f32::NAN, 800), 400);
    }

    #[test]
    fn resolved_swipe_scissors_and_intersections_never_produce_zero_rects() {
        let left = ResolvedSwipe::from_state(
            SwipeRenderState {
                split_t: 0.0,
                left_layer: Some(1),
                right_layer: Some(2),
            },
            200,
        );
        assert_eq!(left.scissor_for_layer(1, 200, 100), None);
        assert_eq!(left.scissor_for_layer(2, 200, 100), Some([0, 0, 200, 100]));
        assert_eq!(left.scissor_for_layer(3, 200, 100), Some([0, 0, 200, 100]));

        let right = ResolvedSwipe::from_state(
            SwipeRenderState {
                split_t: 1.0,
                left_layer: Some(1),
                right_layer: Some(2),
            },
            200,
        );
        assert_eq!(right.scissor_for_layer(1, 200, 100), Some([0, 0, 200, 100]));
        assert_eq!(right.scissor_for_layer(2, 200, 100), None);

        let mid = ResolvedSwipe::from_state(
            SwipeRenderState {
                split_t: 0.4,
                left_layer: Some(1),
                right_layer: Some(2),
            },
            200,
        );
        assert_eq!(mid.scissor_for_layer(1, 200, 100), Some([0, 0, 80, 100]));
        assert_eq!(mid.scissor_for_layer(2, 200, 100), Some([80, 0, 120, 100]));

        // Same layer on both sides collapses safely to Full.
        let same = ResolvedSwipe::from_state(
            SwipeRenderState {
                split_t: 0.5,
                left_layer: Some(7),
                right_layer: Some(7),
            },
            200,
        );
        assert_eq!(same.clip_for_layer(7), SwipeClip::Full);

        // Disjoint and touching intersections return None.
        assert_eq!(
            intersect_scissor_rects([0, 0, 80, 100], [80, 0, 120, 100]),
            None
        );
        assert_eq!(
            intersect_scissor_rects([10, 20, 100, 50], [50, 30, 80, 100]),
            Some([50, 30, 60, 40])
        );
    }
}
