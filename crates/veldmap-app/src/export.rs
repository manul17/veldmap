//! Cartographic PNG export: pure-Rust CPU overlay compositor (metric scale bar,
//! rotated North arrow, scientific colorbar, and acquisition date/caption pill)
//! invoked strictly off the UI thread before encoding via
//! [`veldmap_raster::encode_png_rgba8`].

use crate::colormap::sample_colormap_srgb;
use crate::format::{self, compute_scale_bar};
use crate::layers::ColorbarInfo;

/// User-configurable toggles for the Export PNG dialog.
#[derive(Clone, Debug)]
pub(crate) struct CartographicExportOptions {
    pub(crate) include_scale_bar: bool,
    pub(crate) include_north_arrow: bool,
    pub(crate) include_colorbar: bool,
    pub(crate) include_caption: bool,
    pub(crate) caption_text: String,
}

impl Default for CartographicExportOptions {
    fn default() -> Self {
        Self {
            include_scale_bar: true,
            include_north_arrow: true,
            include_colorbar: true,
            include_caption: true,
            caption_text: String::new(),
        }
    }
}

/// Immutable cartographic overlay specification captured at PNG export trigger
/// time and rendered on CPU off the UI thread.
#[derive(Clone, Debug)]
pub(crate) struct CartographicOverlaySpec {
    pub(crate) pixels_per_point: f32,
    pub(crate) m_per_pt: f64,
    pub(crate) heading_deg: f64,
    pub(crate) include_scale_bar: bool,
    pub(crate) include_north_arrow: bool,
    pub(crate) include_colorbar: bool,
    pub(crate) colorbar: Option<ColorbarInfo>,
    pub(crate) caption: Option<String>,
}

const BG_CARD: [u8; 4] = [11, 19, 32, 204];
const BORDER_SUBTLE: [u8; 4] = [168, 185, 155, 130];
const INK_PRIMARY: [u8; 4] = [245, 242, 232, 255];
const NEEDLE_NORTH: [u8; 4] = [220, 78, 66, 255];
const NEEDLE_SOUTH: [u8; 4] = [210, 215, 220, 255];

struct Canvas<'a> {
    width: i32,
    height: i32,
    rgba: &'a mut [u8],
}

impl<'a> Canvas<'a> {
    fn new(width: u32, height: u32, rgba: &'a mut [u8]) -> Option<Self> {
        let expected = (width as usize)
            .checked_mul(height as usize)?
            .checked_mul(4)?;
        if rgba.len() < expected || width == 0 || height == 0 {
            return None;
        }
        Some(Self {
            width: i32::try_from(width).ok()?,
            height: i32::try_from(height).ok()?,
            rgba,
        })
    }

    #[inline]
    fn blend_px(&mut self, x: i32, y: i32, src: [u8; 4]) {
        if x < 0 || y < 0 || x >= self.width || y >= self.height || src[3] == 0 {
            return;
        }
        let idx = ((y as usize) * (self.width as usize) + (x as usize)) * 4;
        if src[3] == 255 {
            self.rgba[idx..idx + 4].copy_from_slice(&src);
            return;
        }
        let sa = u32::from(src[3]);
        let inv = 255 - sa;
        for (dst_byte, &src_byte) in self.rgba[idx..idx + 3].iter_mut().zip(src.iter()) {
            let dst = u32::from(*dst_byte);
            let out = (u32::from(src_byte) * sa + dst * inv + 127) / 255;
            *dst_byte = out as u8;
        }
        self.rgba[idx + 3] = 255;
    }

    fn fill_rect(&mut self, x0: i32, y0: i32, w: i32, h: i32, color: [u8; 4]) {
        if w <= 0 || h <= 0 {
            return;
        }
        let xs = x0.max(0);
        let ys = y0.max(0);
        let xe = x0.saturating_add(w).min(self.width);
        let ye = y0.saturating_add(h).min(self.height);
        for y in ys..ye {
            for x in xs..xe {
                self.blend_px(x, y, color);
            }
        }
    }

    fn stroke_rect(&mut self, x0: i32, y0: i32, w: i32, h: i32, t: i32, color: [u8; 4]) {
        let t = t.max(1);
        self.fill_rect(x0, y0, w, t, color);
        self.fill_rect(x0, y0 + h - t, w, t, color);
        self.fill_rect(x0, y0 + t, t, (h - 2 * t).max(0), color);
        self.fill_rect(x0 + w - t, y0 + t, t, (h - 2 * t).max(0), color);
    }

    fn fill_circle(&mut self, cx: f32, cy: f32, radius: f32, color: [u8; 4]) {
        let x0 = (cx - radius).floor() as i32;
        let x1 = (cx + radius).ceil() as i32;
        let y0 = (cy - radius).floor() as i32;
        let y1 = (cy + radius).ceil() as i32;
        let r2 = radius * radius;
        for y in y0..=y1 {
            let dy = (y as f32 + 0.5) - cy;
            for x in x0..=x1 {
                let dx = (x as f32 + 0.5) - cx;
                if dx * dx + dy * dy <= r2 {
                    self.blend_px(x, y, color);
                }
            }
        }
    }

    fn stroke_circle(&mut self, cx: f32, cy: f32, radius: f32, thickness: f32, color: [u8; 4]) {
        let outer = radius;
        let inner = (radius - thickness.max(1.0)).max(0.0);
        let x0 = (cx - outer).floor() as i32;
        let x1 = (cx + outer).ceil() as i32;
        let y0 = (cy - outer).floor() as i32;
        let y1 = (cy + outer).ceil() as i32;
        let outer2 = outer * outer;
        let inner2 = inner * inner;
        for y in y0..=y1 {
            let dy = (y as f32 + 0.5) - cy;
            for x in x0..=x1 {
                let dx = (x as f32 + 0.5) - cx;
                let d2 = dx * dx + dy * dy;
                if d2 <= outer2 && d2 >= inner2 {
                    self.blend_px(x, y, color);
                }
            }
        }
    }

    fn fill_triangle(&mut self, a: [f32; 2], b: [f32; 2], c: [f32; 2], color: [u8; 4]) {
        let min_x = a[0].min(b[0]).min(c[0]).floor() as i32;
        let max_x = a[0].max(b[0]).max(c[0]).ceil() as i32;
        let min_y = a[1].min(b[1]).min(c[1]).floor() as i32;
        let max_y = a[1].max(b[1]).max(c[1]).ceil() as i32;
        let edge = |p0: [f32; 2], p1: [f32; 2], px: f32, py: f32| {
            (px - p0[0]) * (p1[1] - p0[1]) - (py - p0[1]) * (p1[0] - p0[0])
        };
        let area = edge(a, b, c[0], c[1]);
        if area.abs() < 1e-4 {
            return;
        }
        let sign = area.signum();
        for y in min_y.max(0)..=max_y.min(self.height - 1) {
            let py = y as f32 + 0.5;
            for x in min_x.max(0)..=max_x.min(self.width - 1) {
                let px = x as f32 + 0.5;
                if edge(a, b, px, py) * sign >= 0.0
                    && edge(b, c, px, py) * sign >= 0.0
                    && edge(c, a, px, py) * sign >= 0.0
                {
                    self.blend_px(x, y, color);
                }
            }
        }
    }

    fn draw_text(&mut self, x0: i32, y0: i32, text: &str, scale: i32, color: [u8; 4]) {
        let s = scale.max(1);
        let mut cursor_x = x0;
        for ch in text.chars() {
            let glyph = glyph_5x7(ch);
            for (row, &bits) in glyph.iter().enumerate() {
                for col in 0..5 {
                    if (bits & (1 << (4 - col))) != 0 {
                        self.fill_rect(cursor_x + col * s, y0 + (row as i32) * s, s, s, color);
                    }
                }
            }
            cursor_x += 6 * s;
        }
    }
}

fn text_width_px(text: &str, scale: i32) -> i32 {
    let count = text.chars().count() as i32;
    if count <= 0 {
        0
    } else {
        (count * 6 - 1) * scale.max(1)
    }
}

/// Compact 5×7 ASCII bitmap font covering digits, uppercase and lowercase
/// letters (preserving SI unit casing such as `m`, `km`, `mm`, `dB`), and
/// common GIS punctuation.
fn glyph_5x7(ch: char) -> [u8; 7] {
    match ch {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        'a' => [0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F],
        'b' => [0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x1E],
        'c' => [0x00, 0x00, 0x0E, 0x10, 0x10, 0x11, 0x0E],
        'd' => [0x01, 0x01, 0x0F, 0x11, 0x11, 0x11, 0x0F],
        'e' => [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E],
        'f' => [0x06, 0x08, 0x1E, 0x08, 0x08, 0x08, 0x08],
        'g' => [0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x0E],
        'h' => [0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x11],
        'i' => [0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E],
        'j' => [0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C],
        'k' => [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12],
        'l' => [0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'm' => [0x00, 0x00, 0x1A, 0x15, 0x15, 0x15, 0x15],
        'n' => [0x00, 0x00, 0x1E, 0x11, 0x11, 0x11, 0x11],
        'o' => [0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E],
        'p' => [0x00, 0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10],
        'q' => [0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x01],
        'r' => [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10],
        's' => [0x00, 0x00, 0x0F, 0x10, 0x0E, 0x01, 0x1E],
        't' => [0x08, 0x08, 0x1E, 0x08, 0x08, 0x09, 0x06],
        'u' => [0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D],
        'v' => [0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'w' => [0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A],
        'x' => [0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11],
        'y' => [0x00, 0x00, 0x11, 0x11, 0x0F, 0x01, 0x0E],
        'z' => [0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        ',' => [0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x08],
        '-' | '−' | '—' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '+' => [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00],
        '=' => [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00],
        ':' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00],
        '/' => [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10],
        '[' => [0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E],
        ']' => [0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        '%' => [0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03],
        '_' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F],
        '\'' => [0x04, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00],
        '"' => [0x0A, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00],
        '°' => [0x06, 0x09, 0x06, 0x00, 0x00, 0x00, 0x00],
        '·' => [0x00, 0x00, 0x00, 0x06, 0x06, 0x00, 0x00],
        '²' => [0x0C, 0x02, 0x04, 0x0E, 0x00, 0x00, 0x00],
        '×' => [0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11],
        'µ' | 'μ' => [0x00, 0x00, 0x11, 0x11, 0x11, 0x1E, 0x10],
        _ => [0x00; 7],
    }
}

struct OverlayMetrics {
    ppp: f32,
    width_pt: f32,
    height_pt: f32,
    text_scale: i32,
    margin: i32,
    border_t: i32,
}

impl OverlayMetrics {
    fn from_canvas(width: u32, height: u32, pixels_per_point: f32) -> Self {
        let ppp = if pixels_per_point.is_finite() {
            pixels_per_point.clamp(0.75, 4.0)
        } else {
            1.0
        };
        Self {
            ppp,
            width_pt: (width as f32) / ppp,
            height_pt: (height as f32) / ppp,
            text_scale: ppp.round().clamp(1.0, 3.0) as i32,
            margin: (16.0 * ppp).round() as i32,
            border_t: (1.0 * ppp).round().max(1.0) as i32,
        }
    }
}

fn draw_caption_badge(
    canvas: &mut Canvas<'_>,
    spec: &CartographicOverlaySpec,
    m: &OverlayMetrics,
) -> i32 {
    let Some(caption) = spec
        .caption
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return m.margin;
    };
    if m.width_pt < 180.0 || m.height_pt < 90.0 {
        return m.margin;
    }
    let pad_x = (8.0 * m.ppp).round() as i32;
    let pad_y = (6.0 * m.ppp).round() as i32;
    let north_reserve = if spec.include_north_arrow {
        (56.0 * m.ppp).round() as i32 + m.margin
    } else {
        0
    };
    let max_badge_w = canvas.width - m.margin * 2 - north_reserve;
    let min_badge_w = 4 * 6 * m.text_scale + pad_x * 2;
    if max_badge_w < min_badge_w {
        return m.margin;
    }
    let max_chars = ((max_badge_w - pad_x * 2) / (6 * m.text_scale).max(1)).max(1) as usize;
    let clipped: String = caption.chars().take(max_chars).collect();
    if !clipped
        .chars()
        .any(|c| glyph_5x7(c).iter().any(|&b| b != 0))
    {
        return m.margin;
    }
    let tw = text_width_px(&clipped, m.text_scale);
    let box_w = tw + pad_x * 2;
    let box_h = 7 * m.text_scale + pad_y * 2;
    canvas.fill_rect(m.margin, m.margin, box_w, box_h, BG_CARD);
    canvas.stroke_rect(m.margin, m.margin, box_w, box_h, m.border_t, BORDER_SUBTLE);
    canvas.draw_text(
        m.margin + pad_x,
        m.margin + pad_y,
        &clipped,
        m.text_scale,
        INK_PRIMARY,
    );
    m.margin + box_h + (8.0 * m.ppp).round() as i32
}

fn draw_scale_bar(
    canvas: &mut Canvas<'_>,
    spec: &CartographicOverlaySpec,
    m: &OverlayMetrics,
    top_left_bottom: i32,
) -> Option<i32> {
    if !spec.include_scale_bar
        || !spec.m_per_pt.is_finite()
        || spec.m_per_pt <= 0.0
        || m.width_pt < 160.0
        || m.height_pt < 100.0
    {
        return None;
    }
    let max_w_pt = (m.width_pt * 0.35).clamp(48.0, 160.0);
    if spec.m_per_pt * f64::from(max_w_pt) > 10_000_000.0 {
        return None;
    }
    let (label, bar_w_pt) = compute_scale_bar(spec.m_per_pt, max_w_pt);
    let bar_w_px = (bar_w_pt * m.ppp).round().max(8.0) as i32;
    let tw = text_width_px(&label, m.text_scale);
    let pad_x = (10.0 * m.ppp).round() as i32;
    let content_w = bar_w_px.max(tw);
    let box_w = (content_w + pad_x * 2).min(canvas.width - m.margin * 2);
    let box_h = (34.0 * m.ppp).round() as i32;
    let bx = m.margin;
    let by = canvas.height - m.margin - box_h;

    if by < top_left_bottom || box_w < bar_w_px + pad_x * 2 {
        return None;
    }
    canvas.fill_rect(bx, by, box_w, box_h, BG_CARD);
    canvas.stroke_rect(bx, by, box_w, box_h, m.border_t, BORDER_SUBTLE);

    let tick_t = (2.0 * m.ppp).round().max(1.0) as i32;
    let tick_h = (6.0 * m.ppp).round().max(3.0) as i32;
    let y_bar = by + box_h - (9.0 * m.ppp).round() as i32;
    let x0 = bx + (box_w - bar_w_px) / 2;
    canvas.fill_rect(x0, y_bar, bar_w_px, tick_t, INK_PRIMARY);
    canvas.fill_rect(x0, y_bar - tick_h + tick_t, tick_t, tick_h, INK_PRIMARY);
    canvas.fill_rect(
        x0 + bar_w_px - tick_t,
        y_bar - tick_h + tick_t,
        tick_t,
        tick_h,
        INK_PRIMARY,
    );

    let tx = bx + (box_w - tw) / 2;
    let ty = by + (7.0 * m.ppp).round() as i32;
    canvas.draw_text(tx, ty, &label, m.text_scale, INK_PRIMARY);
    Some(by)
}

fn draw_colorbar(
    canvas: &mut Canvas<'_>,
    cb: &ColorbarInfo,
    m: &OverlayMetrics,
    top_left_bottom: i32,
    scale_bar_top: Option<i32>,
) {
    if m.width_pt < 240.0 || m.height_pt < 160.0 {
        return;
    }
    let desired_w = ((m.width_pt * 0.28).clamp(180.0, 260.0) * m.ppp).round() as i32;
    let box_w = desired_w.min(canvas.width - m.margin * 2);
    let box_h = (56.0 * m.ppp).round() as i32;
    let bx = m.margin;
    let bottom_y = scale_bar_top.map_or(canvas.height - m.margin, |top| {
        top - (8.0 * m.ppp).round() as i32
    });
    let by = bottom_y - box_h;
    if by < top_left_bottom || box_w < (80.0 * m.ppp).round() as i32 {
        return;
    }

    canvas.fill_rect(bx, by, box_w, box_h, BG_CARD);
    canvas.stroke_rect(bx, by, box_w, box_h, m.border_t, BORDER_SUBTLE);

    let pad_x = (8.0 * m.ppp).round() as i32;
    let unit_suffix = if cb.log10_db {
        " [dB]".to_owned()
    } else if let Some(ref u) = cb.units {
        format!(" [{u}]")
    } else {
        String::new()
    };
    let max_chars = ((box_w - pad_x * 2) / (6 * m.text_scale).max(1)).max(4) as usize;
    let title_full = format!("{}{unit_suffix}", cb.title);
    let title: String = title_full.chars().take(max_chars).collect();
    canvas.draw_text(
        bx + pad_x,
        by + (6.0 * m.ppp).round() as i32,
        &title,
        m.text_scale,
        INK_PRIMARY,
    );

    let bar_x = bx + pad_x;
    let bar_y = by + (22.0 * m.ppp).round() as i32;
    let bar_w = (box_w - pad_x * 2).max(16);
    let bar_h = (12.0 * m.ppp).round().max(4.0) as i32;
    for dx in 0..bar_w {
        let frac = (dx as f32) / ((bar_w - 1).max(1) as f32);
        let rgb = sample_colormap_srgb(cb.colormap, frac, cb.invert_colormap);
        canvas.fill_rect(bar_x + dx, bar_y, 1, bar_h, [rgb[0], rgb[1], rgb[2], 255]);
    }
    canvas.stroke_rect(bar_x, bar_y, bar_w, bar_h, m.border_t, BORDER_SUBTLE);

    let min_s = format::format_scalar(cb.min);
    let max_s = format::format_scalar(cb.max);
    let tick_y = bar_y + bar_h + (5.0 * m.ppp).round() as i32;
    let min_w = text_width_px(&min_s, m.text_scale);
    let max_w = text_width_px(&max_s, m.text_scale);
    canvas.draw_text(bar_x, tick_y, &min_s, m.text_scale, INK_PRIMARY);
    if min_w + max_w + 4 * m.text_scale <= bar_w {
        canvas.draw_text(
            bar_x + bar_w - max_w,
            tick_y,
            &max_s,
            m.text_scale,
            INK_PRIMARY,
        );
    }
}

fn draw_north_arrow(canvas: &mut Canvas<'_>, heading_deg: f64, m: &OverlayMetrics) {
    if m.width_pt < 120.0 || m.height_pt < 120.0 {
        return;
    }
    let radius = 24.0 * m.ppp;
    let cx = (canvas.width as f32) - (m.margin as f32) - radius;
    let cy = (m.margin as f32) + radius;
    canvas.fill_circle(cx, cy, radius, BG_CARD);
    canvas.stroke_circle(cx, cy, radius, (1.2 * m.ppp).max(1.0), BORDER_SUBTLE);

    let heading = if heading_deg.is_finite() {
        heading_deg
    } else {
        0.0
    };
    let angle = -(heading as f32).to_radians();
    let (sin_a, cos_a) = angle.sin_cos();
    let rot = |dx: f32, dy: f32| -> [f32; 2] {
        [cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a]
    };

    let tip_n = rot(0.0, -radius * 0.48);
    let tip_s = rot(0.0, radius * 0.52);
    let left = rot(-4.5 * m.ppp, 0.0);
    let right = rot(4.5 * m.ppp, 0.0);
    canvas.fill_triangle(tip_n, right, left, NEEDLE_NORTH);
    canvas.fill_triangle(tip_s, left, right, NEEDLE_SOUTH);

    let n_pos = rot(0.0, -radius * 0.72);
    let nw = text_width_px("N", m.text_scale);
    let nh = 7 * m.text_scale;
    canvas.draw_text(
        n_pos[0].round() as i32 - nw / 2,
        n_pos[1].round() as i32 - nh / 2,
        "N",
        m.text_scale,
        INK_PRIMARY,
    );
}

/// Composite optional cartographic overlays (metric scale bar, rotated North
/// arrow, scientific colorbar, and acquisition date/caption pill) directly onto
/// an sRGB RGBA8 buffer off the UI thread.
pub(crate) fn composite_cartographic_overlays(
    width: u32,
    height: u32,
    rgba: &mut [u8],
    spec: &CartographicOverlaySpec,
) {
    let Some(mut canvas) = Canvas::new(width, height, rgba) else {
        return;
    };
    let metrics = OverlayMetrics::from_canvas(width, height, spec.pixels_per_point);
    // Place the top-left caption first so bottom-left overlays never collide on short viewports.
    let top_left_bottom = draw_caption_badge(&mut canvas, spec, &metrics);
    let scale_bar_top = draw_scale_bar(&mut canvas, spec, &metrics, top_left_bottom);
    if spec.include_colorbar
        && let Some(cb) = &spec.colorbar
    {
        draw_colorbar(&mut canvas, cb, &metrics, top_left_bottom, scale_bar_top);
    }
    if spec.include_north_arrow {
        draw_north_arrow(&mut canvas, spec.heading_deg, &metrics);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colormap::ColormapId;

    #[test]
    fn cartographic_compositor_renders_overlays_and_encodes_valid_png() {
        let width = 320;
        let height = 240;
        let mut rgba = vec![0u8; (width * height * 4) as usize];
        for px in rgba.chunks_exact_mut(4) {
            px.copy_from_slice(&[20, 40, 60, 255]);
        }

        let spec = CartographicOverlaySpec {
            pixels_per_point: 1.0,
            m_per_pt: 10.0,
            heading_deg: 25.0,
            include_scale_bar: true,
            include_north_arrow: true,
            include_colorbar: true,
            colorbar: Some(ColorbarInfo {
                title: "NDVI".to_owned(),
                min: -0.2,
                max: 0.85,
                colormap: ColormapId::RdYlGn,
                invert_colormap: false,
                units: None,
                log10_db: false,
            }),
            caption: Some("2026-11-27 UTC".to_owned()),
        };

        composite_cartographic_overlays(width, height, &mut rgba, &spec);

        let modified_pixels = rgba
            .chunks_exact(4)
            .filter(|px| *px != [20, 40, 60, 255])
            .count();
        assert!(
            modified_pixels > 4_000,
            "expected overlays to paint > 4000 pixels, got {modified_pixels}"
        );

        let png_bytes = veldmap_raster::encode_png_rgba8(width, height, &rgba)
            .expect("encode cartographic png");
        assert!(veldmap_raster::has_png_magic(&png_bytes));

        // Non-finite metrics and compact 2x HiDPI viewports must never panic or
        // paint out-of-bounds.
        let mut hidpi_rgba = vec![0u8; (300 * 270 * 4) as usize];
        let edge_spec = CartographicOverlaySpec {
            pixels_per_point: f32::NAN,
            m_per_pt: 25_000.0,
            heading_deg: f64::NAN,
            include_scale_bar: true,
            include_north_arrow: true,
            include_colorbar: true,
            colorbar: spec.colorbar,
            caption: Some("2026-11-28 · 500 m".to_owned()),
        };
        composite_cartographic_overlays(300, 270, &mut hidpi_rgba, &edge_spec);
        assert!(glyph_5x7('m').iter().any(|&b| b != 0));
        assert!(glyph_5x7('m') != glyph_5x7('M'));
    }
}
