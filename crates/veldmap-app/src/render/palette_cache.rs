//! Cached 256-entry linear-light GPU palette LUT uniform buffers and bind
//! groups shared across scientific (`R32Float`), categorical
//! (`R8Uint`/`R16Uint`), and Band Math composite overlay draws.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use veldmap_core::tile::SourceId;
use wgpu::util::DeviceExt;

use crate::colormap::{
    CategoricalPaletteId, ColormapId, categorical_lut_linear, scientific_lut_linear,
};

use super::snapshot::DrawOverlayTrack;
use super::uniforms::PaletteUniform;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PaletteKey {
    Scientific(ColormapId),
    CategoricalPreset(CategoricalPaletteId),
    /// Keyed by `(source_id, arc_ptr)` so two layers opening the same palette
    /// file independently never thrash each other's cached GPU palette buffer.
    CustomSource(SourceId, usize),
}

fn custom_source_key(source: SourceId, cmap: &Arc<[[u8; 4]]>) -> PaletteKey {
    PaletteKey::CustomSource(source, Arc::as_ptr(cmap) as *const () as usize)
}

fn palette_key(track: &DrawOverlayTrack) -> Option<PaletteKey> {
    match track {
        DrawOverlayTrack::Display => None,
        DrawOverlayTrack::Scientific { colormap, .. } => Some(PaletteKey::Scientific(*colormap)),
        DrawOverlayTrack::Categorical {
            source,
            palette,
            custom_color_map,
            ..
        } => {
            if *palette == CategoricalPaletteId::SourceColorMap
                && let Some(cmap) = custom_color_map
            {
                Some(custom_source_key(*source, cmap))
            } else {
                Some(PaletteKey::CategoricalPreset(*palette))
            }
        }
    }
}

struct CachedPalette {
    _buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    custom_weak: Option<Weak<[[u8; 4]]>>,
}

impl CachedPalette {
    fn is_live(&self) -> bool {
        self.custom_weak
            .as_ref()
            .is_none_or(|w| w.strong_count() > 0)
    }
}

/// Lazily populated cache of GPU palette LUT uniform bind groups with
/// `Weak`-guarded automatic sweeping for source-specific TIFF `ColorMap` tables.
#[derive(Default)]
pub(super) struct PaletteCache {
    entries: HashMap<PaletteKey, CachedPalette>,
}

impl PaletteCache {
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Drop custom TIFF `ColorMap` palette entries whose owning source dropped.
    pub(super) fn sweep_stale(&mut self) {
        self.entries.retain(|_, p| p.is_live());
    }

    pub(super) fn ensure(
        &mut self,
        device: &wgpu::Device,
        palette_layout: &wgpu::BindGroupLayout,
        track: &DrawOverlayTrack,
    ) {
        let Some(key) = palette_key(track) else {
            return;
        };
        if self.entries.get(&key).is_some_and(CachedPalette::is_live) {
            return;
        }

        let (lut, custom_weak) = match track {
            DrawOverlayTrack::Display => return,
            DrawOverlayTrack::Scientific { colormap, .. } => {
                (scientific_lut_linear(*colormap), None)
            }
            DrawOverlayTrack::Categorical {
                palette,
                custom_color_map,
                ..
            } => {
                if *palette == CategoricalPaletteId::SourceColorMap
                    && let Some(cmap) = custom_color_map
                {
                    (
                        categorical_lut_linear(*palette, Some(cmap.as_ref())),
                        Some(Arc::downgrade(cmap)),
                    )
                } else {
                    (categorical_lut_linear(*palette, None), None)
                }
            }
        };

        self.insert_lut(
            device,
            palette_layout,
            key,
            lut,
            custom_weak,
            "overlay palette LUT",
        );
    }

    pub(super) fn ensure_colormap(
        &mut self,
        device: &wgpu::Device,
        palette_layout: &wgpu::BindGroupLayout,
        colormap: ColormapId,
    ) {
        let key = PaletteKey::Scientific(colormap);
        if self.entries.get(&key).is_some_and(CachedPalette::is_live) {
            return;
        }
        self.insert_lut(
            device,
            palette_layout,
            key,
            scientific_lut_linear(colormap),
            None,
            "band-math palette LUT",
        );
    }

    fn insert_lut(
        &mut self,
        device: &wgpu::Device,
        palette_layout: &wgpu::BindGroupLayout,
        key: PaletteKey,
        entries: [[f32; 4]; 256],
        custom_weak: Option<Weak<[[u8; 4]]>>,
        label: &str,
    ) {
        let uniform = PaletteUniform { entries };
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: palette_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        self.entries.insert(
            key,
            CachedPalette {
                _buffer: buffer,
                bind_group,
                custom_weak,
            },
        );
    }

    pub(super) fn colormap_bind_group(&self, colormap: ColormapId) -> Option<&wgpu::BindGroup> {
        let key = PaletteKey::Scientific(colormap);
        self.entries
            .get(&key)
            .filter(|entry| entry.is_live())
            .map(|entry| &entry.bind_group)
    }

    pub(super) fn bind_group(&self, track: &DrawOverlayTrack) -> Option<&wgpu::BindGroup> {
        let key = palette_key(track)?;
        self.entries
            .get(&key)
            .filter(|entry| entry.is_live())
            .map(|entry| &entry.bind_group)
    }
}
