//! Cached 256-entry linear-light GPU palette LUT uniform buffers and bind
//! groups shared across scientific (`R32Float`) and categorical
//! (`R8Uint`/`R16Uint`) overlay draws.

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
        self.entries
            .retain(|_, p| p.custom_weak.as_ref().is_none_or(|w| w.strong_count() > 0));
    }

    pub(super) fn ensure(
        &mut self,
        device: &wgpu::Device,
        palette_layout: &wgpu::BindGroupLayout,
        track: &DrawOverlayTrack,
    ) {
        let (key, lut, custom_weak) = match track {
            DrawOverlayTrack::Display => return,
            DrawOverlayTrack::Scientific { colormap, .. } => {
                let key = PaletteKey::Scientific(*colormap);
                if self.entries.contains_key(&key) {
                    return;
                }
                (key, scientific_lut_linear(*colormap), None)
            }
            DrawOverlayTrack::Categorical {
                source,
                palette,
                custom_color_map,
                ..
            } => {
                if *palette == CategoricalPaletteId::SourceColorMap
                    && let Some(cmap) = custom_color_map
                {
                    let key = custom_source_key(*source, cmap);
                    if self.entries.contains_key(&key) {
                        return;
                    }
                    (
                        key,
                        categorical_lut_linear(*palette, Some(cmap.as_ref())),
                        Some(Arc::downgrade(cmap)),
                    )
                } else {
                    let key = PaletteKey::CategoricalPreset(*palette);
                    if self.entries.contains_key(&key) {
                        return;
                    }
                    (key, categorical_lut_linear(*palette, None), None)
                }
            }
        };

        let uniform = PaletteUniform { entries: lut };
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("overlay palette LUT uniform buffer"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay palette LUT bind group"),
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

    pub(super) fn bind_group(&self, track: &DrawOverlayTrack) -> Option<&wgpu::BindGroup> {
        let key = palette_key(track)?;
        self.entries.get(&key).map(|p| &p.bind_group)
    }
}
