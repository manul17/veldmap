//! Typed TIFF IFD tag decoders over [`IfdDirectory`] and [`SinglePassIfdReader`].

use crate::RasterError;

use super::wire::{IfdDirectory, SinglePassIfdReader, TiffByteOrder, field_type_size};

fn unsigned_elem_size(tag: u16, field_type: u16) -> Result<usize, RasterError> {
    if !matches!(field_type, 1 | 3 | 4 | 13 | 16 | 18) {
        return Err(RasterError::Decode(format!(
            "TIFF tag {tag} has unexpected non-unsigned field type {field_type}"
        )));
    }
    let size = field_type_size(field_type).ok_or_else(|| {
        RasterError::Decode(format!(
            "TIFF tag {tag} has unexpected non-unsigned field type {field_type}"
        ))
    })?;
    Ok(size as usize)
}

fn for_each_unsigned_value(
    order: TiffByteOrder,
    field_type: u16,
    bytes: &[u8],
    count: usize,
    mut visit: impl FnMut(u64) -> Result<(), RasterError>,
) -> Result<(), RasterError> {
    match field_type {
        1 => {
            for &b in bytes.iter().take(count) {
                visit(u64::from(b))?;
            }
        }
        3 => {
            for chunk in bytes.chunks_exact(2).take(count) {
                visit(u64::from(order.read_u16(chunk)))?;
            }
        }
        4 | 13 => {
            for chunk in bytes.chunks_exact(4).take(count) {
                visit(u64::from(order.read_u32(chunk)))?;
            }
        }
        16 | 18 => {
            for chunk in bytes.chunks_exact(8).take(count) {
                visit(order.read_u64(chunk))?;
            }
        }
        _ => unreachable!("field_type validated by unsigned_elem_size"),
    }
    Ok(())
}

impl IfdDirectory {
    fn with_unsigned_tag<R>(
        &self,
        reader: &SinglePassIfdReader,
        tag: u16,
        f: impl FnOnce(u16, &[u8], usize) -> Result<R, RasterError>,
    ) -> Result<Option<R>, RasterError> {
        let Some(entry) = self.entry(tag) else {
            return Ok(None);
        };
        let field_type = entry.field_type();
        let elem_size = unsigned_elem_size(tag, field_type)?;
        let bytes = self.resolve_bytes(&entry, reader)?;
        let raw_count = usize::try_from(entry.count()).map_err(|_| {
            RasterError::DecoderLimitsExceeded(format!("TIFF tag {tag} count overflow"))
        })?;
        let count = raw_count.min(bytes.len() / elem_size);
        f(field_type, bytes, count).map(Some)
    }

    fn read_u64_scalar_strict(
        &self,
        reader: &SinglePassIfdReader,
        tag: u16,
    ) -> Result<Option<u64>, RasterError> {
        let order = self.byte_order();
        self.with_unsigned_tag(reader, tag, |field_type, bytes, count| {
            if count != 1 {
                return Err(RasterError::Decode(format!(
                    "TIFF tag {tag} expected 1 unsigned value, got {count}"
                )));
            }
            let mut scalar = None;
            for_each_unsigned_value(order, field_type, bytes, 1, |v| {
                scalar = Some(v);
                Ok(())
            })?;
            scalar.ok_or_else(|| {
                RasterError::Decode(format!("TIFF tag {tag} expected 1 unsigned value, got 0"))
            })
        })
    }

    pub(super) fn read_u16_strict(
        &self,
        reader: &SinglePassIfdReader,
        tag: u16,
    ) -> Result<Option<u16>, RasterError> {
        let Some(val) = self.read_u64_scalar_strict(reader, tag)? else {
            return Ok(None);
        };
        let v = u16::try_from(val).map_err(|_| {
            RasterError::Decode(format!("TIFF tag {tag} value {val} overflows u16"))
        })?;
        Ok(Some(v))
    }

    pub(super) fn read_u32_strict(
        &self,
        reader: &SinglePassIfdReader,
        tag: u16,
    ) -> Result<Option<u32>, RasterError> {
        let Some(val) = self.read_u64_scalar_strict(reader, tag)? else {
            return Ok(None);
        };
        let v = u32::try_from(val).map_err(|_| {
            RasterError::Decode(format!("TIFF tag {tag} value {val} overflows u32"))
        })?;
        Ok(Some(v))
    }

    pub(in crate::tiff) fn read_u32_opt(
        &self,
        reader: &SinglePassIfdReader,
        tag: u16,
    ) -> Option<u32> {
        self.read_u32_strict(reader, tag).ok().flatten()
    }

    pub(super) fn read_u16_vec_strict(
        &self,
        reader: &SinglePassIfdReader,
        tag: u16,
    ) -> Result<Option<Vec<u16>>, RasterError> {
        let order = self.byte_order();
        self.with_unsigned_tag(reader, tag, |field_type, bytes, count| {
            let mut out = Vec::with_capacity(count);
            for_each_unsigned_value(order, field_type, bytes, count, |v| {
                out.push(u16::try_from(v).map_err(|_| {
                    RasterError::Decode(format!("TIFF tag {tag} value {v} overflows u16"))
                })?);
                Ok(())
            })?;
            Ok(out)
        })
    }

    pub(in crate::tiff) fn read_u16_vec_opt(
        &self,
        reader: &SinglePassIfdReader,
        tag: u16,
    ) -> Vec<u16> {
        self.read_u16_vec_strict(reader, tag)
            .ok()
            .flatten()
            .unwrap_or_default()
    }

    pub(super) fn read_u64_vec_strict(
        &self,
        reader: &SinglePassIfdReader,
        tag: u16,
    ) -> Result<Option<Vec<u64>>, RasterError> {
        let order = self.byte_order();
        self.with_unsigned_tag(reader, tag, |field_type, bytes, count| {
            let mut out = Vec::with_capacity(count);
            for_each_unsigned_value(order, field_type, bytes, count, |v| {
                out.push(v);
                Ok(())
            })?;
            Ok(out)
        })
    }

    pub(super) fn read_u8_vec_strict(
        &self,
        reader: &SinglePassIfdReader,
        tag: u16,
    ) -> Result<Option<Vec<u8>>, RasterError> {
        let Some(entry) = self.entry(tag) else {
            return Ok(None);
        };
        let field_type = entry.field_type();
        if !matches!(field_type, 1 | 7) {
            return Err(RasterError::Decode(format!(
                "TIFF tag {tag} expected BYTE/UNDEFINED, got field type {field_type}"
            )));
        }
        let bytes = self.resolve_bytes(&entry, reader)?;
        Ok(Some(bytes.to_vec()))
    }

    pub(super) fn read_numeric_f64_vec_opt(
        &self,
        reader: &SinglePassIfdReader,
        tag: u16,
    ) -> Option<Vec<f64>> {
        let entry = self.entry(tag)?;
        let field_type = entry.field_type();
        if matches!(field_type, 2 | 7) {
            return None;
        }
        let elem_size = field_type_size(field_type)? as usize;
        let bytes = self.resolve_bytes(&entry, reader).ok()?;
        let count = usize::try_from(entry.count())
            .ok()?
            .min(bytes.len() / elem_size);
        let order = self.byte_order();
        let mut out = Vec::with_capacity(count);
        match field_type {
            1 => out.extend(bytes.iter().take(count).map(|&b| f64::from(b))),
            6 => out.extend(bytes.iter().take(count).map(|&b| f64::from(b as i8))),
            3 => out.extend(
                bytes
                    .chunks_exact(2)
                    .take(count)
                    .map(|c| f64::from(order.read_u16(c))),
            ),
            8 => out.extend(
                bytes
                    .chunks_exact(2)
                    .take(count)
                    .map(|c| f64::from(order.read_i16(c))),
            ),
            4 | 13 => out.extend(
                bytes
                    .chunks_exact(4)
                    .take(count)
                    .map(|c| f64::from(order.read_u32(c))),
            ),
            9 => out.extend(
                bytes
                    .chunks_exact(4)
                    .take(count)
                    .map(|c| f64::from(order.read_i32(c))),
            ),
            16 | 18 => out.extend(
                bytes
                    .chunks_exact(8)
                    .take(count)
                    .map(|c| order.read_u64(c) as f64),
            ),
            17 => out.extend(
                bytes
                    .chunks_exact(8)
                    .take(count)
                    .map(|c| order.read_i64(c) as f64),
            ),
            5 => {
                for c in bytes.chunks_exact(8).take(count) {
                    let n = order.read_u32(&c[0..4]);
                    let d = order.read_u32(&c[4..8]);
                    if d != 0 {
                        out.push(f64::from(n) / f64::from(d));
                    }
                }
            }
            10 => {
                for c in bytes.chunks_exact(8).take(count) {
                    let n = order.read_i32(&c[0..4]);
                    let d = order.read_i32(&c[4..8]);
                    if d != 0 {
                        out.push(f64::from(n) / f64::from(d));
                    }
                }
            }
            11 => out.extend(
                bytes
                    .chunks_exact(4)
                    .take(count)
                    .map(|c| f64::from(order.read_f32(c))),
            ),
            12 => out.extend(bytes.chunks_exact(8).take(count).map(|c| order.read_f64(c))),
            _ => return None,
        }
        Some(out)
    }

    pub(in crate::tiff) fn read_f64_vec_opt(
        &self,
        reader: &SinglePassIfdReader,
        tag: u16,
    ) -> Vec<f64> {
        let Some(entry) = self.entry(tag) else {
            return Vec::new();
        };
        if !matches!(entry.field_type(), 11 | 12) {
            return Vec::new();
        }
        self.read_numeric_f64_vec_opt(reader, tag)
            .unwrap_or_default()
    }

    pub(in crate::tiff) fn read_ascii_bytes_opt(
        &self,
        reader: &SinglePassIfdReader,
        tag: u16,
    ) -> Vec<u8> {
        let Some(entry) = self.entry(tag) else {
            return Vec::new();
        };
        if !matches!(entry.field_type(), 2 | 1 | 7) {
            return Vec::new();
        }
        let Ok(bytes) = self.resolve_bytes(&entry, reader) else {
            return Vec::new();
        };
        // Truncate TIFF ASCII payloads at the first NUL byte.
        bytes.split(|&b| b == 0).next().unwrap_or(&[]).to_vec()
    }

    pub(in crate::tiff) fn read_ascii_opt(&self, reader: &SinglePassIfdReader, tag: u16) -> String {
        let raw = self.read_ascii_bytes_opt(reader, tag);
        if raw.is_empty() {
            return String::new();
        }
        String::from_utf8_lossy(&raw).into_owned()
    }
}
