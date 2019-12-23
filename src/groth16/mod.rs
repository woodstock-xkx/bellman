//! The [Groth16] proving system.
//!
//! [Groth16]: https://eprint.iacr.org/2016/260

use groupy::{CurveAffine, EncodedPoint};
use paired::{Engine, PairingCurveAffine};

use crate::multiexp::SourceBuilder;
use crate::SynthesisError;

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use memmap::{Mmap, MmapOptions};
use std::fs::File;
use std::io::{self, Read, Write};
use std::marker::PhantomData;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(test)]
mod tests;

mod generator;
mod prover;
mod verifier;

pub use self::generator::*;
pub use self::prover::*;
pub use self::verifier::*;

#[derive(Clone)]
pub struct Proof<E: Engine> {
    pub a: E::G1Affine,
    pub b: E::G2Affine,
    pub c: E::G1Affine,
}

impl<E: Engine> PartialEq for Proof<E> {
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b && self.c == other.c
    }
}

impl<E: Engine> Proof<E> {
    pub fn write<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writer.write_all(self.a.into_compressed().as_ref())?;
        writer.write_all(self.b.into_compressed().as_ref())?;
        writer.write_all(self.c.into_compressed().as_ref())?;

        Ok(())
    }

    pub fn read<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut g1_repr = <E::G1Affine as CurveAffine>::Compressed::empty();
        let mut g2_repr = <E::G2Affine as CurveAffine>::Compressed::empty();

        reader.read_exact(g1_repr.as_mut())?;
        let a = g1_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            .and_then(|e| {
                if e.is_zero() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })?;

        reader.read_exact(g2_repr.as_mut())?;
        let b = g2_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            .and_then(|e| {
                if e.is_zero() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })?;

        reader.read_exact(g1_repr.as_mut())?;
        let c = g1_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            .and_then(|e| {
                if e.is_zero() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })?;

        Ok(Proof { a, b, c })
    }
}

#[derive(Clone)]
pub struct VerifyingKey<E: Engine> {
    // alpha in g1 for verifying and for creating A/C elements of
    // proof. Never the point at infinity.
    pub alpha_g1: E::G1Affine,

    // beta in g1 and g2 for verifying and for creating B/C elements
    // of proof. Never the point at infinity.
    pub beta_g1: E::G1Affine,
    pub beta_g2: E::G2Affine,

    // gamma in g2 for verifying. Never the point at infinity.
    pub gamma_g2: E::G2Affine,

    // delta in g1/g2 for verifying and proving, essentially the magic
    // trapdoor that forces the prover to evaluate the C element of the
    // proof with only components from the CRS. Never the point at
    // infinity.
    pub delta_g1: E::G1Affine,
    pub delta_g2: E::G2Affine,

    // Elements of the form (beta * u_i(tau) + alpha v_i(tau) + w_i(tau)) / gamma
    // for all public inputs. Because all public inputs have a dummy constraint,
    // this is the same size as the number of inputs, and never contains points
    // at infinity.
    pub ic: Vec<E::G1Affine>,
}

impl<E: Engine> PartialEq for VerifyingKey<E> {
    fn eq(&self, other: &Self) -> bool {
        self.alpha_g1 == other.alpha_g1
            && self.beta_g1 == other.beta_g1
            && self.beta_g2 == other.beta_g2
            && self.gamma_g2 == other.gamma_g2
            && self.delta_g1 == other.delta_g1
            && self.delta_g2 == other.delta_g2
            && self.ic == other.ic
    }
}

impl<E: Engine> VerifyingKey<E> {
    pub fn write<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writer.write_all(self.alpha_g1.into_uncompressed().as_ref())?;
        writer.write_all(self.beta_g1.into_uncompressed().as_ref())?;
        writer.write_all(self.beta_g2.into_uncompressed().as_ref())?;
        writer.write_all(self.gamma_g2.into_uncompressed().as_ref())?;
        writer.write_all(self.delta_g1.into_uncompressed().as_ref())?;
        writer.write_all(self.delta_g2.into_uncompressed().as_ref())?;
        writer.write_u32::<BigEndian>(self.ic.len() as u32)?;
        for ic in &self.ic {
            writer.write_all(ic.into_uncompressed().as_ref())?;
        }

        Ok(())
    }

    pub fn read<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut g1_repr = <E::G1Affine as CurveAffine>::Uncompressed::empty();
        let mut g2_repr = <E::G2Affine as CurveAffine>::Uncompressed::empty();

        reader.read_exact(g1_repr.as_mut())?;
        let alpha_g1 = g1_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        reader.read_exact(g1_repr.as_mut())?;
        let beta_g1 = g1_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        reader.read_exact(g2_repr.as_mut())?;
        let beta_g2 = g2_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        reader.read_exact(g2_repr.as_mut())?;
        let gamma_g2 = g2_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        reader.read_exact(g1_repr.as_mut())?;
        let delta_g1 = g1_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        reader.read_exact(g2_repr.as_mut())?;
        let delta_g2 = g2_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let ic_len = reader.read_u32::<BigEndian>()? as usize;

        let mut ic = vec![];

        for _ in 0..ic_len {
            reader.read_exact(g1_repr.as_mut())?;
            let g1 = g1_repr
                .into_affine()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
                .and_then(|e| {
                    if e.is_zero() {
                        Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "point at infinity",
                        ))
                    } else {
                        Ok(e)
                    }
                })?;

            ic.push(g1);
        }

        Ok(VerifyingKey {
            alpha_g1,
            beta_g1,
            beta_g2,
            gamma_g2,
            delta_g1,
            delta_g2,
            ic,
        })
    }

    // This method is provided as a proof of concept, but isn't used.
    // It's equivalent to the read method, in that it loads all
    // parameters to RAM.
    pub fn read_mmap(mmap: &Mmap, offset: &mut usize) -> io::Result<Self> {
        let read_g1 = |mmap: &Mmap, offset: usize| {
            let ptr = &mmap[offset..offset + std::mem::size_of::<E::G1Affine>()];
            let g1_repr: <E::G1Affine as CurveAffine>::Uncompressed = unsafe {
                *(ptr as *const [u8] as *const <E::G1Affine as CurveAffine>::Uncompressed)
            };

            (
                g1_repr,
                std::mem::size_of::<<E::G1Affine as CurveAffine>::Uncompressed>(),
            )
        };

        let read_g2 = |mmap: &Mmap, offset: usize| {
            let ptr = &mmap[offset..offset + std::mem::size_of::<E::G2Affine>()];
            let g2_repr: <E::G2Affine as CurveAffine>::Uncompressed = unsafe {
                *(ptr as *const [u8] as *const <E::G2Affine as CurveAffine>::Uncompressed)
            };

            (
                g2_repr,
                std::mem::size_of::<<E::G2Affine as CurveAffine>::Uncompressed>(),
            )
        };

        let (g1_repr, len) = read_g1(&mmap, *offset);
        *offset += len;
        let alpha_g1 = g1_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let (g1_repr, len) = read_g1(&mmap, *offset);
        *offset += len;
        let beta_g1 = g1_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let (g2_repr, len) = read_g2(&mmap, *offset);
        *offset += len;
        let beta_g2 = g2_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let (g2_repr, len) = read_g2(&mmap, *offset);
        *offset += len;
        let gamma_g2 = g2_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let (g1_repr, len) = read_g1(&mmap, *offset);
        *offset += len;
        let delta_g1 = g1_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let (g2_repr, len) = read_g2(&mmap, *offset);
        *offset += len;
        let delta_g2 = g2_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let u32_len = std::mem::size_of::<u32>();

        let mut raw_ic_len = &mmap[*offset..*offset + u32_len];
        let ic_len = raw_ic_len.read_u32::<BigEndian>()? as usize;
        *offset += u32_len;

        let mut ic = vec![];

        for _ in 0..ic_len {
            let (g1_repr, len) = read_g1(&mmap, *offset);
            *offset += len;
            let g1 = g1_repr
                .into_affine()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
                .and_then(|e| {
                    if e.is_zero() {
                        Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "point at infinity",
                        ))
                    } else {
                        Ok(e)
                    }
                })?;

            ic.push(g1);
        }

        Ok(VerifyingKey {
            alpha_g1,
            beta_g1,
            beta_g2,
            gamma_g2,
            delta_g1,
            delta_g2,
            ic,
        })
    }

    pub fn update_mapped_vk_offset(mmap: &Mmap, offset: &mut usize) -> io::Result<()> {
        *offset += 3 * std::mem::size_of::<<E::G1Affine as CurveAffine>::Uncompressed>();
        *offset += 3 * std::mem::size_of::<<E::G2Affine as CurveAffine>::Uncompressed>();

        let u32_len = std::mem::size_of::<u32>();
        let mut raw_ic_len = &mmap[*offset..*offset + u32_len];
        let ic_len = raw_ic_len.read_u32::<BigEndian>()? as usize;
        *offset += u32_len;
        *offset += ic_len * std::mem::size_of::<<E::G1Affine as CurveAffine>::Uncompressed>();

        Ok(())
    }
}

#[derive(Clone)]
pub struct Parameters<E: Engine> {
    pub vk: VerifyingKey<E>,

    // Elements of the form ((tau^i * t(tau)) / delta) for i between 0 and
    // m-2 inclusive. Never contains points at infinity.
    pub h: Arc<Vec<E::G1Affine>>,

    // Elements of the form (beta * u_i(tau) + alpha v_i(tau) + w_i(tau)) / delta
    // for all auxiliary inputs. Variables can never be unconstrained, so this
    // never contains points at infinity.
    pub l: Arc<Vec<E::G1Affine>>,

    // QAP "A" polynomials evaluated at tau in the Lagrange basis. Never contains
    // points at infinity: polynomials that evaluate to zero are omitted from
    // the CRS and the prover can deterministically skip their evaluation.
    pub a: Arc<Vec<E::G1Affine>>,

    // QAP "B" polynomials evaluated at tau in the Lagrange basis. Needed in
    // G1 and G2 for C/B queries, respectively. Never contains points at
    // infinity for the same reason as the "A" polynomials.
    pub b_g1: Arc<Vec<E::G1Affine>>,
    pub b_g2: Arc<Vec<E::G2Affine>>,
}

impl<E: Engine> PartialEq for Parameters<E> {
    fn eq(&self, other: &Self) -> bool {
        self.vk == other.vk
            && self.h == other.h
            && self.l == other.l
            && self.a == other.a
            && self.b_g1 == other.b_g1
            && self.b_g2 == other.b_g2
    }
}

impl<E: Engine> Parameters<E> {
    pub fn write<W: Write>(&self, mut writer: W) -> io::Result<()> {
        self.vk.write(&mut writer)?;

        writer.write_u32::<BigEndian>(self.h.len() as u32)?;
        for g in &self.h[..] {
            writer.write_all(g.into_uncompressed().as_ref())?;
        }

        writer.write_u32::<BigEndian>(self.l.len() as u32)?;
        for g in &self.l[..] {
            writer.write_all(g.into_uncompressed().as_ref())?;
        }

        writer.write_u32::<BigEndian>(self.a.len() as u32)?;
        for g in &self.a[..] {
            writer.write_all(g.into_uncompressed().as_ref())?;
        }

        writer.write_u32::<BigEndian>(self.b_g1.len() as u32)?;
        for g in &self.b_g1[..] {
            writer.write_all(g.into_uncompressed().as_ref())?;
        }

        writer.write_u32::<BigEndian>(self.b_g2.len() as u32)?;
        for g in &self.b_g2[..] {
            writer.write_all(g.into_uncompressed().as_ref())?;
        }

        Ok(())
    }

    // Quickly iterates through the parameter file, recording all
    // parameter offsets and caches the verifying key (vk) for quick
    // access via reference.
    pub fn build_mapped_parameters(
        param_file: PathBuf,
        checked: bool,
    ) -> io::Result<MappedParameters<E>> {
        let params = File::open(&param_file)?;
        let mmap = unsafe { MmapOptions::new().map(&params)? };

        let mut offset: usize = 0;
        let u32_len = std::mem::size_of::<u32>();
        let vk = VerifyingKey::<E>::read_mmap(&mmap, &mut offset)?;

        let mut h = vec![];
        let mut l = vec![];
        let mut a = vec![];
        let mut b_g1 = vec![];
        let mut b_g2 = vec![];

        {
            let mut raw_len = &mmap[offset..offset + u32_len];
            let len = raw_len.read_u32::<BigEndian>()? as usize;
            offset += u32_len;

            for _ in 0..len {
                h.push(Range {
                    start: offset,
                    end: offset + std::mem::size_of::<<E::G1Affine as CurveAffine>::Uncompressed>(),
                });
                offset += std::mem::size_of::<<E::G1Affine as CurveAffine>::Uncompressed>();
            }
        }

        {
            let mut raw_len = &mmap[offset..offset + u32_len];
            offset += u32_len;

            let len = raw_len.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                l.push(Range {
                    start: offset,
                    end: offset + std::mem::size_of::<<E::G1Affine as CurveAffine>::Uncompressed>(),
                });
                offset += std::mem::size_of::<<E::G1Affine as CurveAffine>::Uncompressed>();
            }
        }

        {
            let mut raw_len = &mmap[offset..offset + u32_len];
            offset += u32_len;

            let len = raw_len.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                a.push(Range {
                    start: offset,
                    end: offset + std::mem::size_of::<<E::G1Affine as CurveAffine>::Uncompressed>(),
                });
                offset += std::mem::size_of::<<E::G1Affine as CurveAffine>::Uncompressed>();
            }
        }

        {
            let mut raw_len = &mmap[offset..offset + u32_len];
            offset += u32_len;

            let len = raw_len.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                b_g1.push(Range {
                    start: offset,
                    end: offset + std::mem::size_of::<<E::G1Affine as CurveAffine>::Uncompressed>(),
                });
                offset += std::mem::size_of::<<E::G1Affine as CurveAffine>::Uncompressed>();
            }
        }

        {
            let mut raw_len = &mmap[offset..offset + u32_len];
            offset += u32_len;

            let len = raw_len.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                b_g2.push(Range {
                    start: offset,
                    end: offset + std::mem::size_of::<<E::G2Affine as CurveAffine>::Uncompressed>(),
                });
                offset += std::mem::size_of::<<E::G2Affine as CurveAffine>::Uncompressed>();
            }
        }

        Ok(MappedParameters {
            param_file,
            vk,
            h,
            l,
            a,
            b_g1,
            b_g2,
            checked,
            _e: Default::default(),
        })
    }

    pub fn read_mmap(mmap: &Mmap, checked: bool) -> io::Result<Self> {
        let read_g1 = |mmap: &Mmap, offset: usize| -> io::Result<E::G1Affine> {
            let ptr = &mmap[offset
                ..offset + std::mem::size_of::<<E::G1Affine as CurveAffine>::Uncompressed>()];
            let repr: <E::G1Affine as CurveAffine>::Uncompressed = unsafe {
                *(ptr as *const [u8] as *const <E::G1Affine as CurveAffine>::Uncompressed)
            };

            if checked {
                repr.into_affine()
            } else {
                repr.into_affine_unchecked()
            }
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            .and_then(|e| {
                if e.is_zero() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })
        };

        let read_g2 = |mmap: &Mmap, offset: usize| -> io::Result<E::G2Affine> {
            let ptr = &mmap[offset
                ..offset + std::mem::size_of::<<E::G2Affine as CurveAffine>::Uncompressed>()];
            let repr: <E::G2Affine as CurveAffine>::Uncompressed = unsafe {
                *(ptr as *const [u8] as *const <E::G2Affine as CurveAffine>::Uncompressed)
            };

            if checked {
                repr.into_affine()
            } else {
                repr.into_affine_unchecked()
            }
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            .and_then(|e| {
                if e.is_zero() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })
        };

        let mut offset: usize = 0;
        let u32_len = std::mem::size_of::<u32>();
        let vk = VerifyingKey::<E>::read_mmap(&mmap, &mut offset)?;

        let mut h = vec![];
        let mut l = vec![];
        let mut a = vec![];
        let mut b_g1 = vec![];
        let mut b_g2 = vec![];

        {
            let mut raw_len = &mmap[offset..offset + u32_len];
            let len = raw_len.read_u32::<BigEndian>()? as usize;
            offset += u32_len;

            for _ in 0..len {
                h.push(read_g1(&mmap, offset)?);
                offset += std::mem::size_of::<<E::G1Affine as CurveAffine>::Uncompressed>();
            }
        }

        {
            let mut raw_len = &mmap[offset..offset + u32_len];
            offset += u32_len;

            let len = raw_len.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                l.push(read_g1(&mmap, offset)?);
                offset += std::mem::size_of::<<E::G1Affine as CurveAffine>::Uncompressed>();
            }
        }

        {
            let mut raw_len = &mmap[offset..offset + u32_len];
            offset += u32_len;

            let len = raw_len.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                a.push(read_g1(&mmap, offset)?);
                offset += std::mem::size_of::<<E::G1Affine as CurveAffine>::Uncompressed>();
            }
        }

        {
            let mut raw_len = &mmap[offset..offset + u32_len];
            offset += u32_len;

            let len = raw_len.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                b_g1.push(read_g1(&mmap, offset)?);
                offset += std::mem::size_of::<<E::G1Affine as CurveAffine>::Uncompressed>();
            }
        }

        {
            let mut raw_len = &mmap[offset..offset + u32_len];
            offset += u32_len;

            let len = raw_len.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                b_g2.push(read_g2(&mmap, offset)?);
                offset += std::mem::size_of::<<E::G2Affine as CurveAffine>::Uncompressed>();
            }
        }

        Ok(Parameters {
            vk,
            h: Arc::new(h),
            l: Arc::new(l),
            a: Arc::new(a),
            b_g1: Arc::new(b_g1),
            b_g2: Arc::new(b_g2),
        })
    }

    pub fn read<R: Read>(mut reader: R, checked: bool) -> io::Result<Self> {
        let read_g1 = |reader: &mut R| -> io::Result<E::G1Affine> {
            let mut repr = <E::G1Affine as CurveAffine>::Uncompressed::empty();
            reader.read_exact(repr.as_mut())?;

            if checked {
                repr.into_affine()
            } else {
                repr.into_affine_unchecked()
            }
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            .and_then(|e| {
                if e.is_zero() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })
        };

        let read_g2 = |reader: &mut R| -> io::Result<E::G2Affine> {
            let mut repr = <E::G2Affine as CurveAffine>::Uncompressed::empty();
            reader.read_exact(repr.as_mut())?;

            if checked {
                repr.into_affine()
            } else {
                repr.into_affine_unchecked()
            }
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            .and_then(|e| {
                if e.is_zero() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })
        };

        let vk = VerifyingKey::<E>::read(&mut reader)?;

        let mut h = vec![];
        let mut l = vec![];
        let mut a = vec![];
        let mut b_g1 = vec![];
        let mut b_g2 = vec![];

        {
            let len = reader.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                h.push(read_g1(&mut reader)?);
            }
        }

        {
            let len = reader.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                l.push(read_g1(&mut reader)?);
            }
        }

        {
            let len = reader.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                a.push(read_g1(&mut reader)?);
            }
        }

        {
            let len = reader.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                b_g1.push(read_g1(&mut reader)?);
            }
        }

        {
            let len = reader.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                b_g2.push(read_g2(&mut reader)?);
            }
        }

        Ok(Parameters {
            vk,
            h: Arc::new(h),
            l: Arc::new(l),
            a: Arc::new(a),
            b_g1: Arc::new(b_g1),
            b_g2: Arc::new(b_g2),
        })
    }
}

#[derive(Clone)]
pub struct MappedParameters<E: Engine> {
    // The parameter file we're reading from.  This is stored so that
    // each (bulk) access can re-map the file, rather than trying to
    // be clever and keeping a persistent memory map around.  This is
    // a much safer way to go (as mmap life-times and consistency
    // guarantees are difficult), and the cost of the mappings should
    // not outweigh the benefits of lazy-loading parameters.
    param_file: PathBuf,

    // This is always loaded (i.e. not lazily loaded).
    pub vk: VerifyingKey<E>,

    // Elements of the form ((tau^i * t(tau)) / delta) for i between 0 and
    // m-2 inclusive. Never contains points at infinity.
    pub h: Vec<Range<usize>>,

    // Elements of the form (beta * u_i(tau) + alpha v_i(tau) + w_i(tau)) / delta
    // for all auxiliary inputs. Variables can never be unconstrained, so this
    // never contains points at infinity.
    pub l: Vec<Range<usize>>,

    // QAP "A" polynomials evaluated at tau in the Lagrange basis. Never contains
    // points at infinity: polynomials that evaluate to zero are omitted from
    // the CRS and the prover can deterministically skip their evaluation.
    pub a: Vec<Range<usize>>,

    // QAP "B" polynomials evaluated at tau in the Lagrange basis. Needed in
    // G1 and G2 for C/B queries, respectively. Never contains points at
    // infinity for the same reason as the "A" polynomials.
    pub b_g1: Vec<Range<usize>>,
    pub b_g2: Vec<Range<usize>>,

    checked: bool,

    _e: PhantomData<E>,
}

unsafe impl<E: Engine> Sync for MappedParameters<E> {}

pub struct PreparedVerifyingKey<E: Engine> {
    /// Pairing result of alpha*beta
    alpha_g1_beta_g2: E::Fqk,
    /// -gamma in G2
    neg_gamma_g2: <E::G2Affine as PairingCurveAffine>::Prepared,
    /// -delta in G2
    neg_delta_g2: <E::G2Affine as PairingCurveAffine>::Prepared,
    /// Copy of IC from `VerifiyingKey`.
    ic: Vec<E::G1Affine>,
}

pub trait ParameterSource<E: Engine> {
    type G1Builder: SourceBuilder<E::G1Affine>;
    type G2Builder: SourceBuilder<E::G2Affine>;

    fn get_vk(&mut self, num_ic: usize) -> Result<VerifyingKey<E>, SynthesisError>;
    fn get_h(&mut self, num_h: usize) -> Result<Self::G1Builder, SynthesisError>;
    fn get_l(&mut self, num_l: usize) -> Result<Self::G1Builder, SynthesisError>;
    fn get_a(
        &mut self,
        num_inputs: usize,
        num_aux: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError>;
    fn get_b_g1(
        &mut self,
        num_inputs: usize,
        num_aux: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError>;
    fn get_b_g2(
        &mut self,
        num_inputs: usize,
        num_aux: usize,
    ) -> Result<(Self::G2Builder, Self::G2Builder), SynthesisError>;
}

impl<'a, E: Engine> ParameterSource<E> for &'a Parameters<E> {
    type G1Builder = (Arc<Vec<E::G1Affine>>, usize);
    type G2Builder = (Arc<Vec<E::G2Affine>>, usize);

    fn get_vk(&mut self, _: usize) -> Result<VerifyingKey<E>, SynthesisError> {
        Ok(self.vk.clone())
    }

    fn get_h(&mut self, _: usize) -> Result<Self::G1Builder, SynthesisError> {
        Ok((self.h.clone(), 0))
    }

    fn get_l(&mut self, _: usize) -> Result<Self::G1Builder, SynthesisError> {
        Ok((self.l.clone(), 0))
    }

    fn get_a(
        &mut self,
        num_inputs: usize,
        _: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        Ok(((self.a.clone(), 0), (self.a.clone(), num_inputs)))
    }

    fn get_b_g1(
        &mut self,
        num_inputs: usize,
        _: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        Ok(((self.b_g1.clone(), 0), (self.b_g1.clone(), num_inputs)))
    }

    fn get_b_g2(
        &mut self,
        num_inputs: usize,
        _: usize,
    ) -> Result<(Self::G2Builder, Self::G2Builder), SynthesisError> {
        Ok(((self.b_g2.clone(), 0), (self.b_g2.clone(), num_inputs)))
    }
}

// A re-usable method for parameter loading via mmap.
fn read_g1<E: Engine>(
    mmap: &Mmap,
    start: usize,
    end: usize,
    checked: bool,
) -> Result<E::G1Affine, std::io::Error> {
    let ptr = &mmap[start..end];
    let repr =
        unsafe { *(ptr as *const [u8] as *const <E::G1Affine as CurveAffine>::Uncompressed) };

    if checked {
        repr.into_affine()
    } else {
        repr.into_affine_unchecked()
    }
    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    .and_then(|e| {
        if e.is_zero() {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "point at infinity",
            ))
        } else {
            Ok(e)
        }
    })
}

// A re-usable method for parameter loading via mmap.
fn read_g2<E: Engine>(
    mmap: &Mmap,
    start: usize,
    end: usize,
    checked: bool,
) -> Result<E::G2Affine, std::io::Error> {
    let ptr = &mmap[start..end];
    let repr =
        unsafe { *(ptr as *const [u8] as *const <E::G2Affine as CurveAffine>::Uncompressed) };

    if checked {
        repr.into_affine()
    } else {
        repr.into_affine_unchecked()
    }
    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    .and_then(|e| {
        if e.is_zero() {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "point at infinity",
            ))
        } else {
            Ok(e)
        }
    })
}

impl<'a, E: Engine> ParameterSource<E> for &'a MappedParameters<E> {
    type G1Builder = (Arc<Vec<E::G1Affine>>, usize);
    type G2Builder = (Arc<Vec<E::G2Affine>>, usize);

    fn get_vk(&mut self, _: usize) -> Result<VerifyingKey<E>, SynthesisError> {
        Ok(self.vk.clone())
    }

    fn get_h(&mut self, _num_h: usize) -> Result<Self::G1Builder, SynthesisError> {
        let params = File::open(self.param_file.clone())?;
        let mmap = unsafe { MmapOptions::new().map(&params)? };

        let mut h = vec![];
        for i in 0..self.h.len() {
            h.push(read_g1::<E>(
                &mmap,
                self.h[i].start,
                self.h[i].end,
                self.checked,
            )?);
        }

        Ok((Arc::new(h), 0))
    }

    fn get_l(&mut self, _num_l: usize) -> Result<Self::G1Builder, SynthesisError> {
        let params = File::open(self.param_file.clone())?;
        let mmap = unsafe { MmapOptions::new().map(&params)? };

        let mut l = vec![];
        for i in 0..self.l.len() {
            l.push(read_g1::<E>(
                &mmap,
                self.l[i].start,
                self.l[i].end,
                self.checked,
            )?);
        }

        Ok((Arc::new(l), 0))
    }

    fn get_a(
        &mut self,
        num_inputs: usize,
        _num_a: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        let params = File::open(self.param_file.clone())?;
        let mmap = unsafe { MmapOptions::new().map(&params)? };

        let mut a = vec![];
        for i in 0..self.a.len() {
            a.push(read_g1::<E>(
                &mmap,
                self.a[i].start,
                self.a[i].end,
                self.checked,
            )?);
        }

        Ok(((Arc::new(a.clone()), 0), (Arc::new(a), num_inputs)))
    }

    fn get_b_g1(
        &mut self,
        num_inputs: usize,
        _num_b_g1: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        let params = File::open(self.param_file.clone())?;
        let mmap = unsafe { MmapOptions::new().map(&params)? };

        let mut b_g1 = vec![];
        for i in 0..self.b_g1.len() {
            b_g1.push(read_g1::<E>(
                &mmap,
                self.b_g1[i].start,
                self.b_g1[i].end,
                self.checked,
            )?);
        }

        Ok(((Arc::new(b_g1.clone()), 0), (Arc::new(b_g1), num_inputs)))
    }

    fn get_b_g2(
        &mut self,
        num_inputs: usize,
        _num_b_g2: usize,
    ) -> Result<(Self::G2Builder, Self::G2Builder), SynthesisError> {
        let params = File::open(self.param_file.clone())?;
        let mmap = unsafe { MmapOptions::new().map(&params)? };

        let mut b_g2 = vec![];
        for i in 0..self.b_g2.len() {
            b_g2.push(read_g2::<E>(
                &mmap,
                self.b_g2[i].start,
                self.b_g2[i].end,
                self.checked,
            )?);
        }

        Ok(((Arc::new(b_g2.clone()), 0), (Arc::new(b_g2), num_inputs)))
    }
}

#[cfg(test)]
mod test_with_bls12_381 {
    use super::*;
    use crate::{Circuit, ConstraintSystem, SynthesisError};

    use ff::Field;
    use paired::bls12_381::{Bls12, Fr};
    use rand::thread_rng;

    #[test]
    fn serialization() {
        struct MySillyCircuit<E: Engine> {
            a: Option<E::Fr>,
            b: Option<E::Fr>,
        }

        impl<E: Engine> Circuit<E> for MySillyCircuit<E> {
            fn synthesize<CS: ConstraintSystem<E>>(
                self,
                cs: &mut CS,
            ) -> Result<(), SynthesisError> {
                let a = cs.alloc(|| "a", || self.a.ok_or(SynthesisError::AssignmentMissing))?;
                let b = cs.alloc(|| "b", || self.b.ok_or(SynthesisError::AssignmentMissing))?;
                let c = cs.alloc_input(
                    || "c",
                    || {
                        let mut a = self.a.ok_or(SynthesisError::AssignmentMissing)?;
                        let b = self.b.ok_or(SynthesisError::AssignmentMissing)?;

                        a.mul_assign(&b);
                        Ok(a)
                    },
                )?;

                cs.enforce(|| "a*b=c", |lc| lc + a, |lc| lc + b, |lc| lc + c);

                Ok(())
            }
        }

        let rng = &mut thread_rng();

        let params =
            generate_random_parameters::<Bls12, _, _>(MySillyCircuit { a: None, b: None }, rng)
                .unwrap();

        {
            let mut v = vec![];

            params.write(&mut v).unwrap();
            assert_eq!(v.len(), 2136);

            let de_params = Parameters::read(&v[..], true).unwrap();
            assert!(params == de_params);

            let de_params = Parameters::read(&v[..], false).unwrap();
            assert!(params == de_params);
        }

        let pvk = prepare_verifying_key::<Bls12>(&params.vk);

        for _ in 0..100 {
            let a = Fr::random(rng);
            let b = Fr::random(rng);
            let mut c = a;
            c.mul_assign(&b);

            let proof = create_random_proof(
                MySillyCircuit {
                    a: Some(a),
                    b: Some(b),
                },
                &params,
                rng,
            )
            .unwrap();

            let mut v = vec![];
            proof.write(&mut v).unwrap();

            assert_eq!(v.len(), 192);

            let de_proof = Proof::read(&v[..]).unwrap();
            assert!(proof == de_proof);

            assert!(verify_proof(&pvk, &proof, &[c]).unwrap());
            assert!(!verify_proof(&pvk, &proof, &[a]).unwrap());
        }
    }
}
